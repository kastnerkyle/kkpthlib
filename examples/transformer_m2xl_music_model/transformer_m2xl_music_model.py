from __future__ import print_function
import os
import sys

import numpy as np
import torch
from torch import nn
import torch.functional as F

from kkpthlib import AWDTransformerXLDecoderBlock
from kkpthlib import HParams
from kkpthlib import Linear
from kkpthlib import relu
from kkpthlib import EmbeddingDropout
from kkpthlib import WordCorpus
from kkpthlib import make_batches_from_list
from kkpthlib import StepIterator
from kkpthlib import CategoricalCrossEntropyFromLogits
from kkpthlib import clipping_grad_norm_
from kkpthlib import clipping_grad_value_
from kkpthlib import RampOpt
from kkpthlib import run_loop

from kkpthlib import fetch_jsb_chorales
from kkpthlib import MusicJSONFlatKeyframeMeasureCorpus
from kkpthlib import convert_voice_lists_to_music_json
from kkpthlib import music_json_to_midi
from kkpthlib import write_music_json

hp = HParams(memory_len=20,
             context_len=64,
             max_sequence_length=256,
             transformer_input_dim=380,
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             learning_rate=1E-4,
             min_learning_rate=1E-6,
             clip=.25,
             batch_size=12,
             n_layers=24,
             embedding_dropout_keep_prob=0.8,
             attention_dropout_keep_prob=0.8,
             input_dropout_keep_prob=0.4,
             inner_dropout_keep_prob=0.8,
             hidden_dropout_keep_prob=1.0,
             output_dropout_keep_prob=0.5,
             random_seed=2122)


def get_hparams():
    return hp

def build_model(hp, file_corpus):
    random_state = np.random.RandomState(hp.random_seed)
    class Model(nn.Module):
        def __init__(self, hp, this_file_corpus):
            super(Model, self).__init__()

            '''
            [fingerprint_features_zero, : 0
            fingerprint_features_one, : 1
            duration_features_zero, : 2
            duration_features_mid, : 3
            duration_features_one, : 4
            voices, : 5
            centers, : 6
            key_zero_base : 7
            key_zero, : 8
            key_durations_zero, : 9
            key_one_base : 10
            key_one, : 11
            key_durations_one, : 12
            key_indicators, : 13
            targets : 14]
            '''
            # cut the features into chunks then feed without indicators
            # this is ... messy but necessary

            # probably a better way than handcoding the relationships here but w/e
            used_features = [0, 1, 2, 3, 4, 5, 7, 8, 9, "target0", "target1", "target2", "target3"]
            self.used_features = used_features
            self.embeddings = nn.ModuleList()
            for _n, elem in enumerate(used_features):
                inp = 128 #3 * hp.transformer_input_dim // len(used_features)
                # just project down to transformer_input_dim size
                #if _n == len(used_features) - 1:
                #    used_dim = inp * (len(used_features) - 1)
                #    leftover_dim = hp.transformer_input_dim - used_dim
                #    inp = leftover_dim

                if elem == 0:
                    vocab_size = len(this_file_corpus.fingerprint_features_zero_dictionary.idx2word)
                elif elem == 1:
                    vocab_size = len(this_file_corpus.fingerprint_features_one_dictionary.idx2word)
                elif elem == 2:
                    vocab_size = len(this_file_corpus.duration_features_zero_dictionary.idx2word)
                elif elem == 3:
                    vocab_size = len(this_file_corpus.duration_features_mid_dictionary.idx2word)
                elif elem == 4:
                    vocab_size = len(this_file_corpus.duration_features_one_dictionary.idx2word)
                elif elem == 5:
                    vocab_size = len(this_file_corpus.voices_dictionary.idx2word)
                elif elem == 7:
                    vocab_size = len(this_file_corpus.keypoint_base_dictionary.idx2word)
                elif elem == 8:
                    vocab_size = len(this_file_corpus.keypoint_dictionary.idx2word)
                elif elem == 9:
                    vocab_size = len(this_file_corpus.keypoint_durations_dictionary.idx2word)
                elif elem == "target0":
                    vocab_size = len(this_file_corpus.target_0_dictionary.idx2word)
                elif elem == "target1":
                    vocab_size = len(this_file_corpus.target_1_dictionary.idx2word)
                elif elem == "target2":
                    vocab_size = len(this_file_corpus.target_2_dictionary.idx2word)
                elif elem == "target3":
                    vocab_size = len(this_file_corpus.target_3_dictionary.idx2word)
                else:
                    raise ValueError("Unhandled embedding element {}".format(elem))
                emb = EmbeddingDropout(vocab_size,
                                       inp,
                                       dropout_keep_prob=hp.embedding_dropout_keep_prob,
                                       random_state=random_state,
                                       device=hp.use_device,
                                       name="embed_{}".format(elem))
                self.embeddings.append(emb)
            self.in_proj = Linear([inp] * len(used_features),
                                   hp.transformer_input_dim,
                                   random_state=random_state,
                                   device=hp.use_device,
                                   init="normal",
                                   scale=0.02,
                                   name="model_in")

            self.transformer = AWDTransformerXLDecoderBlock([hp.transformer_input_dim],
                                                            name="transformer_block",
                                                            random_state=random_state,
                                                            memory_len=hp.memory_len, context_len=hp.context_len, n_layers=hp.n_layers, input_dropout_keep_prob=hp.input_dropout_keep_prob,
                                                            attention_dropout_keep_prob=hp.attention_dropout_keep_prob,
                                                            inner_dropout_keep_prob=hp.inner_dropout_keep_prob,
                                                            hidden_dropout_keep_prob=hp.hidden_dropout_keep_prob,
                                                            output_dropout_keep_prob=hp.output_dropout_keep_prob,
                                                            init="normal",
                                                            scale=0.02,
                                                            device=hp.use_device)

            self.out_proj0 = Linear([hp.transformer_input_dim],
                                    len(this_file_corpus.target_0_dictionary.idx2word),
                                    random_state=random_state,
                                    device=hp.use_device,
                                    init="normal",
                                    scale=0.02,
                                    name="model_out0")
            self.out_proj1 = Linear([hp.transformer_input_dim],
                                    len(this_file_corpus.target_1_dictionary.idx2word),
                                    random_state=random_state,
                                    device=hp.use_device,
                                    init="normal",
                                    scale=0.02,
                                    name="model_out1")
            self.out_proj2 = Linear([hp.transformer_input_dim],
                                    len(this_file_corpus.target_2_dictionary.idx2word),
                                    random_state=random_state,
                                    device=hp.use_device,
                                    init="normal",
                                    scale=0.02,
                                    name="model_out2")
            self.out_proj3 = Linear([hp.transformer_input_dim],
                                    len(this_file_corpus.target_3_dictionary.idx2word),
                                    random_state=random_state,
                                    device=hp.use_device,
                                    init="normal",
                                    scale=0.02,
                                    name="model_out3")

        def forward(self, list_of_input_batches, list_of_input_batch_masks, list_of_mems=None):
            all_e = []
            for _i in range(len(list_of_input_batches)):
                e, d_e = self.embeddings[_i](list_of_input_batches[_i])
                all_e.append(e)
            x1 = relu(self.in_proj(all_e))
            # right now assume all batch masks are the same...
            out, l_o_m = self.transformer(x1, list_of_input_batch_masks[0], list_of_mems=list_of_mems)
            p0 = self.out_proj0([out])
            p1 = self.out_proj1([out])
            p2 = self.out_proj2([out])
            p3 = self.out_proj3([out])
            return p0, p1, p2, p3, l_o_m
    return Model(hp, file_corpus).to(hp.use_device)

if __name__ == "__main__":
    jsb = fetch_jsb_chorales()

    # sort into minor / major, then by key
    all_transposed = sorted([f for f in jsb["files"] if "original" not in f], key=lambda x:
                            (x.split(os.sep)[-1].split(".")[-2].split("transposed")[0].split("-")[0],
                             x.split(os.sep)[-1].split(".")[-2].split("transposed")[0].split("-")[1]))

    bwv_names = sorted(list(set([f.split(os.sep)[-1].split(".")[0] for f in all_transposed])))
    vrng = np.random.RandomState(144)
    vrng.shuffle(bwv_names)

    # 15 is ~5% of the data if you hold out all 12 transpositions
    # holding out whole songs so actually a pretty hard validation set...
    valid_names = bwv_names[:15]
    train_files = [f for f in all_transposed if all([vn not in f for vn in valid_names])]
    valid_files = [f for f in all_transposed if any([vn in f for vn in valid_names])]
    assert all([vf not in train_files for vf in valid_files])

    filepath = "train_trainfiles.txt"
    with open(filepath, 'w') as file_handler:
        for item in sorted(train_files):
            file_handler.write("{}\n".format(item))

    filepath = "train_validfiles.txt"
    with open(filepath, 'w') as file_handler:
        for item in sorted(valid_files):
            file_handler.write("{}\n".format(item))

    # shuffle the train and valid files before we make the flat_measure_corpus
    vrng.shuffle(train_files)
    vrng.shuffle(valid_files)
    flat_measure_corpus = MusicJSONFlatKeyframeMeasureCorpus(train_data_file_paths=train_files,
                                                             valid_data_file_paths=valid_files)
    # need to handle measure boundaries
    break_id = flat_measure_corpus.voices_dictionary.word2idx[9999]
    train_cut_points = [el == break_id for el in flat_measure_corpus.train[5]]
    valid_cut_points = [el == break_id for el in flat_measure_corpus.valid[5]]
    train_batch_indices = make_batches_from_list(np.arange(len(flat_measure_corpus.train[0])), batch_size=hp.batch_size,
                                                 sequence_length=hp.max_sequence_length, overlap=hp.context_len,
                                                 cut_points=train_cut_points,
                                                 fill_value=-1)
    valid_batch_indices = make_batches_from_list(np.arange(len(flat_measure_corpus.valid[0])), batch_size=hp.batch_size,
                                                 sequence_length=hp.max_sequence_length, overlap=hp.context_len,
                                                 cut_points=valid_cut_points,
                                                 fill_value=-1)

    train_random_state = np.random.RandomState(hp.random_seed)
    valid_random_state = np.random.RandomState(hp.random_seed + 1)
    gen_random_state = np.random.RandomState(hp.random_seed + 2)

    train_itr = StepIterator([train_batch_indices], circular_rotation=True, random_state=train_random_state)
    valid_itr = StepIterator([valid_batch_indices], random_state=valid_random_state)

    # need this for now
    def make_batches_from_indices(data_lists, indices_batch,
                                  fill_value=-1):
        # use transformer convention, 0s unmasked 1s masked
        all_batches = []
        all_batches_masks = []
        for i in range(len(data_lists)):
            # currently works because fill value is -1
            tb = [[[data_lists[i][el] for el in indices_batch[:, b]] for b in range(indices_batch.shape[1])]]
            tb_mask = [[[el for el in indices_batch[:, b]] for b in range(indices_batch.shape[1])]]
            # has dim of 1, slice it temporarily then put it back
            tb = np.array(tb)
            if len(tb.shape) == 4:
                # edge case for 3d features
                tb = tb[0]
                tb = np.transpose(tb, (1, 0, 2))
            elif len(tb.shape) == 3:
                tb = np.transpose(tb, (2, 1, 0))
            elif len(tb.shape) == 2:
                tb = tb.T[:, :, None]
            # zero out everything that was equal to fill_value?
            batch_mask = np.array(tb_mask)
            batch_mask = (batch_mask == fill_value).astype(np.int32)
            # remove spurious axis, swap to seq len, batch
            batch_mask = batch_mask[0].T
            all_batches.append(tb)
            all_batches_masks.append(batch_mask)
        return all_batches, all_batches_masks

    '''
    train_indices = next(train_itr)
    valid_indices = next(valid_itr)
    train_batches_list, train_batches_masks_list = make_batches_from_indices(flat_measure_corpus.train, train_indices)
    valid_batches_list, valid_batches_masks_list = make_batches_from_indices(flat_measure_corpus.valid, valid_indices)
    # ignore it for first pass but consider investigating

    # need to write code to put the audio back together again from given or predictable info
    midi_sample_dir = "midi_samples"
    if not os.path.exists(midi_sample_dir):
        os.mkdir(midi_sample_dir)

    """
    [fingerprint_features_zero, : 0
    fingerprint_features_one, : 1
    duration_features_zero, : 2
    duration_features_mid, : 3
    duration_features_one, : 4
    voices, : 5
    centers, : 6
    key_zero_base, : 7
    key_zero, : 8
    key_durations_zero, : 9
    key_one_base, : 10
    key_one, : 11
    key_durations_one, : 12
    key_indicators, : 13
    targets] : 14
    """
    # do we just make a function on the original data class?
    for i in range(train_batches_list[0].shape[1]):
        # first get back the "left" keypoint
        # as well as the duration
        key_zero_base = train_batches_list[7]
        key_zero = train_batches_list[8]
        key_durations_zero = train_batches_list[9]
        
        key_one_base = train_batches_list[10]
        key_one = train_batches_list[11]
        key_durations_one = train_batches_list[12]

        key_indicators = train_batches_list[13]

        # same mask for all of em
        this_mask = train_batches_masks_list[0][:, i]
        f_m = np.where(this_mask)[0][0]

        key_zero_base = key_zero_base[:f_m, i, 0]
        key_zero = key_zero[:f_m, i, 0]
        key_durations_zero = key_durations_zero[:f_m, i, 0]

        key_one_base = key_one_base[:f_m, i, 0]
        key_one = key_one[:f_m, i, 0]
        key_durations_one = key_durations_one[:f_m, i, 0]
        key_indicators = key_indicators[:f_m, i, 0]

        boundary_points = np.concatenate((np.array([-1]), key_indicators))[:-1] != key_indicators
        s_s = np.concatenate((np.where(boundary_points)[0], np.array([len(key_indicators)])))
        boundary_pairs = list(zip(s_s[:-1], s_s[1:]))

        pitches_list = []
        durations_list = []
        voices_list = []
        for s, e in boundary_pairs:
            # in each chunk, do keypoint vector, then rest
            this_key = key_zero[s:e]
            assert all([tk == this_key[0] for tk in this_key])
            this_key = flat_measure_corpus.keypoint_dictionary.idx2word[this_key[0]]

            this_key_base = key_zero_base[s:e]
            assert all([tkb == this_key_base[0] for tkb in this_key_base])
            this_key = tuple([tk + flat_measure_corpus.keypoint_base_dictionary.idx2word[this_key_base[0]] for tk in this_key])

            this_key_durations = key_durations_zero[s:e]
            assert all([tkd == this_key_durations[0] for tkd in this_key_durations])
            this_key_durations = flat_measure_corpus.keypoint_durations_dictionary.idx2word[this_key_durations[0]]

            centers = train_batches_list[6]
            centers = centers[s:e, i]
            center_0 = flat_measure_corpus.centers_0_dictionary.idx2word[centers[0][0]]
            center_1 = flat_measure_corpus.centers_1_dictionary.idx2word[centers[0][1]]
            center_2 = flat_measure_corpus.centers_2_dictionary.idx2word[centers[0][2]]
            center_3 = flat_measure_corpus.centers_3_dictionary.idx2word[centers[0][3]]

            targets = train_batches_list[-1]
            targets = targets[s:e, i]
            target_0_values = [flat_measure_corpus.target_0_dictionary.idx2word[targets[z][0]] for z in range(len(targets))]
            target_1_values = [flat_measure_corpus.target_1_dictionary.idx2word[targets[z][1]] for z in range(len(targets))]
            target_2_values = [flat_measure_corpus.target_2_dictionary.idx2word[targets[z][2]] for z in range(len(targets))]
            target_3_values = [flat_measure_corpus.target_3_dictionary.idx2word[targets[z][3]] for z in range(len(targets))]

            # 100 was rest
            remapped_0 = [this_key[0] + t_0 if t_0 != 100 else 0 for t_0 in target_0_values]
            remapped_1 = [this_key[1] + t_1 if t_1 != 100 else 0 for t_1 in target_1_values]
            remapped_2 = [this_key[2] + t_2 if t_2 != 100 else 0 for t_2 in target_2_values]
            remapped_3 = [this_key[3] + t_3 if t_3 != 100 else 0 for t_3 in target_3_values]

            assert all([remapped_0[n] == remapped_1[n] for n in range(len(remapped_0))])
            assert all([remapped_0[n] == remapped_2[n] for n in range(len(remapped_0))])
            assert all([remapped_0[n] == remapped_3[n] for n in range(len(remapped_0))])

            durations = train_batches_list[3]
            durations = durations[s:e, i, 0]
            duration_values = [flat_measure_corpus.duration_features_mid_dictionary.idx2word[d_el] for d_el in durations]

            voices = train_batches_list[5]
            voices = voices[s:e, i, 0]
            voice_values = [flat_measure_corpus.voices_dictionary.idx2word[v_el] for v_el in voices]

            final_pitch_chunk = []
            final_duration_chunk = []
            final_voice_chunk = []
            key_itr = 0
            last_v = -1
            for n, v in enumerate(voice_values):
                # we assume it is SSSSSSAAAAAAAAAAAATTTTTTTBB
                # style format
                # need to insert key values at the right spot
                if v != last_v:
                    final_pitch_chunk.append(this_key[key_itr])
                    final_duration_chunk.append(this_key_durations[key_itr])
                    final_voice_chunk.append(key_itr)

                    key_itr += 1
                    last_v = v

                final_pitch_chunk.append(int(remapped_0[n]))
                final_duration_chunk.append(duration_values[n])
                final_voice_chunk.append(v)
            pitches_list.extend(final_pitch_chunk)
            durations_list.extend(final_duration_chunk)
            voices_list.extend(final_voice_chunk)

        data = convert_voice_lists_to_music_json(pitch_lists=pitches_list, duration_lists=durations_list, voices_list=voices_list)
        json_fpath = midi_sample_dir + os.sep + "true{}.json".format(i)
        write_music_json(data, json_fpath)

        fpath = midi_sample_dir + os.sep + "true{}.midi".format(i)
        music_json_to_midi(data, fpath)
        print("Wrote out {}".format(fpath))
    from IPython import embed; embed(); raise ValueError()
    '''

    model = build_model(hp, flat_measure_corpus)
    loss_fun = CategoricalCrossEntropyFromLogits()

    def get_std_ramp_opt(model):
        return RampOpt(hp.learning_rate, 1, 3000, 3000 + 125 * 450,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1E-9),
                       min_decay_learning_rate=hp.min_learning_rate)

    optimizer = get_std_ramp_opt(model)

    first_valid = True
    first_train = True

    def loop(itr, extras, stateful_args):
        global first_valid
        global first_train
        indices = next(itr)
        if extras["train"]:
            batches_list, batches_masks_list = make_batches_from_indices(flat_measure_corpus.train, indices)
        else:
            batches_list, batches_masks_list = make_batches_from_indices(flat_measure_corpus.valid, indices)
        '''
        [fingerprint_features_zero, : 0
        fingerprint_features_one, : 1
        duration_features_zero, : 2
        duration_features_mid, : 3
        duration_features_one, : 4
        voices, : 5
        centers, : 6
        key_zero_base : 7
        key_zero, : 8
        key_durations_zero, : 9
        key_one_base : 10
        key_one, : 11
        key_durations_one, : 12
        key_indicators, : 13
        targets : 14]
        '''
        # cut the features into chunks then feed without indicators
        # this is ... messy but necessary

        used_features = [0, 1, 2, 3, 4, 5, 7, 8, 9]
        feature_batches_list = [torch.tensor(b).to(hp.use_device) for n, b in enumerate(batches_list) if n in used_features]
        feature_batches_masks_list = [torch.tensor(mb).to(hp.use_device) for n, mb in enumerate(batches_masks_list) if n in used_features]
        # AFTER SANITY CHECK, SHIFTED TARGETS BECOME INPUTS TOO
        # 4 sets of targets, alternate definiteions of the same value
        # distance from kp0_0 etc
        np_targets = batches_list[-1]

        list_of_inputs = feature_batches_list
        list_of_input_masks = feature_batches_masks_list
        targets = torch.tensor(np_targets).long().to(hp.use_device)

        # shift by one to account for the fact that keypoint features and targets are identical
        for _i in range(len(list_of_inputs)):
             list_of_inputs[_i] = torch.cat((list_of_inputs[_i][:1] * 0 + 0, list_of_inputs[_i]), 0)
             list_of_input_masks[_i] = torch.cat((list_of_input_masks[_i][:1] * 0 + 0, list_of_input_masks[_i]), 0)

        # shift targets by 1 and pass as inputs as well
        shifted_targets = torch.cat((targets[:1] * 0 + 0, targets), 0)
        # pad by 1 at the back to make targets match
        targets = torch.cat((targets, targets[:1] * 0 + 0), 0)

        list_of_inputs.extend([shifted_targets[:, :, 0][..., None],
                               shifted_targets[:, :, 1][..., None],
                               shifted_targets[:, :, 2][..., None],
                               shifted_targets[:, :, 3][..., None]])

        in_mems = stateful_args
        if extras["train"]:
            if first_train:
                model.train()
                out0, out1, out2, out3, out_mems = model(list_of_inputs, list_of_input_masks, list_of_mems=None)
                first_train = False
                first_valid = True
            else:
                model.train()
                out0, out1, out2, out3, out_mems = model(list_of_inputs, list_of_input_masks, list_of_mems=in_mems)
        else:
            if first_valid:
                model.eval()
                out0, out1, out2, out3, out_mems = model(list_of_inputs, list_of_input_masks, list_of_mems=None)
                first_valid = False
                first_train = True
            else:
                model.eval()
                out0, out1, out2, out3, out_mems = model(list_of_inputs, list_of_input_masks, list_of_mems=in_mems)


        # inputs have already been cut to context length inside transformer
        loss0 = loss_fun(out0, targets[..., :, 0][..., None])
        loss1 = loss_fun(out1, targets[..., :, 1][..., None])
        loss2 = loss_fun(out2, targets[..., :, 2][..., None])
        loss3 = loss_fun(out3, targets[..., :, 3][..., None])
        #loss = (loss0 + loss1 + loss2 + loss3) / 4.
        loss = loss0 + 0. * loss1 + 0. * loss2 + 0. * loss3
        target_masks = feature_batches_masks_list[0]
        # masks use transformer convention 0 if valid, 1 if invalid
        loss = loss * (1. - target_masks)
        loss = loss.sum() / target_masks.sum()

        l = loss.cpu().data.numpy()
        optimizer.zero_grad()
        if extras["train"]:
            loss.backward()
            clipping_grad_norm_(model.parameters(), hp.clip)
            optimizer.step()
        else:
            pass
            # don't carry over valid memories into train, just copy the train ones over
            # this means our validation score in training will be a bit off, but the memory
            # during training should be OK
            #out_mems = in_mems
        return l, None, out_mems

    """
    # the out-of-loop-check
    rs = []
    for i in range(5):
        print(i)
        if i == 0:
            r = loop(train_itr, {"train": True}, None)
        else:
            r = loop(train_itr, {"train": True}, rs[-1][-1])
        rs.append(r)
    from IPython import embed; embed(); raise ValueError()
    """

    s = {"model": model,
         "optimizer": optimizer,
         "hparams": hp}

    run_loop(loop, train_itr,
             loop, valid_itr,
             s,
             n_train_steps_per=2000,
             n_valid_steps_per=250)
