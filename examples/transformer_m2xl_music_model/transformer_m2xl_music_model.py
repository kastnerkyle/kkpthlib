from __future__ import print_function
import os
import sys

import numpy as np
import torch
from torch import nn
import torch.functional as F

from kkpthlib import AWDXLNetDecoderBlock
from kkpthlib import HParams
from kkpthlib import Linear
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
             learning_rate=1E-5,
             min_learning_rate=1E-6,
             clip=.25,
             batch_size=12,
             n_layers=24,
             mask_K=6,
             max_n_gram=24,
             embedding_dropout_keep_prob=.9,
             input_dropout_keep_prob=1.,
             inner_dropout_keep_prob=1.,
             hidden_dropout_keep_prob=1.,
             attention_dropout_keep_prob=1.,
             output_dropout_keep_prob=.9,

             use_target_mappings=True,

             #embedding_dropout_keep_prob=1.,
             #input_dropout_keep_prob=1.,
             #inner_dropout_keep_prob=1.,
             #hidden_dropout_keep_prob=1.,
             #attention_dropout_keep_prob=1.,
             #output_dropout_keep_prob=1.,

             data_storage_dir="kjv",
             max_vocabulary_size=69,
             random_seed=2122)


def get_hparams():
    return hp

def build_model(hp):
    random_state = np.random.RandomState(hp.random_seed)
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.embedding_pitch_k = EmbeddingDropout(hp.max_vocabulary_size,
                                                      hp.transformer_input_dim,
                                                      dropout_keep_prob=hp.embedding_dropout_keep_prob,
                                                      random_state=random_state,
                                                      device=hp.use_device,
                                                      name="embed_pitch")
            self.embedding_duration_k = EmbeddingDropout(hp.max_vocabulary_size,
                                                         hp.transformer_input_dim,
                                                         dropout_keep_prob=hp.embedding_dropout_keep_prob,
                                                         random_state=random_state,
                                                         device=hp.use_device,
                                                         name="embed_duration")
            self.embedding_voice_k = EmbeddingDropout(hp.max_vocabulary_size,
                                                      hp.transformer_input_dim,
                                                      dropout_keep_prob=hp.embedding_dropout_keep_prob,
                                                      random_state=random_state,
                                                      device=hp.use_device,
                                                      name="embed_voice")
            l = np.sqrt(6. / (hp.transformer_input_dim))
            self.embedding_q = nn.Parameter(torch.tensor(random_state.uniform(-l, l, size=(hp.transformer_input_dim,))))
            self.transformer = AWDXLNetDecoderBlock([hp.transformer_input_dim],
                                                     name="xlnet_block",
                                                     input_dropout_keep_prob=hp.input_dropout_keep_prob,
                                                     attention_dropout_keep_prob=hp.attention_dropout_keep_prob,
                                                     inner_dropout_keep_prob=hp.inner_dropout_keep_prob,
                                                     hidden_dropout_keep_prob=hp.hidden_dropout_keep_prob,
                                                     output_dropout_keep_prob=hp.output_dropout_keep_prob,
                                                     random_state=random_state,
                                                     memory_len=hp.memory_len,
                                                     context_len=hp.context_len,
                                                     n_layers=hp.n_layers,
                                                     init="normal",
                                                     scale=0.02,
                                                     device=hp.use_device)

            self.out_proj = Linear([hp.transformer_input_dim],
                                    hp.max_vocabulary_size,
                                    random_state=random_state,
                                    device=hp.use_device,
                                    init="normal",
                                    scale=0.02,
                                    name="model_out")

        def forward(self, input_ks, input_qs, perm_masks, target_mappings, target_masks, list_of_mems=None):
            xe_pitch_k, de_pitch_k = self.embedding_pitch_k(input_ks[..., 0][..., None])
            xe_duration_k, de_duration_k = self.embedding_duration_k(input_ks[..., 1][..., None])
            xe_voice_k, de_voice_k = self.embedding_voice_k(input_ks[..., -1][..., None])

            xe_base_k = xe_duration_k + xe_voice_k

            xe_full_k = xe_base_k + xe_pitch_k

            if target_mappings == None:
                # use xe_k for size broadcasting
                _q = xe_base_k + self.embedding_q[None, None]

                # everywhere there is a target, blank out the standard embedding and replace with the learned mask emb
                # inside transformer if target_mappings is provided this gets sliced down
                xe_q = target_masks[..., None] * _q + (1. - target_masks[..., None]) * xe_full_k
            else:
                xe_q = torch.einsum("ibf,jib->jbf", xe_base_k, target_mappings) + self.embedding_q[None, None]

            xe_k = xe_full_k

            # debug hooks for grad checks
            #xe_k = xe_k.detach()
            #xe_q = xe_q.detach()

            #self.tmp_k = xe_k
            #self.tmp_k.requires_grad = True
            #self.tmp_q = xe_q
            #self.tmp_q.requires_grad = True

            #xe_k.detach()

            out_h, out_g, l_o_m = self.transformer(xe_k, xe_q,
                                                   perm_masks,
                                                   target_mappings,
                                                   target_masks,
                                                   list_of_mems=list_of_mems)
            p_h = self.out_proj([out_h])
            p_g = self.out_proj([out_g])
            return p_h, p_g, l_o_m
    return Model().to(hp.use_device)

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

    train_indices = next(train_itr)
    valid_indices = next(valid_itr)
    train_batches_list, train_batches_masks_list = make_batches_from_indices(flat_measure_corpus.train, train_indices)
    valid_batches_list, valid_batches_masks_list = make_batches_from_indices(flat_measure_corpus.valid, valid_indices)
    # a few of the target values are... insanely far from the norm
    # ignore it for first pass but consider investigating

    # need to write code to put the audio back together again from given or predictable info
    midi_sample_dir = "midi_samples"
    if not os.path.exists(midi_sample_dir):
        os.mkdir(midi_sample_dir)

    """
    [fingerprint_features_zero,
    fingerprint_features_one,
    duration_features_zero,
    duration_features_mid,
    duration_features_one,
    voices,
    centers,
    key_zero,
    key_durations_zero,
    key_one,
    key_durations_one,
    key_indicators,
    targets]
    """
    # do we just make a function on the original data class?
    for i in range(train_batches_list[0].shape[1]):
        # first get back the "left" keypoint
        # as well as the duration
        key_zero = train_batches_list[-6]
        key_durations_zero = train_batches_list[-5]
        key_one = train_batches_list[-4]
        key_durations_one = train_batches_list[-3]
        key_indicators = train_batches_list[-2]

        # same mask for all of em
        this_mask = train_batches_masks_list[0][:, i]
        f_m = np.where(this_mask)[0][0]

        key_zero = key_zero[:f_m, i, 0]
        key_durations_zero = key_durations_zero[:f_m, i, 0]
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
            this_key_durations = key_durations_zero[s:e]
            assert all([tkd == this_key_durations[0] for tkd in this_key_durations])
            this_key_durations = flat_measure_corpus.keypoint_durations_dictionary.idx2word[this_key_durations[0]]

            centers = train_batches_list[-7]
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
            remapped_0 = [center_0 + t_0 if t_0 != 100 else 0 for t_0 in target_0_values]
            remapped_1 = [center_1 + t_1 if t_1 != 100 else 0 for t_1 in target_1_values]
            remapped_2 = [center_2 + t_2 if t_2 != 100 else 0 for t_2 in target_2_values]
            remapped_3 = [center_3 + t_3 if t_3 != 100 else 0 for t_3 in target_3_values]

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

    model = build_model(hp)
    loss_fun = CategoricalCrossEntropyFromLogits()

    def get_std_ramp_opt(model):
        return RampOpt(hp.learning_rate, 1, 3000, 3000 + 125 * 450,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1E-9),
                       min_decay_learning_rate=hp.min_learning_rate)

    optimizer = get_std_ramp_opt(model)

    def loop(itr, extras, stateful_args):
        np_data = next(itr)
        # split it, use same mask for all. need to figure out ks, qs, etc
        np_pitch_data = np_data[..., 0]
        np_duration_data = np_data[..., 1]
        np_velocity_data = np_data[..., 2]
        np_voice_data = np_data[..., 3]

        #np_perm_masks, np_target_mappings, np_target_masks, np_input_ks, np_input_qs, np_targets, np_perm_orders = model.transformer.make_inputs_targets_masks_and_mappings(np_pitch_data, K=6, context_cut=hp.context_len, random_state=gen_random_state)
        np_perm_masks, np_target_mappings, np_target_masks, _, _, _, np_perm_orders = model.transformer.make_inputs_targets_masks_and_mappings(np_pitch_data, K=hp.mask_K, max_n_gram=hp.max_n_gram, context_cut=hp.context_len, random_state=gen_random_state)
        #qs are target_mask
        #ks are input data
        np_targets = np_pitch_data[:-1]
        # will have to split this up
        np_input_ks = np_data[:-1]
        np_input_qs = np_target_masks

        # we don't use input ks and qs here, whatever masks and mappings we sample first (over the target domain, pitch) will be used for all
        if hp.use_target_mappings:
            old = np.copy(np_targets)
            old_subs = [old[np.where(np_target_masks[:, i])[0], i][None] for i in range(hp.batch_size)]
            np_targets = np.concatenate(old_subs).transpose(1, 0)
            # all 1s mask
            np_target_masks = 0. * np_targets + 1.

            pad_l = np.zeros((hp.context_len, hp.batch_size))
            np_targets = np.concatenate((pad_l, np_targets))
            np_target_masks = np.concatenate((pad_l, np_target_masks))
            # assume len(np_input_ks) == orig len(targets)
            pad_r = np.zeros((len(np_input_ks) - len(np_targets), hp.batch_size))
            np_targets = np.concatenate((np_targets, pad_r))
            np_target_masks = np.concatenate((np_target_masks, pad_r))
        else:
            np_target_mappings = None

        input_ks = torch.tensor(np_input_ks).to(hp.use_device)
        input_qs = torch.tensor(np_input_qs).to(hp.use_device)
        targets = torch.tensor(np_targets).long().to(hp.use_device)

        #input_ks = input_ks[..., None]
        input_qs = input_qs[..., None]
        targets = targets[..., None]

        perm_masks = torch.tensor(np_perm_masks).to(hp.use_device)
        target_masks = torch.tensor(np_target_masks).to(hp.use_device)

        if np_target_mappings is not None:
            target_mappings = torch.tensor(np_target_mappings).to(hp.use_device)
        else:
            target_mappings = None

        in_mems = stateful_args
        if extras["train"]:
            out_h, out_g, out_mems = model(input_ks, input_qs, perm_masks, target_mappings, target_masks, list_of_mems=in_mems)
        else:
            out_h, out_g, out_mems = model(input_ks, input_qs, perm_masks, target_mappings, target_masks, list_of_mems=None)

        if target_mappings is None:
            loss = loss_fun(out_g, targets)
            loss = target_masks * loss
            loss = loss.sum() / target_masks.sum()
        else:
            loss = loss_fun(out_g, targets[hp.context_len:(hp.context_len + out_g.shape[0])])
            #loss = target_masks * loss
            loss = loss.sum() / target_masks.sum()
            #loss = loss.mean()

        l = loss.cpu().data.numpy()
        optimizer.zero_grad()
        if extras["train"]:
            loss.backward()
            clipping_grad_norm_(model.parameters(), hp.clip)
            optimizer.step()
        else:
            # don't carry over valid memories into train, just copy the train ones over
            out_mems = in_mems
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
