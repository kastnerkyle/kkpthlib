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
from kkpthlib import piano_roll_from_music_json_file
from kkpthlib import MusicJSONFlatMeasureCorpus

hp = HParams(memory_len=20,
             context_len=64,
             max_sequence_length=128,
             transformer_input_dim=380,
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             learning_rate=1E-5,
             min_learning_rate=1E-6,
             clip=.25,
             batch_size=6,
             n_layers=24,

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
             max_vocabulary_size=68,
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
    # 15 is ~5% of the data
    # holding out whole songs so actually a pretty hard validation set...
    valid_names = bwv_names[:15]
    train_files = [f for f in all_transposed if all([vn not in f for vn in valid_names])]
    valid_files = [f for f in all_transposed if any([vn in f for vn in valid_names])]

    flat_measure_corpus = MusicJSONFlatMeasureCorpus(train_data_file_paths=train_files,
                                                     valid_data_file_paths=valid_files)

    # length of that list is the number of feature groups!
    train_batches = None
    valid_batches = None
    for v in range(len(flat_measure_corpus.train)):
        # num_batch_steps, time_length_of_batch, batch_size
        this_train_batches = make_batches_from_list(flat_measure_corpus.train[v], batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len)
        this_valid_batches = make_batches_from_list(flat_measure_corpus.valid[v], batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len)
        if train_batches is None:
            train_batches = this_train_batches[..., None]
            valid_batches = this_valid_batches[..., None]
        else:
            # num_batch_steps, time_length_of_batch, batch_size, features (4 - pitch, duration, velocity, voice)
            train_batches = np.concatenate((train_batches, this_train_batches[..., None]), axis=-1)
            valid_batches = np.concatenate((valid_batches, this_valid_batches[..., None]), axis=-1)

    train_random_state = np.random.RandomState(hp.random_seed)
    valid_random_state = np.random.RandomState(hp.random_seed + 1)
    gen_random_state = np.random.RandomState(hp.random_seed + 2)

    train_itr = StepIterator([train_batches], circular_rotation=True, random_state=train_random_state)
    valid_itr = StepIterator([valid_batches], random_state=valid_random_state)


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
        np_perm_masks, np_target_mappings, np_target_masks, _, _, _, np_perm_orders = model.transformer.make_inputs_targets_masks_and_mappings(np_pitch_data, K=6, context_cut=hp.context_len, random_state=gen_random_state)
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
