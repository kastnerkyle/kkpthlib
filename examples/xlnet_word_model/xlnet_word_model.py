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
             max_vocabulary_size=10000,
             random_seed=2122)


def get_hparams():
    return hp

def build_model(hp):
    random_state = np.random.RandomState(hp.random_seed)
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.embedding_k = EmbeddingDropout(hp.max_vocabulary_size,
                                                hp.transformer_input_dim,
                                                dropout_keep_prob=hp.embedding_dropout_keep_prob,
                                                random_state=random_state,
                                                device=hp.use_device,
                                                name="embed")
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
            xe_k, de_k = self.embedding_k(input_ks)

            if target_mappings == None:
                # use xe_k for size broadcasting
                _q = 0. * xe_k.detach() + self.embedding_q[None, None]
                xe_q = 0. * xe_k.detach()
                # everywhere there is a target, blank out the standard embedding and replace with the learned mask emb
                # inside transformer if target_mappings is provided this gets sliced down
                xe_q = target_masks[..., None] * _q + (1. - target_masks[..., None]) * xe_k
            else:
                # 
                xe_q = 0. * xe_k[:target_masks.sum(dim=0).max().long()].detach() + self.embedding_q[None, None]

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
    # are memories correct, or should they be shifted by half to account for "context"
    train_data_file_path = hp.data_storage_dir + os.sep + "train.txt"
    valid_data_file_path = hp.data_storage_dir + os.sep + "valid.txt"
    test_data_file_path = hp.data_storage_dir + os.sep + "test.txt"
    corpus = WordCorpus(train_data_file_path=train_data_file_path,
                        valid_data_file_path=valid_data_file_path,
                        test_data_file_path=test_data_file_path,
                        cleaner_fn="lower_ascii_keep_standard_punctuation",
                        max_vocabulary_size=hp.max_vocabulary_size,
                        use_eos=False)
    train_batches = make_batches_from_list(corpus.train, batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len)
    valid_batches = make_batches_from_list(corpus.valid, batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len)
    test_batches = make_batches_from_list(corpus.test, batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len)

    train_random_state = np.random.RandomState(hp.random_seed)
    valid_random_state = np.random.RandomState(hp.random_seed + 1)
    test_random_state = np.random.RandomState(hp.random_seed + 2)

    gen_random_state = np.random.RandomState(hp.random_seed + 3)

    train_itr = StepIterator([train_batches], circular_rotation=True, random_state=train_random_state)
    valid_itr = StepIterator([valid_batches], random_state=valid_random_state)
    test_itr = StepIterator([test_batches], random_state=test_random_state)

    model = build_model(hp)
    loss_fun = CategoricalCrossEntropyFromLogits()

    def get_std_ramp_opt(model):
        return RampOpt(hp.learning_rate, 1, 3000, 3000 + 125 * 450,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1E-9),
                       min_decay_learning_rate=hp.min_learning_rate)

    optimizer = get_std_ramp_opt(model)

    def loop(itr, extras, stateful_args):
        np_data = next(itr)
        np_perm_masks, np_target_mappings, np_target_masks, np_input_ks, np_input_qs, np_targets, np_perm_orders = model.transformer.make_inputs_targets_masks_and_mappings(np_data, K=6, context_cut=hp.context_len, random_state=gen_random_state)

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

        input_ks = input_ks[..., None]
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
