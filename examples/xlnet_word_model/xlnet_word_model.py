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
from kkpthlib import RampOpt
from kkpthlib import run_loop


hp = HParams(memory_len=20,
             context_len=70,
             embedding_dropout_keep_prob=.8,
             transformer_input_dim=380,
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             learning_rate=3E-4,
             min_learning_rate=1E-4,
             clip=.25,
             batch_size=10,
             n_layers=16,
             data_storage_dir="kjv",
             max_sequence_length=140,
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
                                                     random_state=random_state,
                                                     memory_len=hp.memory_len,
                                                     context_len=hp.context_len,
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
            # target_mappings is not None case
            # use xe_k for size broadcasting
            # mask embedding is kind of silly with target_mappings provided
            xe_q = 0. * xe_k.detach() + self.embedding_q[None, None]
            out, l_o_m = self.transformer(xe_k, xe_q,
                                          perm_masks,
                                          target_mappings,
                                          target_masks,
                                          list_of_mems=list_of_mems)
            p = self.out_proj([out])
            return p, l_o_m
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

        # +1 to account for autoregressive targets
        # context_len because we have a reduction mapping - targets are a subset of the "target sequence", effectively
        np_perm_masks, np_target_mappings, np_target_masks, np_input_qs = model.transformer.make_masks_and_mappings(np_data, context_cut=hp.context_len + 1, random_state=gen_random_state)

        input_data = torch.tensor(np_data).to(hp.use_device)
        target = torch.tensor(np_data).long().to(hp.use_device)

        # input_data is input_k_0 in xlnet code
        input_data = input_data[:-1]
        input_data = input_data[..., None]

        # need to embed it?
        target = target[1:]
        target = target[..., None]

        # these 3 use context_cut
        perm_masks = torch.tensor(np_perm_masks).to(hp.use_device)
        target_masks = torch.tensor(np_target_masks).to(hp.use_device)
        input_qs = torch.tensor(np_input_qs).to(hp.use_device)

        # this one doesn't - need to cut manually to match target
        target_mappings = torch.tensor(np_target_mappings[1:]).to(hp.use_device)

        in_mems = stateful_args
        out, out_mems = model(input_data, input_qs, perm_masks, target_mappings, target_masks, list_of_mems=in_mems)
        loss = loss_fun(out, target[hp.context_len:])
        #loss = loss.sum(axis=0).mean()
        loss = loss.mean()
        l = loss.cpu().data.numpy()
        optimizer.zero_grad()
        if extras["train"]:
            loss.backward()
            clipping_grad_norm_(model.parameters(), hp.clip)
            optimizer.step()
        return l, None, out_mems

    """
    # the out-of-loop-check
    r = loop(train_itr, {"train": True}, None)
    r2 = loop(train_itr, {"train": True}, r[-1])
    """

    s = {"model": model,
         "optimizer": optimizer,
         "hparams": hp}

    run_loop(loop, train_itr,
             loop, valid_itr,
             s,
             n_train_steps_per=2000,
             n_valid_steps_per=250)
