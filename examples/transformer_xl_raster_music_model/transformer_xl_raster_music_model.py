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
from kkpthlib import EmbeddingDropout
from kkpthlib import WordCorpus
from kkpthlib import make_batches_from_list
from kkpthlib import StepIterator
from kkpthlib import CategoricalCrossEntropyFromLogits
from kkpthlib import clipping_grad_norm_
from kkpthlib import RampOpt
from kkpthlib import run_loop

from kkpthlib import fetch_jsb_chorales
from kkpthlib import piano_roll_from_music_json_file
from kkpthlib import MusicJSONCorpus

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
             max_sequence_length=140,
             max_vocabulary_size=133, # len(corpus.dictionary.counter)
             random_seed=2122)

def get_hparams():
    return hp

def build_model(hp):
    random_state = np.random.RandomState(hp.random_seed)
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.embedding = EmbeddingDropout(hp.max_vocabulary_size,
                                              hp.transformer_input_dim,
                                              dropout_keep_prob=hp.embedding_dropout_keep_prob,
                                              random_state=random_state,
                                              device=hp.use_device,
                                              name="embed")
            self.transformer = AWDTransformerXLDecoderBlock([hp.transformer_input_dim],
                                                            name="transformer_block",
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

        def forward(self, x, list_of_mems=None):
            xe, de = self.embedding(x)
            out, l_o_m = self.transformer(xe, list_of_mems=list_of_mems)
            p = self.out_proj([out])
            return p, l_o_m
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

    corpus = MusicJSONCorpus(train_data_file_paths=train_files,
                             valid_data_file_paths=valid_files)
    #all_pp = [piano_roll_from_music_json_file(at).ravel() for at in all_transposed]
    #all_pp = [ppi for ppi in np.concatenate(all_pp)]
    # all_pp is one giant list
    train_batches = make_batches_from_list(corpus.train, batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len)
    valid_batches = make_batches_from_list(corpus.valid, batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len)

    train_random_state = np.random.RandomState(hp.random_seed)
    valid_random_state = np.random.RandomState(hp.random_seed + 1)

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
        input_data = torch.tensor(np_data).to(hp.use_device)
        target = torch.tensor(np_data).long().to(hp.use_device)
        input_data = input_data[:-1]
        input_data = input_data[..., None]
        # need to embed it?
        target = target[1:]
        target = target[..., None]

        in_mems = stateful_args
        out, out_mems = model(input_data, list_of_mems=in_mems)
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

    # the out-of-loop-check
    #r = loop(train_itr, {"train": True}, None)
    #r2 = loop(train_itr, {"train": True}, r[-1])

    s = {"model": model,
         "optimizer": optimizer,
         "hparams": hp}

    run_loop(loop, train_itr,
             loop, valid_itr,
             s,
             n_train_steps_per=2000,
             n_valid_steps_per=250)
