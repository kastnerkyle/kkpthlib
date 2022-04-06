from __future__ import print_function
import os
import sys

import numpy as np
import torch
from torch import nn
import torch.functional as F

#from kkpthlib.datasets import fetch_jsb_chorales
from kkpthlib import AWDTransformerXLDecoderBlock
from kkpthlib import HParams
from kkpthlib import Linear
from kkpthlib import Embedding
from kkpthlib import WordCorpus
from kkpthlib import make_batches_from_list
from kkpthlib import StepIterator
from kkpthlib import CategoricalCrossEntropyFromLogits
from kkpthlib import clipping_grad_norm_
from kkpthlib import RampOpt

hp = HParams(input_dim=133, #hardcoded from knowledge of vocab size, can temporarily set to 0 if needed
             #balance token dim and clock dim
             memory_len=20,
             context_len=50,
             transformer_input_dim=380,
             token_embed_dim=96,
             clock_embed_dim=16,
             # maybe change this?
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             learning_rate=0.1,
             clip=100.,
             batch_size=10,
             clocks=[2, 4, 8, 16, 32, 64],
             n_layers=5,
             dropout_keep_prob=1.,
             max_sequence_length=200,
             max_vocabulary_size=10000,
             random_seed=2122,
             data_storage_dir="kjv")

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

train_itr = StepIterator([train_batches], circular_rotation=True, random_state=train_random_state)
valid_itr = StepIterator([valid_batches], random_state=valid_random_state)
test_itr = StepIterator([test_batches], random_state=test_random_state)

def get_hparams():
    return hp

def build_model(hp):
    random_state = np.random.RandomState(hp.random_seed)
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.embedding = Embedding(hp.max_vocabulary_size,
                                       hp.transformer_input_dim,
                                       random_state=random_state,
                                       device=hp.use_device,
                                       name="embed")
            self.transformer = AWDTransformerXLDecoderBlock([hp.transformer_input_dim],
                                                            name="transformer_block",
                                                            random_state=random_state,
                                                            memory_len=hp.memory_len,
                                                            context_len=hp.context_len,
                                                            device=hp.use_device)
            self.out_proj = Linear([hp.transformer_input_dim],
                                    hp.max_vocabulary_size,
                                    random_state=random_state,
                                    device=hp.use_device,
                                    name="model_out")

        def forward(self, x, list_of_mems=None):
            xe, de = self.embedding(x)
            out, l_o_m = self.transformer(xe, list_of_mems=list_of_mems)
            p = self.out_proj([out])
            return p, l_o_m
    return Model().to(hp.use_device)

model = build_model(hp)
loss_fun = CategoricalCrossEntropyFromLogits()

def get_std_ramp_opt(model):
    return RampOpt(hp.learning_rate, 1, 10000, 10000 * 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1E-9))

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
    loss = loss.sum(axis=0).mean()
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
         n_train_steps_per=500,
         n_valid_steps_per=25)
