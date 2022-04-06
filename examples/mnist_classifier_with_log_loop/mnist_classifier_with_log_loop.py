from __future__ import print_function
import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F

from kkpthlib.datasets import fetch_mnist
from kkpthlib import Linear
from kkpthlib import relu
from kkpthlib import clipping_grad_norm_
from kkpthlib import ListIterator
from kkpthlib import run_loop
from kkpthlib import HParams

mnist = fetch_mnist()
hp = HParams(input_dim=mnist["data"].shape[-1],
             hidden_dim=512,
             out_dim=10,
             use_device="cuda",
             learning_rate=0.0001,
             clip=10.,
             batch_size=100,
             n_epochs=100,
             random_seed=2122)

random_state = np.random.RandomState(hp.random_seed)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = Linear([hp.input_dim], hp.hidden_dim, random_state=random_state, name="l1")
        self.out = Linear([hp.hidden_dim], hp.out_dim, random_state=random_state, name="out")

    def forward(self, x):
        h = relu(self.l1([x]))
        o = self.out([h])
        return o

m = Model().to(hp.use_device)
optimizer = torch.optim.Adam(m.parameters(), hp.learning_rate)
l_fun = nn.CrossEntropyLoss()

train_data = mnist["data"][mnist["train_indices"]]
train_target = mnist["target"][mnist["train_indices"]]
valid_data = mnist["data"][mnist["valid_indices"]]
valid_target = mnist["target"][mnist["valid_indices"]]

train_itr = ListIterator([train_data, train_target], batch_size=hp.batch_size, random_state=random_state)
valid_itr = ListIterator([valid_data, valid_target], batch_size=hp.batch_size, random_state=random_state)

def loop(itr, extras, stateful_args):
    data_batch, target_batch = next(itr)
    data_batch = torch.tensor(data_batch).to(hp.use_device)
    target_batch = torch.tensor(target_batch).long().to(hp.use_device)
    r = m(data_batch)
    loss = l_fun(r, target_batch)
    l = loss.cpu().data.numpy()
    optimizer.zero_grad()
    if extras["train"]:
        loss.backward()
        clipping_grad_norm_(m.parameters(), hp.clip)
        optimizer.step()
    return l, None

s = {"model": m,
     "optimizer": optimizer,
     "hparams": hp}

run_loop(loop, train_itr,
         loop, valid_itr,
         s,
         n_epochs=hp.n_epochs)
