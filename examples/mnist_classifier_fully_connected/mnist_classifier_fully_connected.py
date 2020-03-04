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

mnist = fetch_mnist()
input_dim = mnist["data"].shape[-1]
hidden_dim = 512
out_dim = 10
use_device = "cuda"
random_state = np.random.RandomState(2122)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = Linear([input_dim], hidden_dim, random_state=random_state, name="l1")
        self.out = Linear([hidden_dim], out_dim, random_state=random_state, name="out")

    def forward(self, x):
        h = relu(self.l1([x]))
        o = self.out([h])
        return o

m = Model().to(use_device)
learning_rate = 0.0001
clip = 10.
batch_size = 64
epochs = 100

optimizer = torch.optim.Adam(m.parameters(), learning_rate)
l_fun = nn.CrossEntropyLoss()

train_data = mnist["data"][mnist["train_indices"]]
train_target = mnist["target"][mnist["train_indices"]]
valid_data = mnist["data"][mnist["valid_indices"]]
valid_target = mnist["target"][mnist["valid_indices"]]

train_itr = ListIterator([train_data, train_target], batch_size=batch_size, random_state=random_state)
valid_itr = ListIterator([valid_data, valid_target], batch_size=batch_size, random_state=random_state)

for e in range(epochs):
    print("epoch {}".format(e))
    train_losses = []
    for a in train_itr:
        data_batch, target_batch = a
        data_batch = torch.tensor(data_batch).to(use_device)
        target_batch = torch.tensor(target_batch).long().to(use_device)
        r = m(data_batch)
        loss = l_fun(r, target_batch)
        train_losses.append(loss.cpu().data.numpy())
        optimizer.zero_grad()
        loss.backward()
        clipping_grad_norm_(m.parameters(), clip)
        optimizer.step()
    print("epoch {} loss: {}".format(e, np.mean(train_losses)))

    with torch.no_grad():
        valid_losses = []
        for a in valid_itr:
            data_batch, target_batch = a
            data_batch = torch.tensor(data_batch).to(use_device)
            target_batch = torch.tensor(target_batch).long().to(use_device)
            r = m(data_batch)
            loss = l_fun(r, target_batch)
            valid_losses.append(loss.cpu().data.numpy())
        print("epoch {} valid loss: {}".format(e, np.mean(valid_losses)))
