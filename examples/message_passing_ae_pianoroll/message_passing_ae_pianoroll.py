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
from kkpthlib import softmax
from kkpthlib import log_softmax
from kkpthlib import clipping_grad_norm_
from kkpthlib import ListIterator
from kkpthlib import run_loop
from kkpthlib import HParams
from kkpthlib import Conv2d
from kkpthlib import Conv2dTranspose

mnist = fetch_mnist()
hp = HParams(input_dim=1,
             hidden_dim=512,
             out_dim=10,
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             learning_rate=1E-6,
             clip=10.,
             batch_size=20,
             n_epochs=1000,
             random_seed=2122)

def get_hparams():
    return hp

def build_model(hp):
    random_state = np.random.RandomState(hp.random_seed)
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = Conv2d([hp.input_dim], 16, kernel_size=(4, 4), strides=(2, 2), border_mode=(1, 1), random_state=random_state, name="conv1")
            self.conv2 = Conv2d([16], 32, kernel_size=(4, 4), strides=(2, 2), border_mode=(1, 1), random_state=random_state, name="conv2")
            self.conv3 = Conv2d([32], 64, kernel_size=(4, 4), strides=(1, 1), border_mode=0, random_state=random_state, name="conv3")
            self.conv4 = Conv2d([64], 256, kernel_size=(1, 1), strides=(1, 1), border_mode=0, random_state=random_state, name="conv4")

            self.transpose_convpre = Conv2d([256], 64, kernel_size=(1, 1), strides=(1, 1), border_mode=0, random_state=random_state, name="convpre")
            self.transpose_conv1 = Conv2dTranspose([64], 64, kernel_size=(3, 3), strides=(1, 1), border_mode=(1, 1), random_state=random_state, name="transpose_conv1")
            self.transpose_conv2 = Conv2dTranspose([64], 32, kernel_size=(5, 5), strides=(1, 1), border_mode=(0, 0), random_state=random_state, name="transpose_conv2")
            self.transpose_conv3 = Conv2dTranspose([32], 32, kernel_size=(4, 4), strides=(2, 2), border_mode=(2, 2), random_state=random_state, name="transpose_conv3")
            self.transpose_conv4 = Conv2dTranspose([32], 16, kernel_size=(4, 4), strides=(2, 2), border_mode=(1, 1), random_state=random_state, name="transpose_conv4")
            self.transpose_conv5 = Conv2dTranspose([16], 1, kernel_size=(1, 1), strides=(1, 1), border_mode=(0, 0), random_state=random_state, name="transpose_conv5")

        def sample_gumbel(self, logits, temperature=1.):
            noise = random_state.uniform(1E-5, 1. - 1E-5, logits.shape)
            torch_noise = torch.tensor(noise).contiguous().to(hp.use_device)

            #return np.argmax(np.log(softmax(logits, temperature)) - np.log(-np.log(noise)))

            # max indices
            maxes = torch.argmax(logits / float(temperature) - torch.log(-torch.log(torch_noise)), axis=-1, keepdim=True)
            one_hot = 0. * logits
            one_hot.scatter_(-1, maxes, 1)
            return one_hot

        def forward(self, x):
            h1 = relu(self.conv1([x]))
            h2 = relu(self.conv2([h1]))
            h3 = relu(self.conv3([h2]))
            h4 = relu(self.conv4([h3]))

            logits = h4.reshape((h4.shape[0], h4.shape[1], -1))

            # credit to tim cooijmans for this dice trick + code
            # https://arxiv.org/pdf/1802.05098.pdf 
            y = self.sample_gumbel(logits)  # note y is one-hot

            py = softmax(logits)
            logpy = (y * log_softmax(logits)).sum(axis=-1, keepdims=True)
            dice = torch.exp(logpy - logpy.detach())
            sample_res = (y - py).detach() * dice + py.detach()
            sample_res = sample_res.reshape((sample_res.shape[0], sample_res.shape[1], h4.shape[-2], h4.shape[-1])).contiguous()

            pdh1 = relu(self.transpose_convpre([sample_res]))
            dh1 = relu(self.transpose_conv1([pdh1]))
            dh2 = relu(self.transpose_conv2([dh1]))
            dh3 = relu(self.transpose_conv3([dh2]))
            dh4 = relu(self.transpose_conv4([dh3]))
            dh5 = self.transpose_conv5([dh4])
            return dh5, h4, logits, y, sample_res
    return Model().to(hp.use_device)

if __name__ == "__main__":
    m = build_model(hp)
    optimizer = torch.optim.Adam(m.parameters(), hp.learning_rate)
    l_fun = nn.CrossEntropyLoss()

    data_random_state = np.random.RandomState(hp.random_seed)

    train_data = mnist["data"][mnist["train_indices"]]
    train_target = mnist["target"][mnist["train_indices"]]
    valid_data = mnist["data"][mnist["valid_indices"]]
    valid_target = mnist["target"][mnist["valid_indices"]]

    train_itr = ListIterator([train_data, train_target], batch_size=hp.batch_size, random_state=data_random_state,
                             infinite_iterator=True)
    valid_itr = ListIterator([valid_data, valid_target], batch_size=hp.batch_size, random_state=data_random_state,
                             infinite_iterator=True)

    def loop(itr, extras, stateful_args):
        data_batch, target_batch = next(itr)
        # N H W C
        data_batch = data_batch.reshape(data_batch.shape[0], 28, 28, 1)
        data_batch = data_batch.transpose(0, 3, 1, 2) / 255.0
        data_batch = torch.tensor(data_batch).contiguous().to(hp.use_device)

        target_batch = torch.tensor(target_batch).long().to(hp.use_device)

        # output, convolutional logits, flattened logits, sampled, final sampled and reshaped
        r, c_l_s, l_s, s1, s2 = m(data_batch)
        loss = ((r - data_batch) ** 2).mean()
        #loss = l_fun(r, target_batch)
        l = loss.cpu().data.numpy()
        optimizer.zero_grad()
        if extras["train"]:
            loss.backward()
            clipping_grad_norm_(m.parameters(), hp.clip)
            optimizer.step()
        return l, None, None

    s = {"model": m,
         "optimizer": optimizer,
         "hparams": hp}

    """
    # the out-of-loop-check
    r = loop(train_itr, {"train": True}, None)
    r2 = loop(train_itr, {"train": True}, None)
    print("no loop yet")
    from IPython import embed; embed(); raise ValueError()
    """

    run_loop(loop, train_itr,
             loop, valid_itr,
             s,
             n_train_steps_per=1000,
             n_valid_steps_per=250)
