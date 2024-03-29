from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F

from kkpthlib.datasets import fetch_mnist
from kkpthlib import Linear
from kkpthlib import Dropout
from kkpthlib import BernoulliCrossEntropyFromLogits
from kkpthlib import relu
from kkpthlib import softmax
from kkpthlib import log_softmax
#from kkpthlib import clipping_grad_norm_
from kkpthlib import clipping_grad_value_
from kkpthlib import ListIterator
from kkpthlib import run_loop
from kkpthlib import HParams
from kkpthlib import Conv2d
from kkpthlib import Conv2dTranspose

from kkpthlib import MelNetLayer
from kkpthlib import MelNetFullContextLayer
from kkpthlib import relu

from kkpthlib import space2batch
from kkpthlib import batch2space
from kkpthlib import split
from kkpthlib import interleave
from kkpthlib import scan
from kkpthlib import LSTMCell
from kkpthlib import BiLSTMLayer
from kkpthlib import GaussianAttentionCell

mnist = fetch_mnist()

hp = HParams(input_dim=1,
             hidden_dim=512,
             n_mixtures=10,
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             learning_rate=1E-4,
             clip=3.5,
             cell_dropout=.8,
             input_noise=.25,
             input_dropout=1.,
             melnet_init="normal",
             input_image_size=(28, 28),
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
            self.drop = Dropout(hp.input_dropout, random_state=random_state, device=hp.use_device)
            self.mnlayer1 = MelNetLayer([hp.hidden_dim], hp.hidden_dim, cell_dropout=hp.cell_dropout, random_state=random_state, name="mnlayer1",
                                        init=hp.melnet_init)
            self.mnlayer2 = MelNetLayer([hp.hidden_dim], hp.hidden_dim, cell_dropout=hp.cell_dropout, random_state=random_state, name="mnlayer2",
                                        init=hp.melnet_init)
            self.mnlayer3 = MelNetLayer([hp.hidden_dim], hp.hidden_dim, cell_dropout=hp.cell_dropout, random_state=random_state, name="mnlayer3",
                                        init=hp.melnet_init)
            self.mnlayer4 = MelNetLayer([hp.hidden_dim], hp.hidden_dim, cell_dropout=hp.cell_dropout, random_state=random_state, name="mnlayer4",
                                        init=hp.melnet_init)
            self.mnlayer5 = MelNetLayer([hp.hidden_dim], hp.hidden_dim, cell_dropout=hp.cell_dropout, random_state=random_state, name="mnlayer5",
                                        init=hp.melnet_init)
            self.mnlayer6 = MelNetLayer([hp.hidden_dim], hp.hidden_dim, cell_dropout=hp.cell_dropout, random_state=random_state, name="mnlayer6",
                                        init=hp.melnet_init)
            self.cond_mnlayer = MelNetFullContextLayer([hp.hidden_dim], hp.hidden_dim, random_state=random_state, name="cond_mnlayer1",
                                                       init=hp.melnet_init)

            self.conv1 = Conv2d([hp.input_dim], hp.hidden_dim, kernel_size=(1, 1), strides=(1, 1), border_mode=(0, 0),
                                random_state=random_state, name="conv1")
            self.conv2 = Conv2d([hp.input_dim], hp.hidden_dim, kernel_size=(1, 1), strides=(1, 1), border_mode=(0, 0),
                                random_state=random_state, name="conv2")
            self.out_conv1 = Conv2d([hp.hidden_dim], hp.input_dim, kernel_size=(1, 1), strides=(1, 1), border_mode=(0, 0),
                                    random_state=random_state, name="out_conv1")

            self.out_conv2 = Conv2d([hp.hidden_dim], hp.input_dim, kernel_size=(1, 1), strides=(1, 1), border_mode=(0, 0),
                                    random_state=random_state, name="out_conv2")
            self.centralized_proj = Linear([hp.hidden_dim * hp.input_image_size[0] // 2], hp.hidden_dim, random_state=random_state, name="centralized_proj")
            self.centralized_proj2 = Linear([hp.hidden_dim * hp.input_image_size[0] // 2], hp.hidden_dim, random_state=random_state, name="centralized_proj2")


        def forward(self, x):
            # boutta do a lot of depth2space and stuff
            x = self.drop(x)

            # targets
            tier0_1, tier0_2 = split(x, axis=3)
            #tier0_1, tier0_2 = split(tier1_1, axis=2)

            x_proj_split0_1 = self.conv1([tier0_1])
            x_proj_split0_2 = self.conv2([tier0_2])

            # modeling order is - unconditional tier 0_1 , conditional tier0_2 on tier0_1
            # interleave the outputs for both, use to conditional tier1_1 on interleave(tier0_1, tier0_2)
            # repeat the chain
            ii = x_proj_split0_1
            inp_shift_t = torch.cat((0. * ii[:, :, :1, :], ii), axis=2)
            inp_shift_f = torch.cat((0. * ii[:, :, :, :1], ii), axis=3)

            shp = inp_shift_t.shape
            inp_shift_t_c_pre = inp_shift_t.permute(2, 0, 1, 3).reshape(shp[2], shp[0], -1)
            inp_shift_t_c = self.centralized_proj([inp_shift_t_c_pre])

            tier0_1_rec_t, tier0_1_rec_f, tier0_1_rec_c = self.mnlayer1([inp_shift_t, inp_shift_f, inp_shift_t_c])
            tier0_1_rec_t, tier0_1_rec_f, tier0_1_rec_c = self.mnlayer2([tier0_1_rec_t, tier0_1_rec_f, tier0_1_rec_c])
            tier0_1_rec_t, tier0_1_rec_f, tier0_1_rec_c = self.mnlayer3([tier0_1_rec_t, tier0_1_rec_f, tier0_1_rec_c])
            tier0_1_rec_t, tier0_1_rec_f, tier0_1_rec_c = self.mnlayer4([tier0_1_rec_t, tier0_1_rec_f, tier0_1_rec_c])
            out_pred0_1 = self.out_conv1([tier0_1_rec_f[:, :, :, :-1]])

            cond_x = self.cond_mnlayer([x_proj_split0_1])

            ii = x_proj_split0_2
            inp_shift_t = torch.cat((0. * ii[:, :, :1, :], ii), axis=2)
            inp_shift_f = torch.cat((0. * ii[:, :, :, :1], ii), axis=3)
            inp_shift_t[:, :, :-1, :] += cond_x
            inp_shift_f[:, :, :, :-1] += cond_x
            shp = inp_shift_t.shape
            inp_shift_t_c_pre = inp_shift_t.permute(2, 0, 1, 3).reshape(shp[2], shp[0], -1)
            inp_shift_t_c = self.centralized_proj2([inp_shift_t_c_pre])

            tier0_2_rec_t, tier0_2_rec_f, tier0_2_rec_c = self.mnlayer5([inp_shift_t, inp_shift_f, inp_shift_t_c])
            tier0_2_rec_t, tier0_2_rec_f, tier0_2_rec_c = self.mnlayer6([tier0_2_rec_t, tier0_2_rec_f, tier0_2_rec_c])
            out_pred0_2 = self.out_conv2([tier0_2_rec_f[:, :, :, :-1]])
            return out_pred0_1, out_pred0_2

    return Model().to(hp.use_device)

if __name__ == "__main__":
    model = build_model(hp)
    optimizer = torch.optim.Adam(model.parameters(), hp.learning_rate)
    l_fun = BernoulliCrossEntropyFromLogits()

    data_random_state = np.random.RandomState(hp.random_seed)
    train_data = mnist["data"][mnist["train_indices"]]
    valid_data = mnist["data"][mnist["valid_indices"]]

    train_itr = ListIterator([train_data], batch_size=hp.batch_size, random_state=data_random_state,
                             infinite_iterator=True)
    valid_itr = ListIterator([valid_data], batch_size=hp.batch_size, random_state=data_random_state,
                             infinite_iterator=True)

    """
    train_data = mnist["data"][mnist["train_indices"]]
    train_target = mnist["target"][mnist["train_indices"]]
    valid_data = mnist["data"][mnist["valid_indices"]]
    valid_target = mnist["target"][mnist["valid_indices"]]

    train_itr = ListIterator([train_data, train_target], batch_size=hp.batch_size, random_state=data_random_state,
                             infinite_iterator=True)
    valid_itr = ListIterator([valid_data, valid_target], batch_size=hp.batch_size, random_state=data_random_state,
                             infinite_iterator=True)
    """

    def loop(itr, extras, stateful_args):
        if extras["train"]:
            model.train()
            noise_level = hp.input_noise
        else:
            model.eval()
            noise_level = 0.

        data_batch, = next(itr)
        # N H W C
        data_batch = data_batch.reshape(data_batch.shape[0], 28, 28, 1)
        data_batch = data_batch.transpose(0, 3, 1, 2)
        data_batch = data_batch / 255.
        #data_batch = data_batch[:, :, ::2, ::2]
        torch_data_batch = torch.tensor(data_batch).contiguous().to(hp.use_device)
        data_batch = np.clip(data_batch + noise_level * data_random_state.randn(*data_batch.shape), 0., 1.).astype("float32")
        torch_noise_data_batch = torch.tensor(data_batch).contiguous().to(hp.use_device)

        # output, convolutional logits, flattened logits, sampled, final sampled and reshaped
        # targets
        tier0_1, tier0_2 = split(torch_data_batch, axis=3)

        tier0_1_pred, tier0_2_pred = model(torch_noise_data_batch)
        tier0_1_loss = (tier0_1 - tier0_1_pred) ** 2
        tier0_2_loss = (tier0_2 - tier0_2_pred) ** 2
        tier0_loss = tier0_1_loss.mean() + tier0_2_loss.mean()
        loss = tier0_loss
        #tier0_rec = interleave(tier0_1_pred, tier0_2_pred, axis=3)
        #tier0_loss = (tier0_rec - torch_data_batch) ** 2
        #tier0_1loss = (tier0_1 - relu(tier0_1_pred)) ** 2
        #tier0_2loss = (tier0_2 - relu(tier0_2_pred)) ** 2
        # first "tier" done
        # could interleave then do loss?
        #tier0_rec = interleave(tier0_1_rec_f[:, :, :-1, :], tier0_2_rec_f[:, :, :-1, :], axis=2)
        #tier0_loss = tier0_1loss + tier0_2loss
        #loss = tier0_loss.mean()
        l = loss.cpu().data.numpy()

        optimizer.zero_grad()
        if extras["train"]:
            loss.backward()
            clipping_grad_value_(model.parameters(), hp.clip)
            optimizer.step()
        return [l,], None, None

    s = {"model": model,
         "optimizer": optimizer,
         "hparams": hp}

    """
    # the out-of-loop-check
    r = loop(train_itr, {"train": True}, None)
    r2 = loop(train_itr, {"train": True}, None)
    from IPython import embed; embed(); raise ValueError()
    """

    run_loop(loop, train_itr,
             loop, valid_itr,
             s,
             n_train_steps_per=1000,
             n_valid_steps_per=250)
