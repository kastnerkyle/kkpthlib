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
             hidden_dim=2,
             n_mixtures=10,
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             learning_rate=1E-6,
             clip=3.5,
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
            self.conv1 = Conv2d([hp.input_dim], 200, kernel_size=(1, 1), strides=(1, 1), border_mode=(0, 0),
                                random_state=random_state, name="conv1")

            self.conv2 = Conv2d([16], 32, kernel_size=(4, 4), strides=(2, 2), border_mode=(1, 1), random_state=random_state, name="conv2")
            self.conv3 = Conv2d([32], 64, kernel_size=(4, 4), strides=(1, 1), border_mode=0, random_state=random_state, name="conv3")
            self.conv4 = Conv2d([64], 32, kernel_size=(1, 1), strides=(1, 1), border_mode=0, random_state=random_state, name="conv4")

            self.pre_mus = Linear([32 * 4 * 4], hp.hidden_dim * hp.n_mixtures, random_state=random_state, name="pre_mus")
            self.pre_log_sigmas = Linear([32 * 4 * 4], hp.hidden_dim * hp.n_mixtures, random_state=random_state, name="pre_sigmas")
            self.pre_mix = Linear([32 * 4 * 4], hp.n_mixtures, random_state=random_state, name="pre_mix")

            self.post = Linear([hp.hidden_dim], 32 * 4 * 4, random_state=random_state, name="post")

            self.transpose_convpre = Conv2d([32], 64, kernel_size=(1, 1), strides=(1, 1), border_mode=0, random_state=random_state, name="convpre")
            self.transpose_conv1 = Conv2dTranspose([64], 64, kernel_size=(3, 3), strides=(1, 1), border_mode=(1, 1), random_state=random_state, name="transpose_conv1")
            self.transpose_conv2 = Conv2dTranspose([64], 32, kernel_size=(5, 5), strides=(1, 1), border_mode=(0, 0), random_state=random_state, name="transpose_conv2")
            self.transpose_conv3 = Conv2dTranspose([32], 32, kernel_size=(4, 4), strides=(2, 2), border_mode=(2, 2), random_state=random_state, name="transpose_conv3")
            self.transpose_conv4 = Conv2dTranspose([32], 16, kernel_size=(4, 4), strides=(2, 2), border_mode=(1, 1), random_state=random_state, name="transpose_conv4")
            self.transpose_conv5 = Conv2dTranspose([16], hp.input_dim, kernel_size=(1, 1), strides=(1, 1), border_mode=(0, 0), random_state=random_state, name="transpose_conv5")

        def sample_gumbel(self, logits, temperature=1., dropout=True):
            noise = random_state.uniform(1E-5, 1. - 1E-5, logits.shape)
            torch_noise = torch.tensor(noise).contiguous().to(hp.use_device)

            #return np.argmax(np.log(softmax(logits, temperature)) - np.log(-np.log(noise)))

            # max indices
            combined_noise = logits / float(temperature) - torch.log(-torch.log(torch_noise))
            if dropout:
                # .5 keep prob
                dropout_mask = torch.tensor((random_state.uniform(0., 1., logits.shape) < 0.5).astype("float32")).to(hp.use_device)
                combined_noise = combined_noise.detach() * dropout_mask  + (1. - dropout_mask) * -1E6
            # dropout sample
            maxes = torch.argmax(combined_noise, axis=-1, keepdim=True)
            one_hot = 0. * logits
            one_hot.scatter_(-1, maxes, 1)
            return one_hot

        def forward(self, x, dropout_mix=False):

            # boutta do a lot of depth2space and stuff
            # trim y axis down by 4 to sanity check shapes
            x = x[..., :-4]

            x_flat = space2batch(x, axis=2)
            new_x = batch2space(x_flat, n_batch=x.shape[0], axis=2)

            plt.imshow(x[2, 0].cpu().data.numpy())
            plt.savefig("tmp0.png")
            plt.imshow(new_x[2, 0].cpu().data.numpy())
            plt.savefig("tmp1.png")

            x_flat = space2batch(x, axis=3)
            new_x = batch2space(x_flat, n_batch=x.shape[0], axis=3)

            plt.imshow(x[2, 0].cpu().data.numpy())
            plt.savefig("tmp2.png")
            plt.imshow(new_x[2, 0].cpu().data.numpy())
            plt.savefig("tmp3.png")

            # unconditional first tier
            def stepwise_conditional(tier_temp_base_t, tier_temp_base_f, condition_t=None, condition_f=None):
                # condidering axis 2 time 3 frequency
                # do time
                tier_temp_t = space2batch(tier_temp_base_t, axis=2)
                def tier_step_time(inp_t):
                    return [inp_t]

                r = scan(tier_step_time, [tier_temp_t], [None])
                tier_temp_t_res = r[0]

                # some model here
                tier_temp_revert_t = batch2space(tier_temp_t_res, n_batch=tier_temp_base_t.shape[0], axis=2)
                tier_temp_revert_t_merge = tier_temp_base_t + tier_temp_revert_t
                if condition_t is not None:
                    assert condition_f is not None
                    condition_temp_t = space2batch(condition_t, axis=2)
                    def condition_step_time(cond_t):
                        return [cond_t]

                    r = scan(condition_step_time, [condition_temp_t], [None])
                    condition_temp_t_res = r[0]
                    condition_temp_revert_t = batch2space(condition_temp_t_res, n_batch=condition_t.shape[0], axis=2)
                    tier_temp_revert_t_merge = tier_temp_revert_t_merge + condition_temp_revert_t

                # do frequency, noting freq conditions on time...
                tier_temp_base_f[:, :, :-1, :] = tier_temp_base_f[:, :, :-1, :] + tier_temp_revert_t_merge[:, :, :, :-1]
                tier_temp_f = space2batch(tier_temp_base_f, axis=3)

                def tier_step_freq(inp_f):
                    return [inp_f]

                r = scan(tier_step_freq, [tier_temp_f], [None])
                tier_temp_f_res = r[0]

                # some model here
                tier_temp_revert_f = batch2space(tier_temp_f_res, n_batch=tier_temp_base_f.shape[0], axis=3)
                tier_temp_revert_f_merge = tier_temp_revert_f + tier_temp_base_f
                if condition_f is not None:
                    assert condition_t is not None
                    condition_temp_f = space2batch(condition_f, axis=3)
                    def condition_step_freq(cond_f):
                        return [cond_f]
                    r = scan(condition_step_freq, [condition_temp_f], [None])
                    condition_temp_f_res = r[0]
                    condition_temp_revert_f = batch2space(condition_temp_f_res, n_batch=condition_f.shape[0], axis=3)
                    tier_temp_revert_f_merge = tier_temp_revert_f_merge + condition_temp_revert_f
                return tier_temp_revert_t_merge, tier_temp_revert_f_merge

            x_proj = self.conv1([x])

            # targets
            tier1_1, tier1_2 = split(x, axis=3)
            tier0_1, tier0_2 = split(tier1_1, axis=2)

            # modeling order is - unconditional tier 0_1 , conditional tier0_2 on tier0_1
            # interleave the outputs for both, use to conditional tier1_1 on interleave(tier0_1, tier0_2)
            # repeat the chain
            inp_shift_t = torch.cat((0. * tier0_1[:, :, :, :1], tier0_1), axis=3)
            inp_shift_f = torch.cat((0. * tier0_1[:, :, :1, :], tier0_1), axis=2)
            tier0_1_rec_t, tier0_1_rec_f = stepwise_conditional(inp_shift_t, inp_shift_f, condition_t=None, condition_f=None)
            loss0_1 = (tier0_1 - tier0_1_rec_f[:, :, :-1, :]) ** 2

            # shift conditioning terms by post padding then slicing again
            cond_t = torch.cat((tier0_1_rec_t, 0. * tier0_1_rec_t[:, :, :, :1]), axis=3)
            cond_f = torch.cat((tier0_1_rec_f, 0. * tier0_1_rec_f[:, :, :1, :]), axis=2)
            cond_t = cond_t[:, :, :, 1:]
            cond_f = cond_f[:, :, 1:, :]

            inp_shift_t = torch.cat((0. * tier0_2[:, :, :, :1], tier0_2), axis=3)
            inp_shift_f = torch.cat((0. * tier0_2[:, :, :1, :], tier0_2), axis=2)
            tier0_2_rec_t, tier0_2_rec_f = stepwise_conditional(inp_shift_t, inp_shift_f, condition_t=cond_t, condition_f=cond_f)
            loss0_2 = (tier0_2 - tier0_2_rec_f[:, :, :-1, :]) ** 2

            # first "tier" done
            tier0_rec = interleave(tier0_1_rec_f[:, :, :-1, :], tier0_2_rec_f[:, :, :-1, :], axis=2)

            tier0_loss = (tier0_rec - tier1_1) ** 2
            tier0_loss.requires_grad = True
            tier0_loss.sum().backward()

            print("here")
            from IPython import embed; embed(); raise ValueError()

            print("wut")
            from IPython import embed; embed(); raise ValueError()
            # perfect conditional gen
            tier0_2_rec = tier0_2 # stepwise_conditional(tier0_2, condition=tier0_1_rec)
            # combine
            tier1_1_rec = interleave(tier0_1_rec, tier0_2_rec, axis=2)

            # perfect conditional gen
            tier1_2_rec = tier1_2 # stepwise_conditional(tier1_2, condition=tier1_1_rec)
            x_rec = interleave(tier1_1_rec, tier1_2_rec, axis=3)


            """
            # perfect conditional gen
            tier0_2_rec = tier0_2 # stepwise_conditional(tier0_2, condition=tier0_1_rec)
            # combine
            tier1_1_rec = interleave(tier0_1_rec, tier0_2_rec, axis=2)

            # perfect conditional gen
            tier1_2_rec = tier1_2 # stepwise_conditional(tier1_2, condition=tier1_1_rec)
            tier2_1_rec = interleave(tier1_1_rec, tier1_2_rec, axis=3)

            tier2_2_rec = tier2_2 # stepwise_conditional(tier2_2, condition=tier2_1_rec)
            tier3_1_rec = interleave(tier2_1_rec, tier2_2_rec, axis=2)

            tier3_2_rec = tier3_2 # stepwise_conditional(tier3_2, condition=tier3_1_rec)
            x_rec = interleave(tier3_1_rec, tier3_2_rec, axis=3)
            """

            plt.imshow(tier0_1_rec[2, 0].cpu().data.numpy())
            plt.savefig("tier0.png")
            plt.imshow(tier1_1_rec[2, 0].cpu().data.numpy())
            plt.savefig("tier1.png")
            plt.imshow(x_rec[2, 0].cpu().data.numpy())
            plt.savefig("original.png")
            print("dbg")
            from IPython import embed; embed(); raise ValueError()

            new_x_2 = interleave(x_1, x_2, axis=2)

            plt.imshow(new_x_2[2, 0].cpu().data.numpy())
            plt.savefig("tmp3.png")
            plt.imshow(new_x_2[2, 0].cpu().data.numpy())
            plt.savefig("tmp4.png")
            print("dbg")
            from IPython import embed; embed(); raise ValueError()

            # fake the tiers and processing of the real model
            # tier 1 - 3 downsamples (28 x 28 -> 7x7
            # tier 2 - 2 downsamples (14 x 14 -> 28x28)

            h1 = relu(self.conv1([x]))
            h2 = relu(self.conv2([h1]))
            h3 = relu(self.conv3([h2]))
            h4 = relu(self.conv4([h3]))

            self.h1 = h1
            self.h2 = h2
            self.h3 = h3
            self.h4 = h4

            flat_h4 = h4.reshape((h4.shape[0], 32 * 4 * 4))

            mus = self.pre_mus([flat_h4])
            mus = mus.reshape((h4.shape[0], hp.n_mixtures, hp.hidden_dim))

            log_sigmas = self.pre_log_sigmas([flat_h4])
            log_sigmas = log_sigmas.reshape((h4.shape[0], hp.n_mixtures, hp.hidden_dim))

            mix = self.pre_mix([flat_h4])

            # sample mixture component
            logits = mix
            # credit to tim cooijmans for this dice trick + code
            # https://arxiv.org/pdf/1802.05098.pdf 
            y = self.sample_gumbel(logits, dropout=dropout_mix)  # note y is one-hot

            self.sample_y = y

            py = softmax(logits)
            logpy = (y * log_softmax(logits)).sum(axis=-1, keepdims=True)
            dice = torch.exp(logpy - logpy.detach())
            sample_res = (y - py).detach() * dice + py.detach()

            # sample gaussians using reparameterization trick
            vae_noise = random_state.randn(*log_sigmas.shape).astype("float32")
            torch_vae_noise = torch.tensor(vae_noise).contiguous().to(hp.use_device)

            all_gaussian_sample = mus + torch.exp(log_sigmas) * torch_vae_noise
            select_gaussian_sample = (sample_res[..., None] * all_gaussian_sample)
            reduce_gaussian_sample = select_gaussian_sample.sum(axis=1)

            post = self.post([reduce_gaussian_sample])
            post = post.reshape((h4.shape[0], 32, 4, 4)).contiguous()

            pdh1 = relu(self.transpose_convpre([post]))
            dh1 = relu(self.transpose_conv1([pdh1]))
            dh2 = relu(self.transpose_conv2([dh1]))
            dh3 = relu(self.transpose_conv3([dh2]))
            dh4 = relu(self.transpose_conv4([dh3]))
            dh5 = self.transpose_conv5([dh4])

            self.pdh1 = pdh1
            self.dh1 = dh1
            self.dh2 = dh2
            self.dh3 = dh3
            self.dh4 = dh4
            self.dh5 = dh5
            return dh5, mus, log_sigmas, mix, sample_res, all_gaussian_sample

    return Model().to(hp.use_device)

if __name__ == "__main__":
    m = build_model(hp)
    optimizer = torch.optim.Adam(m.parameters(), hp.learning_rate)
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
        data_batch, = next(itr)
        # N H W C
        data_batch = data_batch.reshape(data_batch.shape[0], 28, 28, 1)
        data_batch = data_batch.transpose(0, 3, 1, 2)
        #data_batch = data_batch / 255.
        data_batch = torch.tensor(data_batch).contiguous().to(hp.use_device)

        # output, convolutional logits, flattened logits, sampled, final sampled and reshaped

        r, mus, log_sigmas, mix, sample_res, all_gaussian_sample = m(data_batch)
        base_loss = l_fun(r.reshape(-1)[:, None], data_batch.reshape(-1)[:, None]).mean()

        # collapse to only the gaussian selected in sampling
        # is this theoretically justified?
        select_mus = sample_res[..., None] * mus
        select_log_sigmas = sample_res[..., None] * log_sigmas
        c_mus = select_mus.sum(axis=1)
        c_log_sigmas = select_log_sigmas.sum(axis=1)

        #kl = .5 * (-2 * .5 * c_log_sigmas + c_mus ** 2 + torch.exp(2 * .5 * c_log_sigmas) - 1.) 
        #kl = .5 * (-1 * c_log_sigmas + c_mus ** 2 + torch.exp(c_log_sigmas) - 1.)
        kl = -.5 * (c_log_sigmas - c_mus ** 2 - torch.exp(c_log_sigmas) + 1.)
        kl = kl.mean()
        p_mix = softmax(mix)
        # bounded between 0 and 1
        certainty = (torch.max(p_mix, axis=1)[0] - torch.min(p_mix, axis=1)[0]).mean()
        loss = base_loss + kl# + base_loss * certainty

        l = loss.cpu().data.numpy()

        optimizer.zero_grad()
        if extras["train"]:
            loss.backward()
            clipping_grad_value_(m.parameters(), hp.clip)
            optimizer.step()
        summary = {"reconstruction_loss": base_loss.cpu().data.numpy(),
                   "kl_penalty": kl.cpu().data.numpy()}
        return [l, base_loss.cpu().data.numpy(), kl.cpu().data.numpy(), certainty.cpu().data.numpy()], None, None

    s = {"model": m,
         "optimizer": optimizer,
         "hparams": hp}

    # the out-of-loop-check
    r = loop(train_itr, {"train": True}, None)
    r2 = loop(train_itr, {"train": True}, None)
    print("loop chk")
    from IPython import embed; embed(); raise ValueError()

    run_loop(loop, train_itr,
             loop, valid_itr,
             s,
             n_train_steps_per=1000,
             n_valid_steps_per=250)
