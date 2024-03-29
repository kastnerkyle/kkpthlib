from __future__ import print_function
import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F

#from kkpthlib.datasets import fetch_mnist
from kkpthlib.datasets import fetch_binarized_mnist
from kkpthlib import Linear
from kkpthlib import BernoulliCrossEntropyFromLogits
from kkpthlib import relu
from kkpthlib import sigmoid
from kkpthlib import softmax
from kkpthlib import log_softmax
#from kkpthlib import clipping_grad_norm_
from kkpthlib import clipping_grad_value_
from kkpthlib import ListIterator
from kkpthlib import run_loop
from kkpthlib import HParams
from kkpthlib import Conv2d
from kkpthlib import Conv2dTranspose

from kkpthlib import fetch_jsb_chorales
from kkpthlib import piano_roll_from_music_json_file


hp = HParams(input_dim=48,
             hidden_dim=32,
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             learning_rate=1E-4,
             clip=3.5,
             batch_size=50,
             max_sequence_length=32,
             context_len=16,
             force_column=False,
             no_measure_mark=False,
             #max_vocabulary_size=88, # len(corpus.dictionary.idx2word)
             max_vocabulary_size=89, # len(corpus.dictionary.idx2word)
             n_epochs=1000,
             random_seed=2122)

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

"""
# loop check to get min and max note values, distribution of notes
min_note = np.inf
max_note = -np.inf
from collections import Counter
note_counter = Counter()
ranges = []
for file_list in [train_files, valid_files]:
    for file_idx in range(len(file_list)):
        pr = piano_roll_from_music_json_file(file_list[file_idx], default_velocity=120, quantization_rate=.25, n_voices=4,
                                             separate_onsets=False)
        note_counter.update(pr.ravel())
        this_range = np.max(pr[pr > 0]) - np.min(pr[pr > 0])
        ranges.append(this_range)
# looking at note dist, > 87 is rare, cap at 87
# min value 27, total note range of 61 including 0 for rest
# we could pad up to 64 if we want to
# range never goes outside 41 ... should we just pad to 48 and account for global range offset manually?
# or rather, just take out the offset since music can always be transposed... (thus undoing our data augmentation)
"""

pr_random_state = np.random.RandomState(112233)
def get_pr_batch(batch_size, time_len=48, split="train"):
    batch = []
    for _ii in range(batch_size):
        # set max range to 48
        blank_im = np.zeros((time_len, 48))
        while True:
            if split == "train":
                file_idx = pr_random_state.randint(len(train_files))
                file_list = train_files
            elif split == "valid":
                file_idx = pr_random_state.randint(len(valid_files))
                file_list = valid_files
            else:
                raise ValueError("Unknown split")
            pr = piano_roll_from_music_json_file(file_list[file_idx], default_velocity=120, quantization_rate=.25, n_voices=4,
                                                 separate_onsets=False)
            # be sure we can actually draw a minibatch from this file... should basically always be able to
            # want to only take batches from files with max note < 87
            if len(pr) >= time_len:
                break
        start_offset = pr_random_state.randint(len(pr) - time_len)
        stop_offset = start_offset + time_len
        c = pr[start_offset:stop_offset]
        assert len(c) == time_len
        # offset so notes start from 1, leave rest at 0
        c[c > 0] = (c[c > 0] - np.min(c[c > 0])) + 1
        for _j in range(c.shape[-1]):
            for _t in range(c.shape[0]):
                el = c[_t, _j]
                blank_im[_t, el] = 1.
        blank_im[:, 0] = 0.
        batch.append(blank_im)
    np_batch = np.array(batch)
    return np_batch

train_random_state = np.random.RandomState(hp.random_seed)
valid_random_state = np.random.RandomState(hp.random_seed + 1)

def get_hparams():
    return hp

def build_model(hp):
    random_state = np.random.RandomState(hp.random_seed)
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            # consider this an encoder block?
            self.conv1_1 = Conv2d([hp.input_dim], 16, kernel_size=(1, 3), strides=(1, 1), border_mode=(0, 1),
                                dilation=(1, 1),
                                random_state=random_state, name="conv1_1")
            self.conv1_2 = Conv2d([hp.input_dim], 16, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 3),
                                dilation=(1, 2),
                                random_state=random_state, name="conv1_2")
            self.conv1_4 = Conv2d([hp.input_dim], 16, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 6),
                                dilation=(1, 4),
                                random_state=random_state, name="conv1_4")
            self.conv1_8 = Conv2d([hp.input_dim], 16, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 12),
                                dilation=(1, 8),
                                random_state=random_state, name="conv1_8")
            self.conv1_16 = Conv2d([hp.input_dim], 16, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 24),
                                   dilation=(1, 16),
                                   random_state=random_state, name="conv1_16")
            self.conv1_reduce = Conv2d([16], 32, kernel_size=(1, 4), strides=(1, 2), border_mode=(0, 1), random_state=random_state,
                                       name="conv1_reduce")


            self.conv2_1 = Conv2d([32], 32, kernel_size=(1, 3), strides=(1, 1), border_mode=(0, 1),
                                dilation=(1, 1),
                                random_state=random_state, name="conv2_1")
            self.conv2_2 = Conv2d([32], 32, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 3),
                                dilation=(1, 2),
                                random_state=random_state, name="conv2_2")
            self.conv2_4 = Conv2d([32], 32, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 6),
                                dilation=(1, 4),
                                random_state=random_state, name="conv2_4")
            self.conv2_8 = Conv2d([32], 32, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 12),
                                dilation=(1, 8),
                                random_state=random_state, name="conv2_8")
            self.conv2_reduce = Conv2d([32], 64, kernel_size=(1, 4), strides=(1, 2), border_mode=(0, 1), random_state=random_state,
                                       name="conv2_reduce")


            self.conv3_1 = Conv2d([64], 64, kernel_size=(1, 3), strides=(1, 1), border_mode=(0, 1),
                                dilation=(1, 1),
                                random_state=random_state, name="conv3_1")
            self.conv3_2 = Conv2d([64], 64, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 3),
                                dilation=(1, 2),
                                random_state=random_state, name="conv3_2")
            self.conv3_4 = Conv2d([64], 64, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 6),
                                dilation=(1, 4),
                                random_state=random_state, name="conv3_4")


            """

            self.conv2 = Conv2d([16], 32, kernel_size=(4, 4), strides=(2, 2), border_mode=(1, 1), random_state=random_state, name="conv2")
            self.conv3 = Conv2d([32], 64, kernel_size=(4, 4), strides=(1, 1), border_mode=0, random_state=random_state, name="conv3")
            self.conv4 = Conv2d([64], hp.hidden_dim, kernel_size=(1, 1), strides=(1, 1), border_mode=0, random_state=random_state, name="conv4")

            self.transpose_convpre = Conv2d([hp.hidden_dim], 64, kernel_size=(1, 1), strides=(1, 1), border_mode=0, random_state=random_state, name="convpre")
            """
            self.transpose_conv2_expand = Conv2dTranspose([64], 32, kernel_size=(1, 4), strides=(1, 2), border_mode=(0, 1), random_state=random_state, name="transpose_conv2_expand")

            self.transpose_conv2_1 = Conv2dTranspose([32], 32, kernel_size=(1, 3), strides=(1, 1), border_mode=(0, 1),
                                                     dilation=(1, 1),
                                                     random_state=random_state, name="transpose_conv2_1")
            self.transpose_conv2_2 = Conv2dTranspose([32], 32, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 3),
                                                     dilation=(1, 2),
                                                     random_state=random_state, name="transpose_conv2_2")
            self.transpose_conv2_4 = Conv2dTranspose([32], 32, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 6),
                                                     dilation=(1, 4),
                                                     random_state=random_state, name="transpose_conv2_4")
            self.transpose_conv2_8 = Conv2dTranspose([32], 32, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 12),
                                                     dilation=(1, 8),
                                                     random_state=random_state, name="transpose_conv2_8")


            self.transpose_conv1_expand = Conv2dTranspose([32], 16, kernel_size=(1, 4), strides=(1, 2), border_mode=(0, 1), random_state=random_state, name="transpose_conv1_expand")

            # consider this a decoder block?
            self.transpose_conv1_1 = Conv2dTranspose([16], 16, kernel_size=(1, 3), strides=(1, 1), border_mode=(0, 1),
                                                     dilation=(1, 1),
                                                     random_state=random_state, name="transpose_conv1_1")
            self.transpose_conv1_2 = Conv2dTranspose([16], 16, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 3),
                                                     dilation=(1, 2),
                                                     random_state=random_state, name="transpose_conv1_2")
            self.transpose_conv1_4 = Conv2dTranspose([16], 16, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 6),
                                                     dilation=(1, 4),
                                                     random_state=random_state, name="transpose_conv1_4")
            self.transpose_conv1_8 = Conv2dTranspose([16], 16, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 12),
                                                     dilation=(1, 8),
                                                     random_state=random_state, name="transpose_conv1_8")
            self.transpose_conv1_16 = Conv2dTranspose([16], 16, kernel_size=(1, 4), strides=(1, 1), border_mode=(0, 24),
                                                     dilation=(1, 16),
                                                     random_state=random_state, name="transpose_conv1_16")

            self.conv_out = Conv2d([16], hp.input_dim, kernel_size=(3, 3), strides=(1, 1), border_mode=(1, 1),
                                   dilation=(1, 1),
                                   random_state=random_state, name="conv_out")

            """
            self.transpose_conv1 = Conv2dTranspose([64], 64, kernel_size=(3, 3), strides=(1, 1), border_mode=(1, 1),
                                                   dilation=(1,2)
                                                   random_state=random_state, name="transpose_conv1")

            self.transpose_conv2 = Conv2dTranspose([64], 32, kernel_size=(5, 5), strides=(1, 1), border_mode=(0, 0), random_state=random_state, name="transpose_conv2")
            self.transpose_conv3 = Conv2dTranspose([32], 32, kernel_size=(4, 4), strides=(2, 2), border_mode=(2, 2), random_state=random_state, name="transpose_conv3")
            self.transpose_conv4 = Conv2dTranspose([32], 16, kernel_size=(4, 4), strides=(2, 2), border_mode=(1, 1), random_state=random_state, name="transpose_conv4")
            self.transpose_conv5 = Conv2dTranspose([16], hp.input_dim, kernel_size=(1, 1), strides=(1, 1), border_mode=(0, 0), random_state=random_state, name="transpose_conv5")
            """

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
            h1_1 = relu(self.conv1_1([x]))
            h1_2 = relu(self.conv1_2([x]))
            h1_4 = relu(self.conv1_4([x]))
            h1_8 = relu(self.conv1_8([x]))
            h1_16 = relu(self.conv1_16([x]))
            h1 = h1_1 + h1_2 + h1_4 + h1_8 + h1_16
            h1_reduce = self.conv1_reduce([h1])

            h2_1 = relu(self.conv2_1([h1_reduce]))
            h2_2 = relu(self.conv2_2([h1_reduce]))
            h2_4 = relu(self.conv2_4([h1_reduce]))
            h2_8 = relu(self.conv2_8([h1_reduce]))
            h2 = h2_1 + h2_2 + h2_4 + h2_8
            h2_reduce = self.conv2_reduce([h2])

            h3_1 = relu(self.conv3_1([h2_reduce]))
            h3_2 = relu(self.conv3_2([h2_reduce]))
            h3_4 = relu(self.conv3_4([h2_reduce]))
            h3 = h3_1 + h3_2 + h3_4

            # discrete stuff here
            logits = h3.reshape((h3.shape[0], h3.shape[1], -1))

            self.logits = logits

            # credit to tim cooijmans for this dice trick + code
            # https://arxiv.org/pdf/1802.05098.pdf 
            y = self.sample_gumbel(logits)  # note y is one-hot

            self.sample_y = y

            py = softmax(logits)
            logpy = (y * log_softmax(logits)).sum(axis=-1, keepdims=True)
            dice = torch.exp(logpy - logpy.detach())
            sample_res = (y - py).detach() * dice + py.detach()
            sample_res = sample_res.reshape((sample_res.shape[0], sample_res.shape[1], h3.shape[-2], h3.shape[-1])).contiguous()

            self.sample_res = sample_res

            h2_d = relu(self.transpose_conv2_expand([sample_res]))
            r2_1 = relu(self.transpose_conv2_1([h2_d]))
            r2_2 = relu(self.transpose_conv2_2([h2_d]))
            r2_4 = relu(self.transpose_conv2_4([h2_d]))
            r2_8 = relu(self.transpose_conv2_8([h2_d]))
            r2 = r2_1 + r2_2 + r2_4 + r2_8

            h1_d = relu(self.transpose_conv1_expand([r2]))
            r1_1 = relu(self.transpose_conv1_1([h1_d]))
            r1_2 = relu(self.transpose_conv1_2([h1_d]))
            r1_4 = relu(self.transpose_conv1_4([h1_d]))
            r1_8 = relu(self.transpose_conv1_8([h1_d]))
            r1_16 = relu(self.transpose_conv1_16([h1_d]))
            r1 = r1_1 + r1_2 + r1_4 + r1_8 + r1_16
            o = self.conv_out([r1])
            return o, h3, logits, y, sample_res
    return Model().to(hp.use_device)

if __name__ == "__main__":
    m = build_model(hp)
    optimizer = torch.optim.Adam(m.parameters(), hp.learning_rate)
    l_fun = BernoulliCrossEntropyFromLogits()

    data_random_state = np.random.RandomState(hp.random_seed)

    def loop(itr, extras, stateful_args):
        data_batch = get_pr_batch(hp.batch_size)
        # N H W C
        data_batch = data_batch[..., None]
        # N H W C -> N C H W
        data_batch = data_batch.transpose(0, 3, 1, 2)

        # now put the pitch dimension into channel
        data_batch = data_batch[:, 0, :, :][:, :, None]

        # decide here if we keep the measure marks or no
        data_batch = torch.tensor(data_batch.astype("float32")).contiguous().to(hp.use_device)

        # output, convolutional logits, flattened logits, sampled, final sampled and reshaped
        out, latent_in, latent_flat_logits, gumbel_samp, hard_samp = m(data_batch)

        loss = l_fun(out.reshape(-1)[:, None], data_batch.reshape(-1)[:, None]).mean()
        l = loss.cpu().data.numpy()

        optimizer.zero_grad()
        if extras["train"]:
            loss.backward()
            clipping_grad_value_(m.parameters(), hp.clip)
            optimizer.step()
        return l, None, None

    s = {"model": m,
         "optimizer": optimizer,
         "hparams": hp}

    # the out-of-loop-check
    r = loop(None, {"train": True}, None)
    r2 = loop(None, {"train": True}, None)
    print("loop chk")
    from IPython import embed; embed(); raise ValueError()

    run_loop(loop, train_itr,
             loop, valid_itr,
             s,
             n_train_steps_per=1000,
             n_valid_steps_per=250)
