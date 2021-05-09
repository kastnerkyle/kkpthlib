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
import copy
import sys

import json
import os
import glob

parser = argparse.ArgumentParser(description="script {}".format(__file__))
parser.add_argument('saved_model_path', nargs='*', type=str)
parser.add_argument('--direct_saved_model_path', type=str, default=None,
                    help='location of the saved model, used the same as positional argument')
saved_model_def_default = "../{tag}.py"
parser.add_argument('--direct_saved_model_definition', type=str, default=saved_model_def_default,
                    help='location of saved model definition, a .py file')
parser.add_argument('--random_seed', '-r', type=int, default=2133,
                    help='random seed to use when sampling (default 2133)')
parser.add_argument('--data_seed', '-d', type=int, default=144,
                    help='random seed to use when seeding the sampling (default 144)')
parser.add_argument('--context_mult', '-c', type=int, default=3,
                    help='default context multiplier to combined with context size from training (default 3)')
parser.add_argument('--batch_size', '-b', type=int, default=10,
                    help='batch size for sample (default 10)')
parser.add_argument('--sample_len', '-s', type=int, default=1024,
                    help='how long of a sequence to sample (default 1024)')
parser.add_argument('--temperature', '-t', type=float, default=.9,
                    help='sampling temperature to use (default .9)')
parser.add_argument('--p_cutoff', '-p', type=float, default=.95,
                    help='cutoff to use in top p sampling (default .95)')

args = parser.parse_args()
if len(args.saved_model_path) < 1:
    if args.direct_saved_model_path is None:
        parser.print_help()
        sys.exit(1)

if args.direct_saved_model_path is not None:
    saved_model_path = args.direct_saved_model_path
else:
    saved_model_path = args.saved_model_path[0]

if not os.path.exists(saved_model_path):
    raise ValueError("Unable to find argument {} for file load! Check your paths, and be sure to remove aliases such as '~'".format(saved_model_path))

if args.direct_saved_model_definition == saved_model_def_default:
    # here we search out the saved model file from ../
    saved_model_root = os.sep.join(os.path.split(saved_model_path)[:-1])
    if not saved_model_root.endswith("saved_models"):
        raise ValueError("Default directory structure not found - explicitly pass model def to --direct_saved_model_definition !")
    model_def_root = os.sep.join(os.path.split(saved_model_root)[:-1])
    py_files = sorted(glob.glob(model_def_root + os.sep + "*.py"))
    # find the first file that has a model def
    has_model_def = []
    for pf in py_files:
        with open(pf) as f:
            t = f.read()
            if "def build_model" in t and "def get_hparams" in t:
                has_model_def.append(True)
            else:
                has_model_def.append(False)
    if not any(has_model_def):
        raise ValueError("None of the detected model files {} has 'def build_model' or 'def get_hparams', normal functions for autoload!".format(py_files))
    comb = list(zip(py_files, has_model_def))
    temp = [c for c in comb if c[1]]
    saved_model_definition_path = temp[0][0]
else:
    saved_model_definition_path = args.direct_saved_model_definition


from importlib import import_module
model_import_path, fname = saved_model_definition_path.rsplit(os.sep, 1)
sys.path.insert(0, model_import_path)
p, _ = fname.split(".", 1)
mod = import_module(p)
get_hparams = getattr(mod, "get_hparams")
build_model = getattr(mod, "build_model")
sys.path.pop(0)

hp = get_hparams()
model = build_model(hp)
model_dict = torch.load(saved_model_path, map_location=hp.use_device)
model.load_state_dict(model_dict)
model.eval()

from kkpthlib.datasets import fetch_mnist
from kkpthlib import ListIterator
from kkpthlib.utils import np_interleave
from kkpthlib.utils import interleave
from kkpthlib.utils import split

mnist = fetch_mnist()
data_random_state = np.random.RandomState(hp.random_seed)

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
train_data = mnist["data"][mnist["train_indices"]]
valid_data = mnist["data"][mnist["valid_indices"]]

train_itr = ListIterator([train_data], batch_size=hp.batch_size, random_state=data_random_state,
                         infinite_iterator=True)
valid_itr = ListIterator([valid_data], batch_size=hp.batch_size, random_state=data_random_state,
                         infinite_iterator=True)

itr = valid_itr
data_batch, = next(itr)
data_batch = data_batch.reshape(data_batch.shape[0], 28, 28, 1)
data_batch = data_batch.transpose(0, 3, 1, 2)
data_batch = data_batch / 255.0
data_batch = np.clip(data_batch + .0 * data_random_state.randn(*data_batch.shape), 0., 1.).astype("float32")
data_batch = torch.tensor(data_batch).contiguous().to(hp.use_device)

# eval mode
model.eval()

tier3_1, tier3_2 = split(data_batch, axis=3)
tier2_1, tier2_2 = split(tier3_1, axis=2)
tier1_1, tier1_2 = split(tier2_1, axis=3)
tier0_1, tier0_2 = split(tier1_1, axis=2)

r = model(data_batch)
tier0_1_pred = r[0]
tier0_2_pred = r[1]
tier1_2_pred = r[2]
tier2_2_pred = r[3]
tier3_2_pred = r[4]

if not os.path.exists("sampled"):
    os.makedirs("sampled")

# setup for final samplin
filled0_1 = 0. * tier0_1
filled0_2 = 0. * tier0_2
filled1_2 = 0. * tier1_2
filled2_2 = 0. * tier2_2
filled3_2 = 0. * tier3_2

r1 = np.clip(tier0_1_pred.cpu().data.numpy(), 0, 1)
r2 = np.clip(tier0_2_pred.cpu().data.numpy(), 0, 1)
r3 = np.clip(tier1_2_pred.cpu().data.numpy(), 0, 1)
r4 = np.clip(tier2_2_pred.cpu().data.numpy(), 0, 1)
r5 = np.clip(tier3_2_pred.cpu().data.numpy(), 0, 1)

tier0_1 = tier0_1.cpu().data.numpy()
tier0_2 = tier0_2.cpu().data.numpy()
tier1_2 = tier1_2.cpu().data.numpy()
tier2_2 = tier2_2.cpu().data.numpy()
tier3_2 = tier3_2.cpu().data.numpy()
for i in range(r1.shape[0]):
    f, axarr = plt.subplots(2, 6)
    # 7x7
    # 14x14
    # 28x28
    im_14x7 = np_interleave(r1, r2, axis=2)
    im_14x14 = np_interleave(im_14x7, r3, axis=3)
    im_28x14 = np_interleave(im_14x14, r4, axis=2)
    im_28x28 = np_interleave(im_28x14, r5, axis=3)

    gt_im_14x7 = np_interleave(tier0_1, tier0_2, axis=2)
    gt_im_14x14 = np_interleave(gt_im_14x7, tier1_2, axis=3)
    gt_im_28x14 = np_interleave(gt_im_14x14, tier2_2, axis=2)
    gt_im_28x28 = np_interleave(gt_im_28x14, tier3_2, axis=3)

    axarr[0, 0].imshow(tier0_1[i, 0], cmap="gray")
    axarr[0, 1].imshow(tier0_2[i, 0], cmap="gray")
    axarr[0, 2].imshow(gt_im_14x7[i, 0], cmap="gray")
    axarr[0, 3].imshow(gt_im_14x14[i, 0], cmap="gray")
    axarr[0, 4].imshow(gt_im_28x14[i, 0], cmap="gray")
    axarr[0, 5].imshow(gt_im_28x28[i, 0], cmap="gray")

    axarr[1, 0].imshow(r1[i, 0], cmap="gray")
    axarr[1, 1].imshow(r2[i, 0], cmap="gray")
    axarr[1, 2].imshow(im_14x7[i, 0], cmap="gray")
    axarr[1, 3].imshow(im_14x14[i, 0], cmap="gray")
    axarr[1, 4].imshow(im_28x14[i, 0], cmap="gray")
    axarr[1, 5].imshow(im_28x28[i, 0], cmap="gray")
    plt.savefig("sampled/tf_samp{}.png".format(i))
    plt.close()

for i in range(filled0_1.shape[2]):
    for j in range(filled0_1.shape[3]):
        filled0_1[:, 0, i, j] = torch.FloatTensor(tier0_1[:, 0, i, j])
        # try with gt bottom tier
        #if i < (filled0_1.shape[0] // 2):
        #    filled0_1[:, 0, i, j] = torch.FloatTensor(tier0_1[:, 0, i, j])
        #else:
        #    r = model(0. * data_batch, sub0_1=filled0_1, return0_1=True)
        #    tier0_1_pred = r[0]
        #    filled0_1[:, 0, i, j] = tier0_1_pred[:, 0, i, j]
        print("tier 0 {},{}".format(i, j))

for i in range(filled0_2.shape[2]):
    for j in range(filled0_2.shape[3]):
        r = model(0. * data_batch, sub0_1=filled0_1, sub0_2=filled0_2, return0_2=True)
        tier0_1_pred = r[0]
        tier0_2_pred = r[1]
        filled0_2[:, 0, i, j] = torch.FloatTensor(tier0_2_pred[:, 0, i, j].cpu().data.numpy())
        del tier0_1_pred
        del tier0_2_pred
        print("tier 1 {},{}".format(i, j))

tier0_pred = interleave(filled0_1, filled0_2, axis=2)
for i in range(tier0_pred.shape[2]):
    for j in range(tier0_pred.shape[3]):
        r = model(0. * data_batch, sub0_1=filled0_1, sub0_2=filled0_2, sub1_2=filled1_2, return1_2=True)
        tier0_1_pred = r[0]
        tier0_2_pred = r[1]
        tier1_2_pred = r[2]
        filled1_2[:, 0, i, j] = torch.FloatTensor(tier1_2_pred[:, 0, i, j].cpu().data.numpy())
        del tier0_1_pred
        del tier0_2_pred
        del tier1_2_pred
        print("tier 2 {},{}".format(i, j))

tier1_pred = interleave(tier0_pred, filled1_2, axis=3)
for i in range(tier1_pred.shape[2]):
    for j in range(tier1_pred.shape[3]):
        r = model(0. * data_batch, sub0_1=filled0_1, sub0_2=filled0_2, sub1_2=filled1_2, sub2_2=filled2_2, return2_2=True)
        tier0_1_pred = r[0]
        tier0_2_pred = r[1]
        tier1_2_pred = r[2]
        tier2_2_pred = r[3]
        filled2_2[:, 0, i, j] = torch.FloatTensor(tier2_2_pred[:, 0, i, j].cpu().data.numpy())
        del tier0_1_pred
        del tier0_2_pred
        del tier1_2_pred
        del tier2_2_pred
        print("tier 3 {},{}".format(i, j))

tier2_pred = interleave(tier1_pred, filled2_2, axis=2)
for i in range(tier2_pred.shape[2]):
    for j in range(tier2_pred.shape[3]):
        r = model(0. * data_batch, sub0_1=filled0_1, sub0_2=filled0_2, sub1_2=filled1_2, sub2_2=filled2_2, sub3_2=filled3_2)
        tier0_1_pred = r[0]
        tier0_2_pred = r[1]
        tier1_2_pred = r[2]
        tier2_2_pred = r[3]
        tier3_2_pred = r[4]
        filled3_2[:, 0, i, j] = torch.FloatTensor(tier3_2_pred[:, 0, i, j].cpu().data.numpy())
        del tier0_1_pred
        del tier0_2_pred
        del tier1_2_pred
        del tier2_2_pred
        del tier3_2_pred
        print("tier 4 {},{}".format(i, j))

tier3_pred = interleave(tier2_pred, filled3_2, axis=3)

tier0 = np_interleave(tier0_1, tier0_2, axis=2)
tier1 = np_interleave(tier0, tier1_2, axis=3)
tier2 = np_interleave(tier1, tier2_2, axis=2)
tier3 = np_interleave(tier2, tier3_2, axis=3)

for i in range(tier3_pred.shape[0]):
    f, axarr = plt.subplots(2, 6)
    axarr[1, 0].imshow(filled0_1[i, 0].cpu().data.numpy(), cmap="gray")
    axarr[1, 1].imshow(filled0_2[i, 0].cpu().data.numpy(), cmap="gray")
    axarr[1, 2].imshow(tier0_pred[i, 0].cpu().data.numpy(), cmap="gray")
    axarr[1, 3].imshow(tier1_pred[i, 0].cpu().data.numpy(), cmap="gray")
    axarr[1, 4].imshow(tier2_pred[i, 0].cpu().data.numpy(), cmap="gray")
    axarr[1, 5].imshow(tier3_pred[i, 0].cpu().data.numpy(), cmap="gray")

    axarr[0, 0].imshow(tier0_1[i, 0], cmap="gray")
    axarr[0, 1].imshow(tier0_2[i, 0], cmap="gray")
    axarr[0, 2].imshow(tier0[i, 0], cmap="gray")
    axarr[0, 3].imshow(tier1[i, 0], cmap="gray")
    axarr[0, 4].imshow(tier2[i, 0], cmap="gray")
    axarr[0, 5].imshow(tier3[i, 0], cmap="gray")
    plt.savefig("sampled/vert_samp{}.png".format(i))
    plt.close()

print("finished")
from IPython import embed; embed(); raise ValueError()
