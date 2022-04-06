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

tier0_1, tier0_2 = split(data_batch, axis=3)
tier0_1_pred, tier0_2_pred = model(data_batch)
if not os.path.exists("sampled"):
    os.makedirs("sampled")


r1 = np.clip(tier0_1_pred.cpu().data.numpy(), 0, 1)
r2 = np.clip(tier0_2_pred.cpu().data.numpy(), 0, 1)
for i in range(r1.shape[0]):
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(tier0_1[i, 0].cpu().data.numpy(), cmap="gray")
    axarr[1].imshow(r1[i, 0], cmap="gray")
    axarr[2].imshow(tier0_2[i, 0].cpu().data.numpy(), cmap="gray")
    axarr[3].imshow(r2[i, 0], cmap="gray")
    plt.savefig("sampled/tf_samp{}.png".format(i))
    plt.close()

blank = 0. * data_batch
filled1 = 0. * tier0_1
filled2 = 0. * tier0_1
for i in range(blank.shape[2]):
    for j in range(blank.shape[3]):
        if i < 14:
            blank[:, 0, i, j] = data_batch[:, 0, i, j]
        else:
            tier0_1_pred, tier0_2_pred = model(blank)
            tier0_pred = np_interleave(tier0_1_pred.cpu().data.numpy(), tier0_2_pred.cpu().data.numpy(), axis=3)
            r = np.clip(tier0_pred, 0, 1)
            torch_r = torch.FloatTensor(r[:, 0, i, j])
            blank[:, 0, i, j] = torch_r
            if j % 2 == 0:
                filled1[:, 0, i, j // 2] = torch_r
            else:
                filled2[:, 0, i, j // 2] = torch_r
        print("vert {}, {}".format(i, j))

for i in range(r1.shape[0]):
    f, axarr = plt.subplots(1, 5)
    axarr[0].imshow(data_batch[i, 0].cpu().data.numpy(), cmap="gray")
    axarr[1].imshow(blank[i, 0].cpu().data.numpy(), cmap="gray")
    axarr[2].imshow(filled1[i, 0].cpu().data.numpy(), cmap="gray")
    axarr[3].imshow(filled2[i, 0].cpu().data.numpy(), cmap="gray")
    axarr[4].imshow(torch.abs(filled2[i, 0] - filled1[i, 0]).cpu().data.numpy(), cmap="gray")
    plt.savefig("sampled/vert_samp{}.png".format(i))
    plt.close()

print("finished")
from IPython import embed; embed(); raise ValueError()
