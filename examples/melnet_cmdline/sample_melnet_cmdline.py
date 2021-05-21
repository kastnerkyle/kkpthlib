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
# have to redo training argparse args as well.
parser.add_argument('--axis_splits', type=str, required=True,
                    help='string denoting the axis splits for the model, eg 2121 starting from first split to last\n')
parser.add_argument('--tier_input_tag', type=str, required=True,
                    help='the tier and data split this particular model is training, with 0,0 being the first tier, first subsplit\n')
parser.add_argument('--tier_condition_tag', type=str, default=None,
                    help='the tier and data split this particular model is conditioned by, with 0,0 being the first tier, first subsplit\n')
parser.add_argument('--size_at_depth', '-s', type=str, required=True,
                    help='size of input data in H,W str format, at the specified depth\n')
parser.add_argument('--n_layers', '-n', type=int, required=True,
                    help='number of layers the tier will have\n')
parser.add_argument('--hidden_size', type=int, required=True,
                    help='hidden dimension size for every layer\n')
parser.add_argument('--batch_size', type=int, required=True,
                    help='batch size\n')
parser.add_argument('--experiment_name', type=str, required=True,
                    help='name of overall experiment, will be combined with some of the arg input info for model save')

parser.add_argument('--stored_sampled_tier_data', action="store", nargs='*', type=str, default=None,
                    help='all previously sampled tier data, in order from "previous tier" to beginning tier from left to right')
parser.add_argument('--random_seed', '-r', type=int, default=2133,
                    help='random seed to use when sampling (default 2133)')
parser.add_argument('--data_seed', '-d', type=int, default=144,
                    help='random seed to use when seeding the sampling (default 144)')
parser.add_argument('--sample_len', type=int, default=1024,
                    help='how long of a sequence to sample (default 1024)')
parser.add_argument('--temperature', type=float, default=.9,
                    help='sampling temperature to use (default .9)')
parser.add_argument('--p_cutoff', type=float, default=.95,
                    help='cutoff to use in top p sampling (default .95)')

import builtins
# filthy global hack to pass args to submodules / models imported by importlib
args = parser.parse_args()
builtins.my_args = args

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

input_axis_split_list = [int(args.axis_splits[i]) for i in range(len(args.axis_splits))]
input_size_at_depth = [int(el) for el in args.size_at_depth.split(",")]
input_hidden_size = int(args.hidden_size)
input_n_layers = int(args.n_layers)
input_batch_size = int(args.batch_size)
input_tier_input_tag = [int(el) for el in args.tier_input_tag.split(",")]

assert len(input_size_at_depth) == 2
assert len(input_tier_input_tag) == 2
if args.tier_condition_tag is not None:
    input_tier_condition_tag = [int(el) for el in args.tier_condition_tag.split(",")]
    assert len(input_tier_condition_tag) == 2
else:
    input_tier_condition_tag = None

from kkpthlib.datasets import fetch_mnist
from kkpthlib import ListIterator
from kkpthlib.utils import split
from kkpthlib.utils import interleave_np
from kkpthlib.utils import interleave
from kkpthlib import softmax_np

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
data_batch = data_batch.transpose(0, 3, 1, 2).astype("int32").astype("float32")
data_batch = torch.tensor(data_batch).contiguous().to(hp.use_device)
data_batch = data_batch[:, 0][..., None]

all_x_splits = []
x_t = data_batch
for aa in input_axis_split_list:
    all_x_splits.append(split(x_t, axis=aa))
    x_t = all_x_splits[-1][0]
x_in = all_x_splits[::-1][input_tier_input_tag[0]][input_tier_input_tag[1]]

teacher_forced_in = x_in
if input_tier_condition_tag is None:
    pred_out = model(x_in)
else:
    cond = all_x_splits[::-1][input_tier_condition_tag[0]][input_tier_condition_tag[1]]
    pred_out = model(x_in, spatial_condition=cond)
    teacher_forced_cond = cond

teacher_forced_pred = pred_out

if input_tier_condition_tag is None:
    tag = str(args.experiment_name) + "_tier_{}_{}_sz_{}_{}".format(input_tier_input_tag[0], input_tier_input_tag[1],
                                                                    input_size_at_depth[0], input_size_at_depth[1])
else:
    tag = str(args.experiment_name) + "_tier_{}_{}_cond_{}_{}_sz_{}_{}".format(input_tier_input_tag[0], input_tier_input_tag[1],
                                                                               input_tier_condition_tag[0], input_tier_condition_tag[1],
                                                                               input_size_at_depth[0], input_size_at_depth[1])
folder_name = "sampled_" + tag
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

np_teacher_forced_argmax = np.argmax(teacher_forced_pred.cpu().data.numpy(), axis=-1)[..., None].astype("float32")
for i in range(data_batch.shape[0]):
    if input_tier_condition_tag is None:
        f, axarr = plt.subplots(1, 2)
        axarr[1].matshow(np_teacher_forced_argmax[i], cmap="gray")
    else:
        f, axarr = plt.subplots(1, 3)
        axarr[1].matshow(teacher_forced_cond[i].cpu().data.numpy().astype("float32"), cmap="gray")
        axarr[2].matshow(np_teacher_forced_argmax[i], cmap="gray")
    axarr[0].matshow(teacher_forced_in[i].cpu().data.numpy().astype("float32"), cmap="gray")

    for el_j in range(len(axarr)):
        axarr[el_j].axis("off")

    plt.savefig(folder_name + "/teacher_forced_pred{}.png".format(i))
    plt.close()

print("babby")
from IPython import embed; embed(); raise ValueError()

blank_data_batch = torch.tensor(np.zeros((data_batch.shape[0], input_size_at_depth[0], input_size_at_depth[1])).astype("float32"))
blank_data_batch = blank_data_batch[..., None]

if input_tier_condition_tag is None:
    pred_out = model(blank_data_batch)
else:
    cond = all_x_splits[::-1][input_tier_condition_tag[0]][input_tier_condition_tag[1]]
    pred_out = model(blank_data_batch, spatial_condition=cond)

teacher_conditioned_pred = pred_out

# sample tier 0
noise_random_state = np.random.RandomState(0)
blank = 0. * data_batch
lcl_blank = 0. * x_0_0
for i in range(a_0_0.shape[1]):
    for j in range(a_0_0.shape[2]):
        r = model(blank, direct_x_0_0=lcl_blank)
        pred_0_0 = r[0]
        pred_0_1 = r[1]
        pred_1_1 = r[2]
        pred_2_1 = r[3]
        pred_3_1 = r[4]

        lcl_pred = pred_0_0.cpu().data.numpy()
        eps = 1E-3
        temp = 1.
        noise = -np.log(-np.log(np.clip(noise_random_state.uniform(size=lcl_pred.shape) + eps, eps, 1. - eps)))
        np_pred = np.argmax(softmax_np(lcl_pred + noise) / temp, axis=-1)
        torch_r = torch.FloatTensor(np_pred[:, i, j][..., None])
        lcl_blank[:, i, j] = torch_r
        print("{}: {} {}".format(0, i, j))
lcl_0_0 = lcl_blank
print("sample tier 0 done")

# sample tier 1
blank = 0. * data_batch
lcl_blank = 0. * x_0_1
for i in range(a_0_1.shape[1]):
    for j in range(a_0_1.shape[2]):
        r = model(blank, direct_x_0_0=lcl_0_0, direct_x_0_1=lcl_blank)
        pred_0_0 = r[0]
        pred_0_1 = r[1]
        pred_1_1 = r[2]
        pred_2_1 = r[3]
        pred_3_1 = r[4]

        lcl_pred = pred_0_1.cpu().data.numpy()
        eps = 1E-3
        temp = 1.
        noise = -np.log(-np.log(np.clip(noise_random_state.uniform(size=lcl_pred.shape) + eps, eps, 1. - eps)))
        np_pred = np.argmax(softmax_np(lcl_pred + noise) / temp, axis=-1)
        torch_r = torch.FloatTensor(np_pred[:, i, j][..., None])
        lcl_blank[:, i, j] = torch_r
        print("{}: {} {}".format(1, i, j))
lcl_0_1 = lcl_blank
lcl_1_0 = interleave(lcl_0_0, lcl_0_1, axis=1)
print("sample tier 1 done")

# sample tier 2
blank = 0. * data_batch
lcl_blank = 0. * x_1_1
for i in range(a_1_1.shape[1]):
    for j in range(a_1_1.shape[2]):
        r = model(blank, direct_x_0_0=lcl_0_0, direct_x_0_1=lcl_0_1, direct_x_1_0=lcl_1_0, direct_x_1_1=lcl_blank)
        pred_0_0 = r[0]
        pred_0_1 = r[1]
        pred_1_1 = r[2]
        pred_2_1 = r[3]
        pred_3_1 = r[4]

        lcl_pred = pred_1_1.cpu().data.numpy()
        eps = 1E-3
        temp = 1.
        noise = -np.log(-np.log(np.clip(noise_random_state.uniform(size=lcl_pred.shape) + eps, eps, 1. - eps)))
        np_pred = np.argmax(softmax_np(lcl_pred + noise) / temp, axis=-1)
        torch_r = torch.FloatTensor(np_pred[:, i, j][..., None])
        lcl_blank[:, i, j] = torch_r
        print("{}: {} {}".format(2, i, j))
lcl_1_1 = lcl_blank
lcl_2_0 = interleave(lcl_1_0, lcl_1_1, axis=2)
print("sample tier 2 done")

# sample tier 3
blank = 0. * data_batch
lcl_blank = 0. * x_2_1
for i in range(a_2_1.shape[1]):
    for j in range(a_2_1.shape[2]):
        r = model(blank, direct_x_0_0=lcl_0_0, direct_x_0_1=lcl_0_1, direct_x_1_0=lcl_1_0, direct_x_1_1=lcl_1_1, direct_x_2_0=lcl_2_0,
                         direct_x_2_1=lcl_blank)
        pred_0_0 = r[0]
        pred_0_1 = r[1]
        pred_1_1 = r[2]
        pred_2_1 = r[3]
        pred_3_1 = r[4]

        lcl_pred = pred_2_1.cpu().data.numpy()
        eps = 1E-3
        temp = 1.
        noise = -np.log(-np.log(np.clip(noise_random_state.uniform(size=lcl_pred.shape) + eps, eps, 1. - eps)))
        np_pred = np.argmax(softmax_np(lcl_pred + noise) / temp, axis=-1)
        torch_r = torch.FloatTensor(np_pred[:, i, j][..., None])
        lcl_blank[:, i, j] = torch_r
        print("{}: {} {}".format(3, i, j))
lcl_2_1 = lcl_blank
lcl_3_0 = interleave(lcl_2_0, lcl_2_1, axis=1)
print("sample tier 3 done")

# sample tier 4
blank = 0. * data_batch
lcl_blank = 0. * x_3_1
for i in range(a_3_1.shape[1]):
    for j in range(a_3_1.shape[2]):
        r = model(blank, direct_x_0_0=lcl_0_0, direct_x_0_1=lcl_0_1, direct_x_1_0=lcl_1_0, direct_x_1_1=lcl_1_1, direct_x_2_0=lcl_2_0,
                         direct_x_2_1=lcl_2_1, direct_x_3_0=lcl_3_0, direct_x_3_1=lcl_blank)
        pred_0_0 = r[0]
        pred_0_1 = r[1]
        pred_1_1 = r[2]
        pred_2_1 = r[3]
        pred_3_1 = r[4]

        lcl_pred = pred_3_1.cpu().data.numpy()
        eps = 1E-3
        temp = 1.
        noise = -np.log(-np.log(np.clip(noise_random_state.uniform(size=lcl_pred.shape) + eps, eps, 1. - eps)))
        np_pred = np.argmax(softmax_np(lcl_pred + noise) / temp, axis=-1)
        torch_r = torch.FloatTensor(np_pred[:, i, j][..., None])
        lcl_blank[:, i, j] = torch_r
        print("{}: {} {}".format(4, i, j))
lcl_3_1 = lcl_blank
lcl_out = interleave(lcl_3_0, lcl_3_1, axis=2)
print("sample tier 4 done")

# convert to numpy for plots
lcl_0_0 = lcl_0_0.cpu().data.numpy().astype("float32")
lcl_0_1 = lcl_0_1.cpu().data.numpy().astype("float32")
lcl_1_0 = lcl_1_0.cpu().data.numpy().astype("float32")
lcl_1_1 = lcl_1_1.cpu().data.numpy().astype("float32")
lcl_2_0 = lcl_2_0.cpu().data.numpy().astype("float32")
lcl_2_1 = lcl_2_1.cpu().data.numpy().astype("float32")
lcl_3_0 = lcl_3_0.cpu().data.numpy().astype("float32")
lcl_3_1 = lcl_3_1.cpu().data.numpy().astype("float32")
lcl_out = lcl_out.cpu().data.numpy().astype("float32")

for i in range(data_batch.shape[0]):
    f, axarr = plt.subplots(1, 9)

    axarr[0].matshow(lcl_0_0[i], cmap="gray")
    axarr[1].matshow(lcl_0_1[i], cmap="gray")
    axarr[2].matshow(lcl_1_0[i], cmap="gray")
    axarr[3].matshow(lcl_1_1[i], cmap="gray")
    axarr[4].matshow(lcl_2_0[i], cmap="gray")
    axarr[5].matshow(lcl_2_1[i], cmap="gray")
    axarr[6].matshow(lcl_3_0[i], cmap="gray")
    axarr[7].matshow(lcl_3_1[i], cmap="gray")
    axarr[8].matshow(lcl_out[i], cmap="gray")

    for el_i in range(9):
        axarr[el_i].axis("off")

    plt.savefig("sampled/deep_samp{}.png".format(i))
    plt.close()
