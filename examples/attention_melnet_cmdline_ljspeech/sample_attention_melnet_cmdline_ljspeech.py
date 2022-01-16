from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul

import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F
import re
import copy
import sys
import time

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

# have to redo training argparse args as well for load setup
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
parser.add_argument('--cell_type', type=str, required=True,
                    help='melnet cell type\n')
parser.add_argument('--optimizer', type=str, required=True,
                    help='optimizer type\n')
parser.add_argument('--learning_rate', type=str, required=True,
                    help='learning rate\n')
parser.add_argument('--real_batch_size', type=int, required=True,
                    help='real batch size\n')
parser.add_argument('--virtual_batch_size', type=int, required=True,
                    help='virtual batch size\n')
parser.add_argument('--experiment_name', type=str, required=True,
                    help='name of overall experiment, will be combined with some of the arg input info for model save')
parser.add_argument('--previous_saved_model_path', type=str, default=None,
                    help='path to previously saved checkpoint model')
parser.add_argument('--previous_saved_optimizer_path', type=str, default=None,
                    help='path to previously saved optimizer')
parser.add_argument('--n_previous_save_steps', type=str, default=None,
                    help='number of save steps taken for previously run model, used to "replay" the data generator back to the same point')


parser.add_argument('--terminate_early_attention_plot', action="store_true",
                    help='flag to terminate early for attention plotting meta-scripts')

parser.add_argument('--batch_skips', type=int, default=0,
                    help='number of batches to skip before sampling - allows us to sample different examples!')
parser.add_argument('--use_longest', action="store_true",
                    help='flag to use the longest of N examples for sampling due to biasing')
parser.add_argument('--use_sample_index', type=str, default="0,0",
                    help='flag to use for deterministic sampling of same entry')

parser.add_argument('--custom_conditioning_json', type=str, default=None,
                    help="Path to Gentle formatted json for custom conditioning")
parser.add_argument('--attention_early_termination_file', type=str, default=None,
                    help="Path to attention termination file for reducing sample time")

parser.add_argument('--output_dir', type=str, default=None,
                    help='base directory to output sampled data')
parser.add_argument('--stored_sampled_tier_data', type=str, default=None,
                    help='all previously sampled tier data, in order from beginning tier to previous from left to right. Last array assumed to be the conditioning input. comma separated string, should be path to unnormalized data')

parser.add_argument('--random_seed', '-r', type=int, default=2133,
                    help='random seed to use when sampling (default 2133)')
parser.add_argument('--data_seed', '-d', type=int, default=144,
                    help='random seed to use when seeding the sampling (default 144)')
parser.add_argument('--sample_len', type=int, default=1024,
                    help='how long of a sequence to sample (default 1024)')

parser.add_argument('--temperature', type=float, default=.9,
                    help='sampling temperature to use (default .9)')
parser.add_argument('--bias_data_start', type=float, default=.5,
                    help='amount of data to use from gt biasing data')
parser.add_argument('--p_cutoff', type=float, default=.5,
                    help='cutoff to use in top p sampling (default .5)')
parser.add_argument('--samples_to_average', type=int, default=10,
                    help='number of samples to average in order to form estimated sample')


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

input_axis_split_list = [int(args.axis_splits[i]) for i in range(len(args.axis_splits))]
input_size_at_depth = [int(el) for el in args.size_at_depth.split(",")]
input_hidden_size = int(args.hidden_size)
input_n_layers = int(args.n_layers)
input_real_batch_size = int(args.real_batch_size)
input_batch_skips =int(args.batch_skips)
input_virtual_batch_size = int(args.virtual_batch_size)
input_tier_input_tag = [int(el) for el in args.tier_input_tag.split(",")]
input_use_sample_index = [int(el) for el in args.use_sample_index.split(",")]
if args.custom_conditioning_json is not None:
    input_custom_conditioning_json = str(args.custom_conditioning_json)
else:
    input_custom_conditioning_json = None

if args.attention_early_termination_file is not None:
    input_attention_early_termination_file = str(args.attention_early_termination_file)
else:
    input_attention_early_termination_file = None

if args.output_dir is None:
    raise ValueError("No output_dir passed! Required for sampling")
input_output_dir = args.output_dir if args.output_dir[-1] == "/" else args.output_dir + "/"

assert len(input_size_at_depth) == 2
assert len(input_tier_input_tag) == 2
if args.tier_condition_tag is not None:
    input_tier_condition_tag = [int(el) for el in args.tier_condition_tag.split(",")]
    assert len(input_tier_condition_tag) == 2
else:
    input_tier_condition_tag = None

if args.stored_sampled_tier_data is not None:
    stored_sampled_tier_data_paths = args.stored_sampled_tier_data.split(",")
    if len(stored_sampled_tier_data_paths) == 1:
        input_stored_conditioning = np.load(stored_sampled_tier_data_paths[-1])
    else:
        #x_in_np = all_x_splits[::-1][input_tier_input_tag[0]][input_tier_input_tag[1]]
        # need to figure out how to combine all the saved data into one input
        built_conditioning = None
        # remember there is some kind of eff by 1 / reverse
        for _n, el in enumerate(stored_sampled_tier_data_paths):
            if built_conditioning is None:
                built_conditioning = np.load(el)
            else:
                next_conditioning = np.load(el)
                comb_dims = (next_conditioning.shape[0],
                             built_conditioning.shape[1] + next_conditioning.shape[1],
                             built_conditioning.shape[2] + next_conditioning.shape[2],
                             next_conditioning.shape[-1])
                # reverse split list since it is done in depth order (shallow to deep)
                this_split = input_axis_split_list[::-1][_n]
                if this_split == 1:
                    buffer_np = np.zeros((input_real_batch_size, next_conditioning.shape[1], comb_dims[2], 1)).astype("float32")
                    buffer_np[:, :, ::2, :] = built_conditioning
                    buffer_np[:, :, 1::2, :] = next_conditioning
                    input_stored_conditioning = buffer_np.astype("float32")
                elif this_split == 2:
                    buffer_np = np.zeros((input_real_batch_size, comb_dims[1], next_conditioning.shape[2], 1)).astype("float32")
                    buffer_np[:, ::2, :, :] = built_conditioning
                    buffer_np[:, 1::2, :, :] = next_conditioning
                else:
                    raise ValueError("Unknown split value {} for split index {} from (reversed) split list {}".format(this_split, _n, input_axis_split_list[::-1]))
                built_conditioning = buffer_np.astype("float32")
        input_stored_conditioning = copy.deepcopy(built_conditioning)

        """
        if input_axis_split_list[input_tier_input_tag[0]] == 2:
            buffer_np = np.zeros((input_real_batch_size, input_size_at_depth[0], input_size_at_depth[1], 1)).astype("float32")
            input_stored_conditioning_a = np.load(stored_sampled_tier_data_paths[-2])
            input_stored_conditioning_b = np.load(stored_sampled_tier_data_paths[-1])
            buffer_np[:, ::2, :, :] = input_stored_conditioning_a
            buffer_np[:, 1::2, :, :] = input_stored_conditioning_b
            input_stored_conditioning = buffer_np.astype("float32")
        elif input_axis_split_list[input_tier_input_tag[0]] == 1:
            buffer_np = np.zeros((input_real_batch_size, input_size_at_depth[0], input_size_at_depth[1], 1)).astype("float32")
            input_stored_conditioning_a = np.load(stored_sampled_tier_data_paths[-2])
            input_stored_conditioning_b = np.load(stored_sampled_tier_data_paths[-1])
            buffer_np[:, :, ::2, :] = input_stored_conditioning_a
            buffer_np[:, :, 1::2, :] = input_stored_conditioning_b
            input_stored_conditioning = buffer_np.astype("float32")
        """
    if input_stored_conditioning.shape[1] != input_size_at_depth[0] or input_stored_conditioning.shape[2] != input_size_at_depth[1]:
        err_str = "stored sampled tier data passed via --stored_stampled_tier_data={}".format(args.stored_sampled_tier_data)
        err_str += "\n"
        err_str += "does not have the correct shape after processing!"
        err_str += "\n"
        err_str += "combined shape of numpy arrays should match --size_at_depth={}".format(input_size_at_depth)
        err_str += "\n"
        err_str += "however, current combined shape is {},{}".format(input_stored_conditioning.shape[1], input_stored_conditioning.shape[2])
        raise ValueError(err_str)
else:
    input_stored_conditioning = None


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

use_half = False
if use_half:
    model.half()  # convert to half precision
    for layer in model.modules():
        layer.half()
    [a.half() for a in model.parameters()]

model_dict = torch.load(saved_model_path, map_location=hp.use_device)
model.load_state_dict(model_dict)
model.eval()

from kkpthlib.datasets import EnglishSpeechCorpus
from kkpthlib.utils import split
from kkpthlib.utils import split_np
from kkpthlib.utils import interleave_np
from kkpthlib.utils import interleave
from kkpthlib import softmax_np

data_random_state = np.random.RandomState(hp.random_seed)
folder_base = "/usr/local/data/kkastner/ljspeech_cleaned"
fixed_minibatch_time_secs = 4
fraction_train_split = .9
speech = EnglishSpeechCorpus(metadata_csv=folder_base + "/metadata.csv",
                             wav_folder=folder_base + "/wavs/",
                             alignment_folder=folder_base + "/alignment_json/",
                             fixed_minibatch_time_secs=fixed_minibatch_time_secs,
                             extract_subsequences=False,
                             train_split=fraction_train_split,
                             random_state=data_random_state)

dataset_name = folder_base.split("/")[-1]
dataset_max_limit = fixed_minibatch_time_secs
axis_splits_str = "".join([str(aa) for aa in input_axis_split_list])
axis_size_str = "{}x{}".format(input_size_at_depth[0], input_size_at_depth[1])
tier_depth_str = str(input_tier_input_tag[0])

# hardcoded per-dimension mean and std for mel data from the training iterator, read from a file
full_cached_mean_std_name_for_experiment = "{}_max{}secs_{}splits_{}sz_{}tierdepth_mean_std.npz".format(dataset_name,
                                                                                                        dataset_max_limit,
                                                                                                        axis_splits_str,
                                                                                                        axis_size_str,
                                                                                                        tier_depth_str)
mean_std_cache = os.getcwd() + "/mean_std_cache/"
mean_std_path = mean_std_cache + full_cached_mean_std_name_for_experiment
if not os.path.exists(mean_std_path):
    raise ValueError("Unable to find cached mean std info at {}".format(mean_std_path))

if input_batch_skips > 0:
    for _ in range(input_batch_skips):
        tmp = speech.get_valid_utterances(hp.real_batch_size)

# TODO: fix?
if args.use_longest:
    # sample 50 minibatches, find longest N examples of that...
    print("Performing length selection to choose base samples for biasing")
    valid_el = None
    kept_indices = [[0] * hp.real_batch_size, list(range(hp.real_batch_size))]
    itr_offset = 0
    for _ in range(50):
        this_valid_el = speech.get_valid_utterances(hp.real_batch_size)
        itr_offset += 1
        if valid_el is None:
            valid_el = this_valid_el
        else:
            for candidate in range(len(this_valid_el)):
                for kept in range(len(valid_el)):
                    if this_valid_el[candidate][2].shape[0] > valid_el[kept][2].shape[0]:
                        valid_el[kept] = this_valid_el[candidate]
                        kept_indices[0][kept] = itr_offset
                        kept_indices[1][kept] = candidate
                        break
elif input_use_sample_index[0] != 0 or input_use_sample_index[1] != 0:
    # used this to get names of minibatch examples to form mini dataset
    # not used right now but may be logged in the future
    store_valid_els = []
    for _ in range(input_use_sample_index[0]):
        this_valid_el = speech.get_valid_utterances(hp.real_batch_size)
        store_valid_els.append(this_valid_el)
    names = []
    for _ii in range(len(store_valid_els)):
        for _jj in range(len(store_valid_els[_ii])):
            n = list(store_valid_els[_ii][_jj][3].keys())[0]
            names.append(n)
    valid_el = [this_valid_el[input_use_sample_index[1]]] * hp.real_batch_size
else:
    valid_el = speech.get_valid_utterances(hp.real_batch_size)


cond_seq_data_batch, cond_seq_mask, data_batch, data_mask = speech.format_minibatch(valid_el)
batch_norm_flag = 1.

# this is weird, need 2 load calls for conditional generator...
speech.load_mean_std_from_filepath(mean_std_path)
saved_mean = speech.cached_mean_vec_[None, None, :, None]
saved_std = speech.cached_std_vec_[None, None, :, None]

if use_half:
    model.half()  # convert to half precision
    for layer in model.modules():
        layer.half()
    [a.half() for a in model.parameters()]

torch_cond_seq_data_batch = torch.tensor(cond_seq_data_batch[..., None]).contiguous().to(hp.use_device)
torch_cond_seq_data_mask = torch.tensor(cond_seq_mask).contiguous().to(hp.use_device)

x_mask_t = data_mask[..., None, None] + 0. * data_batch[..., None]
x_t = data_batch[..., None]
divisors = [2, 4, 8]
max_frame_count = x_t.shape[1]
for di in divisors:
    # nearest divisible number above, works because largest divisor divides by smaller
    # we need something that has a length in time (frames) divisible by 2 4 and 8 due to the nature of melnet
    # same for frequency but frequency is a power of 2 so no need to check it
    q = int(max_frame_count / di)
    if float(max_frame_count / di) == int(max_frame_count / di):
        max_frame_count = di * q
    else:
        max_frame_count = di * (q + 1)
assert max_frame_count == int(max_frame_count)

axis_splits = input_axis_split_list
splits_offset = 0
axis1_m = [2 for a in str(axis_splits)[splits_offset:] if a == "1"]
axis2_m = [2 for a in str(axis_splits)[splits_offset:] if a == "2"]
axis1_m = reduce(mul, axis1_m)
axis2_m = reduce(mul, axis2_m)

x_in_np = x_t[:, ::axis1_m, ::axis2_m]
x_mask_in_np = x_mask_t[:, ::axis1_m, ::axis2_m]
x_in_np = (x_in_np - saved_mean) / saved_std

x_in = torch.tensor(x_in_np).contiguous().to(hp.use_device)
x_mask_in = torch.tensor(x_mask_in_np).contiguous().to(hp.use_device)

if use_half:
    torch_cond_seq_data_batch = torch.tensor(cond_seq_data_batch[..., None]).contiguous().to(hp.use_device).half()
    torch_cond_seq_data_mask = torch.tensor(cond_seq_mask).contiguous().to(hp.use_device).half()
    x_in = torch.tensor(x_in_np).contiguous().to(hp.use_device).half()
    x_mask_in = torch.tensor(x_mask_in_np).contiguous().to(hp.use_device).half()

if input_tier_condition_tag is None:
    # no noise here in pred
    with torch.no_grad():
        pred_out = model(x_in, x_mask=x_mask_in,
                         memory_condition=torch_cond_seq_data_batch,
                         memory_condition_mask=torch_cond_seq_data_mask,
                         batch_norm_flag=batch_norm_flag)
else:
    cond_np = all_x_splits[::-1][input_tier_condition_tag[0]][input_tier_condition_tag[1]]
    # conditioning input currently unnormalized
    if input_stored_conditioning is not None:
        cond_np = input_stored_conditioning
    cond = torch.tensor(cond_np).contiguous().to(hp.use_device)
    if use_half:
        cond = torch.tensor(cond_np).contiguous().to(hp.use_device).half()
    with torch.no_grad():
        pred_out = model(x_in, x_mask=x_mask_in,
                         spatial_condition=cond,
                         batch_norm_flag=batch_norm_flag)
#print("testing sample")
#from IPython import embed; embed(); raise ValueError()

import matplotlib.pyplot as plt

if args.terminate_early_attention_plot:
    # take output dir directly for terminate early attention plot
    folder = input_output_dir
    if not os.path.exists(input_output_dir):
        os.mkdir(input_output_dir)

    if not os.path.exists(folder):
        os.mkdir(folder)

    teacher_forced_pred = pred_out
    teacher_forced_attn = model.attention_alignment

    for _i in range(hp.real_batch_size):
        mel_cut = int(x_mask_in[_i, :, 0, 0].cpu().data.numpy().sum())
        text_cut = int(torch_cond_seq_data_mask[:, _i].cpu().data.numpy().sum())
        # matshow vs imshow?
        this_att = teacher_forced_attn[:, _i, 0].cpu().data.numpy()[:mel_cut, :text_cut]
        this_att = this_att.astype("float32")
        plt.imshow(this_att)
        plt.title("{}\n{}\n".format("/".join(saved_model_path.split("/")[:-1]), saved_model_path.split("/")[-1]))
        plt.savefig(folder + "attn_{}.png".format(_i))
        plt.close()

    import sys
    print("Terminating early after only plotting attention due to commandline flag --terminate_early_attention_plot")
    sys.exit()

def fast_sample(x, x_mask=None,
                   spatial_condition=None,
                   memory_condition=None, memory_condition_mask=None,
                   bias_boundary="default",
                   batch_norm_flag=0.,
                   verbose=True):
    frst = time.time()
    new_x = copy.deepcopy(x)
    x = new_x
    if spatial_condition is None:
        assert memory_condition is not None
        mem, mem_e = model.embed_text(memory_condition)

        #mem_conv = self.conv_text([mem], batch_norm_flag)
        # mask based on the conditioning mask

        #mem_conv = mem_conv * memory_condition_mask[..., None]

        # use mask in BiLSTM
        mem_lstm = model.bilstm_text([mem], input_mask=memory_condition_mask)
        # x currently batch, time, freq, 1
        # mem time, batch, feat
        # feed mask for attention calculations as well
        #mn_out, alignment, attn_extras = model.mn_t([x], memory=mem_lstm, memory_mask=memory_condition_mask)
        # centralized stack means we cannot do better than frame cuts
        time_index = 0
        freq_index = 0
        is_initial_step = True
        # b t f feat
        mem_lstm = mem_lstm
        memory_condition_mask = memory_condition_mask
        if bias_boundary == "default":
            start_time_index = x.shape[1] // 2
            start_freq_index = 0
        else:
            start_time_index = int(bias_boundary)
            start_freq_index = 0

        x_a = x[:, :start_time_index]
        mn_out, alignment, attn_extras = model.mn_t.sample([x_a], time_index=time_index, freq_index=freq_index,
                                                                  is_initial_step=is_initial_step,
                                                                  memory=mem_lstm, memory_mask=memory_condition_mask)
        is_initial_step = False

        mem_lstm = mem_lstm
        memory_condition_mask = memory_condition_mask
        x[:, start_time_index:, :] *= 0

        if verbose:
            print("start sample step")
        max_time_step = x.shape[1]
        max_freq_step = x.shape[2]

        total_alignment = alignment
        total_extras = attn_extras

        for _ii in range(start_time_index, max_time_step):
            for _jj in range(start_freq_index, max_freq_step):
                mn_out, alignment, attn_extras = model.mn_t.sample([x], time_index=_ii, freq_index=_jj,
                                                                        is_initial_step=is_initial_step,
                                                                        memory=mem_lstm, memory_mask=memory_condition_mask)
                x[:, _ii, _jj, 0] = mn_out.squeeze()
                if verbose:
                    print("sampled index {},{} out of total size ({},{})".format(_ii, _jj, max_time_step, max_freq_step))
            total_alignment = torch.cat((total_alignment, alignment[None]), dim=0)
            total_extras.append(attn_extras)
        model.attention_alignment = total_alignment
        model.attention_extras = total_extras
    else:
        x = x[0:1, 0:1]
        spatial_condition = spatial_condition[0:1]
        mn_out = model.mn_t.sample([x], list_of_spatial_conditions=[spatial_condition])
    fin = time.time()
    print("fast sampling complete, time {} sec".format(fin - frst))
    return x

"""
if input_tier_condition_tag is None:
    # no noise here in pred
    with torch.no_grad():
        pred_out = fast_sample(x_in, x_mask=x_mask_in,
                                     memory_condition=torch_cond_seq_data_batch,
                                     memory_condition_mask=torch_cond_seq_data_mask,
                                     batch_norm_flag=batch_norm_flag)
else:
    cond_np = all_x_splits[::-1][input_tier_condition_tag[0]][input_tier_condition_tag[1]]
    # conditioning input currently unnormalized
    if input_stored_conditioning is not None:
        cond_np = input_stored_conditioning
    cond = torch.tensor(cond_np).contiguous().to(hp.use_device)
    if use_half:
        cond = torch.tensor(cond_np).contiguous().to(hp.use_device).half()
    with torch.no_grad():
        pred_out = tst(x_in, x_mask=x_mask_in,
                             spatial_condition=cond,
                             batch_norm_flag=batch_norm_flag)
"""

"""
unnormalized = pred_out.cpu().data.numpy() * saved_std + saved_mean
_i = 0
plt.imshow(unnormalized[_i, :, :, 0])
plt.savefig("tmpfast.png")

unnormalized = x_in.cpu().data.numpy() * saved_std + saved_mean
_i = 0
plt.imshow(unnormalized[_i, :, :, 0])
plt.savefig("tmpfast_orig.png")
"""

def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda : one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot

def sample_dml(l, n_mix=10, only_mean=True, deterministic=True, sampling_temperature=1.):
    sampling_temperature = float(sampling_temperature)
    nr_mix = n_mix
    # Pytorch ordering
    #l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])
    # sample mixture indicator from softmax
    noise = torch.FloatTensor(logit_probs.size())
    if l.is_cuda : noise = noise.cuda()
    noise.uniform_(1e-5, 1. - 1e-5)
    # hack to make deterministic JRH
    # could also just take argmax of logit_probs
    if deterministic or only_mean:
        # make temp small so logit_probs dominates equation
        sampling_temperature = 1e-6
    # sampling temperature from kk
    # https://gist.github.com/kastnerkyle/ea08e1aed59a0896e4f7991ac7cdc147
    # discussion on gumbel sm sampling -
    # https://github.com/Rayhane-mamah/Tacotron-2/issues/155
    noise = (logit_probs.data/sampling_temperature) - torch.log(- torch.log(noise))
    _, argmax = noise.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda : u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    # hack to make deterministic
    if deterministic:
        u= u*0.0+0.5
    if only_mean:
        x = means
    else:
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    out = torch.clamp(torch.clamp(x,min=-1.),max=1.)
    # put back in Pytorch ordering
    #out = out.permute(0, 3, 1, 2)
    return out

folder = input_output_dir + "teacher_forced_images/"
if not os.path.exists(input_output_dir):
    os.mkdir(input_output_dir)

if not os.path.exists(folder):
    os.mkdir(folder)

import json
from collections import OrderedDict

if valid_el is not None:
    sample_info_path = folder + "text_info.json"
    cleaned_valid_el = valid_el[0][3]
    with open(sample_info_path, 'w') as f:
        json_string = json.dumps(cleaned_valid_el, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        f.write(json_string)

for _i in range(hp.real_batch_size):
    teacher_forced_pred = pred_out

    reduced_mel_cut = int(x_mask_in[_i, :, 0, 0].cpu().data.numpy().sum())
    full_mel_cut = int(data_mask[_i].sum())

    this_x_in = x_in[_i][None].cpu().data.numpy()
    unnormalized = this_x_in * saved_std + saved_mean
    plt.imshow(unnormalized[0, :reduced_mel_cut, :, 0])
    plt.savefig(folder + os.sep + "small_x{}.png".format(_i))
    plt.close()

    # no normalization on initial data
    unnormalized_full = data_batch[_i, :full_mel_cut, :]
    plt.imshow(unnormalized_full)
    plt.savefig(folder + os.sep + "data_x{}.png".format(_i))
    plt.close()

    teacher_forced_pred = pred_out
    teacher_forced_attn = model.attention_alignment

    if input_tier_condition_tag is None:
        for _i in range(hp.real_batch_size):
            mel_cut = int(x_mask_in[_i, :, 0, 0].cpu().data.numpy().sum())
            text_cut = int(torch_cond_seq_data_mask[:, _i].cpu().data.numpy().sum())
            # matshow vs imshow?
            this_att = teacher_forced_attn[:, _i, 0][:mel_cut, :text_cut]
            this_att = this_att.cpu().data.numpy().astype("float32")
            plt.imshow(this_att)
            plt.title("{}\n{}\n".format("/".join(saved_model_path.split("/")[:-1]), saved_model_path.split("/")[-1]))
            plt.savefig(folder + os.sep + "attn_{}.png".format(_i))
            plt.close()
    else:
        plt.imshow(cond_np[_i, :, :, 0])
        plt.savefig(folder + os.sep + "cond_small_x{}.png".format(_i))
        plt.close()

time_len = x_in_np.shape[1]
if len(valid_el) > 1:
    print("Multiple output samples detected, using default sampling bias of 50% of input length")
    bias_til = time_len // 2
else:
    # try to do a smart split
    timing_info = valid_el[0][3][list(valid_el[0][3].keys())[0]]
    words = [w for w in timing_info["full_alignment"]["words"]]
    # this logic should find the biggest gap with at least 1/2 a second priming
    # in the first half of the overall word sequence
    diffs = np.array([w["start"] for w in words[1:]]) - np.array([w["end"] for w in words[:-1]])
    ends = np.array([w["end"] for w in words])
    midpoint = ends[-1] / 2.
    first_half = ends <= midpoint
    gt_sec = ends >= .5
    valid = first_half & gt_sec
    gap_sort = np.argsort(diffs)[::-1]
    chosen_index = None
    for g in gap_sort:
        if valid[g]:
            chosen_index = g
        else:
            continue
    if chosen_index is None:
        bias_til = time_len // 2
    else:
        # now we need to figure out a time in frames which matches where the gap is
        fs = float(speech.sample_rate)
        stft_sz = speech.stft_size
        stft_step = speech.stft_step
        time_step_per_frame = 1./fs * stft_step
        # due to overlap the neighbors might be valid too but for now just use the chosen index
        full_size_frame_index = ends[chosen_index] / time_step_per_frame
        full_n_frames = axis1_m * input_size_at_depth[0]
        time_downsample_ratio = full_n_frames / input_size_at_depth[0] # should always be integer value
        downsampled_frame_index = full_size_frame_index / float(time_downsample_ratio)
        bias_til = int(downsampled_frame_index)

remaining_steps = time_len - bias_til
sample_buffer = 0. * x_in_np
# 1. means value is used, 0. means it is masked. Can be different with transformers...
sample_mask = 0. * x_mask_in_np + 1.
sample_buffer[:, :bias_til] = x_in_np[:, :bias_til]
original_buffer = copy.deepcopy(x_in_np)

if input_custom_conditioning_json is not None:
    # now we need to construct the combined input conditioning with the bias 
    with open(input_custom_conditioning_json, "r") as f:
        custom_conditioning_alignment = json.load(f)

    frank_el = copy.deepcopy(valid_el)
    if chosen_index == None:
        raise ValueError("automatic bias selection failed and chose arbitrary midpoint. Need word-boundary cuts to use custom_conditioning, try another index with (e.g.) --use_sample_index=0,0")
    # now we replace the old one with the merged and regenerate the minibatch
    frank_info = []
    # sample rate
    frank_info.append(frank_el[0][0])
    # blank wav to be sure no info leaks
    frank_info.append(0. * frank_el[0][1])
    # blank mel spec for the same reason
    frank_info.append(0. * frank_el[0][2])
    # now we need to blend the words of the 2 sequences and make a new el
    tmp = frank_el[0][3]
    new = custom_conditioning_alignment
    k = list(tmp.keys())[0]
    # first lets find the word where we need to split the bias and the followup
    pre_words = tmp[k]["full_alignment"]["words"][:chosen_index + 1]
    next_pre_word = tmp[k]["full_alignment"]["words"][chosen_index + 1]

    # handle offset correction
    pre_offset = next_pre_word["startOffset"]
    pre_t = next_pre_word["start"]
    post_words = new["words"]
    for _i in range(len(post_words)):
        post_words[_i]["start"] = post_words[_i]["start"] + pre_t
        post_words[_i]["end"] = post_words[_i]["end"] + pre_t
        post_words[_i]["startOffset"] = post_words[_i]["startOffset"] + pre_offset
        post_words[_i]["endOffset"] = post_words[_i]["endOffset"] + pre_offset
    # now put the words together
    comb_words = pre_words + post_words

    # now to splice the transcripts, find the first occurence of the word that would have followed, split on that
    search_key = next_pre_word["word"]
    matches = [(m.start(0), m.end(0)) for m in re.finditer(search_key, tmp[k]["transcript"])]
    matches_in_pre = np.where([1 if w["word"] == search_key else 0 for w in pre_words])[0]
    if len(matches_in_pre) > 0:
        this_match = matches[len(matches_in_pre)]
    else:
        this_match = matches[0]
    sl = this_match[0]
    r = tmp[k]["transcript"][:sl] + new["transcript"]
    tmp[k]["transcript"] = r
    tmp[k]["full_alignment"]["transcript"] = r
    tmp[k]["full_alignment"]["words"] = comb_words
    frank_info.append(tmp)
    old_valid_el = valid_el
    valid_el = [tuple(frank_info)]

    sample_info_path = folder + "bias_merged_text_info.json"
    cleaned_valid_el = valid_el[0][3]
    with open(sample_info_path, 'w') as f:
        json_string = json.dumps(cleaned_valid_el, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        f.write(json_string)

    cond_seq_data_batch, cond_seq_mask, _, __ = speech.format_minibatch(valid_el)

    torch_cond_seq_data_batch = torch.tensor(cond_seq_data_batch[..., None]).contiguous().to(hp.use_device)
    torch_cond_seq_data_mask = torch.tensor(cond_seq_mask).contiguous().to(hp.use_device)

if input_attention_early_termination_file == None:
    last_sil_frame_scaled = np.inf
else:
    with open(input_attention_early_termination_file, "r") as f:
        lines = f.readlines()
        last_sil_frame = int(float(lines[1].strip().split(":")[1]))
        last_sil_resolution = int(float(lines[2].strip().split(":")[1]))
        full_n_frames = axis1_m * input_size_at_depth[0]
        this_resolution = full_n_frames / input_size_at_depth[0] # should always be integer value
        # add in extra frame(s) based on the upsampling resolution due to ambiguity
        last_sil_frame_scaled = last_sil_frame * int(last_sil_resolution / this_resolution) + (int(last_sil_resolution / this_resolution) - 1)

import time
begin_time = time.time()
sample_buffer[:, :max(0, bias_til - 5)] = x_in_np[:, :max(0, bias_til - 5)]
batch_norm_flag = 1.
sample_buffer = torch.tensor(sample_buffer).contiguous().to(hp.use_device)
sample_mask = torch.tensor(sample_mask).contiguous().to(hp.use_device)

with torch.no_grad():
    pred_out = fast_sample(sample_buffer,
                           x_mask=sample_mask,
                           memory_condition=torch_cond_seq_data_batch,
                           memory_condition_mask=torch_cond_seq_data_mask,
                           # boundary?
                           bias_boundary=max(0, bias_til - 5),
                           batch_norm_flag=batch_norm_flag)
sample_buffer = pred_out
'''
for time_step in range(remaining_steps):
    for mel_step in range(x_in_np.shape[2]):
        cur_time = time.time()
        this_step = bias_til + time_step
        if this_step > last_sil_frame_scaled:
            print("step {},{} of {},{} -> bypassed due to attention last frame of {}".format(this_step, mel_step, time_len, x_in_np.shape[2], last_sil_frame_scaled))
            # if we have passed the attention termination, just copy the conditioning
            cond_np = all_x_splits[::-1][input_tier_condition_tag[0]][input_tier_condition_tag[1]]
            if input_stored_conditioning is not None:
                cond_np = input_stored_conditioning
            # cond is unnormalized, predictions are normalized
            sample_buffer[:, this_step, mel_step] = ((cond_np - saved_mean) / saved_std)[:, this_step, mel_step]
            continue
        print("step {},{} of {},{}".format(this_step, mel_step, time_len, x_in_np.shape[2]))

        """
        all_x_splits = []
        x_t = data_batch[..., None]
        for aa in input_axis_split_list:
            all_x_splits.append(split_np(x_t, axis=aa))
            x_t = all_x_splits[-1][0]
        x_in_np = all_x_splits[::-1][input_tier_input_tag[0]][input_tier_input_tag[1]]
        x_in_np = (x_in_np - saved_mean) / saved_std

        all_x_mask_splits = []
        # broadcast mask over frequency so we can downsample
        x_mask_t = data_mask[..., None, None] + 0. * data_batch[..., None]
        for aa in input_axis_split_list:
            all_x_mask_splits.append(split_np(x_mask_t, axis=aa))
            x_mask_t = all_x_mask_splits[-1][0]
        x_mask_in_np = all_x_mask_splits[::-1][input_tier_input_tag[0]][input_tier_input_tag[1]]

        x_in = torch.tensor(x_in_np).contiguous().to(hp.use_device)
        x_mask_in = torch.tensor(x_mask_in_np).contiguous().to(hp.use_device)
        """
        x_in = torch.tensor(sample_buffer).contiguous().to(hp.use_device)
        x_mask_in = torch.tensor(sample_mask).contiguous().to(hp.use_device)

        if use_half:
            torch_cond_seq_data_batch = torch.tensor(cond_seq_data_batch[..., None]).contiguous().to(hp.use_device).half()
            torch_cond_seq_data_mask = torch.tensor(cond_seq_mask).contiguous().to(hp.use_device).half()
            x_in = torch.tensor(x_in_np).contiguous().to(hp.use_device).half()
            x_mask_in = torch.tensor(x_mask_in_np).contiguous().to(hp.use_device).half()

        if input_tier_condition_tag is None:
            # no noise here in pred
            with torch.no_grad():
                pred_out = model(x_in, x_mask=x_mask_in,
                                 memory_condition=torch_cond_seq_data_batch,
                                 memory_condition_mask=torch_cond_seq_data_mask,
                                 batch_norm_flag=batch_norm_flag)
        else:
            cond_np = all_x_splits[::-1][input_tier_condition_tag[0]][input_tier_condition_tag[1]]
            #input_stored_conditioning = None
            # conditioning input currently unnormalized
            if input_stored_conditioning is not None:
                cond_np = input_stored_conditioning
            cond = torch.tensor(cond_np).contiguous().to(hp.use_device)
            if use_half:
                cond = torch.tensor(cond_np).contiguous().to(hp.use_device).half()
            with torch.no_grad():
                pred_out = model(x_in, x_mask=x_mask_in,
                                 spatial_condition=cond,
                                 batch_norm_flag=batch_norm_flag)

        teacher_forced_pred = pred_out
        #teacher_forced_attn = model.attention_alignment
        sample_buffer[:, this_step, mel_step] = pred_out[:, this_step, mel_step].cpu().data.numpy()
        end_time = time.time()
        print("minibatch time {} secs".format(end_time - cur_time))
'''

sample_completed = time.time()

folder = input_output_dir + "sampled_forced_images/"
if not os.path.exists(folder):
    os.mkdir(folder)

if valid_el is not None:
    sample_info_path = folder + "text_info.json"
    with open(sample_info_path, 'w') as f:
        cleaned_valid_el = valid_el[0][3]
        json_string = json.dumps(cleaned_valid_el, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        f.write(json_string)

with open(folder + "bias_information.txt", "w") as f:
    full_n_frames = axis1_m * input_axis_split_list[0] # 352 for current settings
    time_downsample_ratio = full_n_frames / input_size_at_depth[0] # should always be integer value
    bias_in_seconds = bias_til * time_downsample_ratio * (1./speech.sample_rate) * speech.stft_step
    out_string = "Biased using groundtruth data until frame {}, (downsampling ratio {}, upscaled frame would be {}, approximately {} seconds)\nstart_frame:{}\n".format(bias_til, time_downsample_ratio, bias_til * time_downsample_ratio, bias_in_seconds, bias_til)
    f.write(out_string)

# function to find contiguous subsequences in a list
def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

for _i in range(hp.real_batch_size):
    unnormalized = sample_buffer.cpu().data.numpy() * saved_std + saved_mean
    plt.imshow(unnormalized[_i, :, :, 0])
    plt.savefig(folder + os.sep + "small_sampled_x{}.png".format(_i))
    plt.close()

    teacher_forced_pred = pred_out
    teacher_forced_attn = model.attention_alignment

    if input_tier_condition_tag is None:
        for _i in range(hp.real_batch_size):
            mel_cut = int(sample_mask[_i, :, 0, 0].cpu().data.numpy().sum())
            text_cut = int(torch_cond_seq_data_mask[:, _i].cpu().data.numpy().sum())
            # matshow vs imshow?
            this_att = teacher_forced_attn[:, _i, 0][:mel_cut, :text_cut]
            this_att = this_att.cpu().data.numpy().astype("float32")
            plt.imshow(this_att)
            plt.title("{}\n{}\n".format("/".join(saved_model_path.split("/")[:-1]), saved_model_path.split("/")[-1]))
            plt.savefig(folder + os.sep + "attn_{}.png".format(_i))
            plt.close()
    else:
        plt.imshow(cond_np[_i, :, :, 0])
        plt.savefig(folder + os.sep + "cond_small_x{}.png".format(_i))
        plt.close()

    # output length is first dim
    # conditioning dim is last dim
    if hasattr(model, "attention_extras"):
        attention_positions = np.array([model.attention_extras[_el]["kappa"].cpu().data.numpy() for _el in range(len(model.attention_extras))])[:, _i]
        attention_terminations = np.array([model.attention_extras[_el]["termination"].cpu().data.numpy() for _el in range(len(model.attention_extras))])[:, _i]

        aa = np.where(unnormalized[_i, :, :, 0].mean(axis=1) < 1E-3)[0]
        # get the start of the last contiguous subsequence with mean amplitude < 1E-3
        # should represent the end silence with a well trained model

        if len(aa) > 0:
            silent_subs = ranges(aa)
            last_sil_start = silent_subs[-1][0]
        else:
            last_sil_start = unnormalized.shape[1]
        with open(folder + "attention_termination_x{}.txt".format(_i), "w") as f:
            full_n_frames = axis1_m * input_axis_split_list[0] # 352 for current settings
            time_downsample_ratio = full_n_frames / input_size_at_depth[0] # should always be integer value
            sil_in_seconds = last_sil_start * time_downsample_ratio * (1./speech.sample_rate) * speech.stft_step
            out_string = "Sil frames begin at {}, (downsampling ratio {}, upscaled frame would be {}, approximately {} seconds)\nend_frame:{}\nend_scale:{}".format(last_sil_start, time_downsample_ratio, last_sil_start * time_downsample_ratio, sil_in_seconds, last_sil_start, time_downsample_ratio)
            f.write(out_string)

np.save(folder + "/" + "raw_samples.npy", sample_buffer.cpu().data.numpy())
np.save(folder + "/" + "unnormalized_samples.npy", sample_buffer.cpu().data.numpy() * saved_std + saved_mean)
np.save(folder + "/" + "minibatch_input.npy", original_buffer)
if input_tier_condition_tag is None:
    np.save(folder + "/" + "attn_activation.npy", teacher_forced_attn.cpu().data.numpy())
else:
    np.save(folder + "/" + "unnormalized_cond_input.npy", cond_np)
print("finished sampling in {} sec".format(sample_completed - begin_time))
sys.exit()
from IPython import embed; embed(); raise ValueError()

for _i in range(hp.real_batch_size):
    mel_cut = int(x_mask_in[_i, :, 0, 0].cpu().data.numpy().sum())
    text_cut = int(torch_cond_seq_data_mask[:, _i].cpu().data.numpy().sum())
    # matshow vs imshow?
    plt.imshow(teacher_forced_attn[..., _i].cpu().data.numpy()[:mel_cut, :text_cut])
    plt.title("{}\n{}\n".format("/".join(saved_model_path.split("/")[:-1]), saved_model_path.split("/")[-1]))
    plt.savefig("teacher_forced_attn_{}.png".format(_i))
    plt.close()


#outs = sample_dml(pred_out)
outs = pred_out * saved_std + saved_mean
for _i in range(hp.real_batch_size):
    reduced_mel_cut = int(x_mask_in[_i, :, 0, 0].cpu().data.numpy().sum())
    full_mel_cut = int(torch_data_mask[_i].cpu().data.numpy().sum())

    plt.imshow(outs[_i, :reduced_mel_cut, :, 0].cpu().data.numpy())
    plt.savefig("pred_x{}.png".format(_i))
    plt.close()

print("plotted")
from IPython import embed; embed(); raise ValueError()

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

# sample and average?
np_teacher_forced_pred = teacher_forced_pred.cpu().data.numpy()

np_teacher_forced_samples = 0. * np_teacher_forced_pred[..., 0][..., None]

global_temperature = float(args.temperature)
global_sample_itr = int(args.samples_to_average)
global_eps = 1E-3
global_top_p = float(args.p_cutoff)
global_bias_data_start = float(args.bias_data_start)

def _q(s):
    return str(s).replace(".", "pt")
file_basename = tag + "_temperature_{}".format(_q(global_temperature))
file_basename += "_samples_avg_{}".format(_q(global_sample_itr))
file_basename += "_top_p_{}".format(_q(global_top_p))
file_basename += "_bias_start_{}".format(_q(global_bias_data_start))

eps = global_eps
sample_itr = global_sample_itr
temperature = global_temperature
noise_random_state = np.random.RandomState(0)

def get_top_p_mask(logits, top_p):
    # 1 is keep 0 remove
    all_indices_to_remove = []
    shp = logits.shape
    r_logits = logits.copy().reshape((-1, logits.shape[-1]))
    for _i in range(len(r_logits)):
        # ::-1 to reverse thus achieving descending order
        sorted_indices = np.argsort(r_logits[_i])[::-1]
        sorted_logits = r_logits[_i][sorted_indices]
        cumulative_probs = np.cumsum(softmax_np(sorted_logits))

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].copy()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        r_logits[_i, :] = 1.
        r_logits[_i, indices_to_remove] = 0.
    mask = r_logits.reshape(*shp)
    return mask

p_mask = get_top_p_mask(np_teacher_forced_pred, global_top_p)
for i in range(sample_itr):
    noise = -np.log(-np.log(np.clip(noise_random_state.uniform(size=np_teacher_forced_pred.shape) + eps, eps, 1. - eps)))
    np_gumbel_noised_pred = np_teacher_forced_pred / temperature + noise
    np_gumbel_pred = np.argmax(p_mask * np_gumbel_noised_pred + (1. - p_mask) * 1E-10, axis=-1)[..., None]
    np_teacher_forced_samples += np_gumbel_pred
np_teacher_forced_samples /= float(sample_itr)
np_teacher_forced_samples = np.clip(np_teacher_forced_samples, 0, 255).astype("int32").astype("float32")

np_teacher_forced_argmax = np.argmax(teacher_forced_pred.cpu().data.numpy(), axis=-1)[..., None].astype("float32")
np_teacher_forced_plot = np_teacher_forced_samples

for i in range(data_batch.shape[0]):
    if input_tier_condition_tag is None:
        f, axarr = plt.subplots(1, 2)
        axarr[1].matshow(np_teacher_forced_plot[i], cmap="gray")
    else:
        f, axarr = plt.subplots(1, 3)
        axarr[1].matshow(teacher_forced_cond[i].cpu().data.numpy().astype("float32"), cmap="gray")
        axarr[2].matshow(np_teacher_forced_plot[i], cmap="gray")
    axarr[0].matshow(teacher_forced_in[i].cpu().data.numpy().astype("float32"), cmap="gray")

    for el_j in range(len(axarr)):
        axarr[el_j].axis("off")

    plt.savefig(folder_name + "/" + file_basename + "_teacher_forced_pred{}.png".format(i))
    plt.close()

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
blank_data_batch = 0. * np_teacher_forced_plot
for i in range(blank_data_batch.shape[1]):
    for j in range(blank_data_batch.shape[2]):
        if i < (int(blank_data_batch.shape[1] * global_bias_data_start)):
            blank_data_batch[:, i, j] = teacher_forced_in.cpu().data.numpy()[:, i, j]
            print("{} {} cell complete".format(i, j))
            continue

        if input_tier_condition_tag is None:
            pred_out = model(torch.tensor(blank_data_batch))
        else:
            cond = all_x_splits[::-1][input_tier_condition_tag[0]][input_tier_condition_tag[1]]
            pred_out = model(torch.tensor(blank_data_batch), spatial_condition=cond)

        np_pred = pred_out.cpu().data.numpy()
        np_samples = 0. * np_pred[..., 0][..., None]
        sample_itr = global_sample_itr
        eps = global_eps
        temperature = global_temperature
        noise_random_state = np.random.RandomState(0)
        p_mask = get_top_p_mask(np_pred, global_top_p)
        for _i in range(sample_itr):
            noise = -np.log(-np.log(np.clip(noise_random_state.uniform(size=np_pred.shape) + eps, eps, 1. - eps)))
            np_gumbel_noised_pred = np_pred / temperature + noise
            np_gumbel_pred = np.argmax(p_mask * np_gumbel_noised_pred + (1. - p_mask) * 1E-10, axis=-1)[..., None]
            np_samples += np_gumbel_pred
        np_samples /= float(sample_itr)
        np_samples = np.clip(np_samples, 0, 255).astype("int32").astype("float32")
        blank_data_batch[:, i, j] = np_samples[:, i, j]
        print("{} {} cell complete".format(i, j))

np_pure_samples = blank_data_batch
for i in range(blank_data_batch.shape[0]):
    if input_tier_condition_tag is None:
        f, axarr = plt.subplots(1, 2)
        axarr[0].matshow(np_pure_samples[i], cmap="gray")
    else:
        f, axarr = plt.subplots(1, 2)
        axarr[0].matshow(teacher_forced_cond[i].cpu().data.numpy().astype("float32"), cmap="gray")
        axarr[1].matshow(np_pure_samples[i], cmap="gray")
    for el_j in range(len(axarr)):
        axarr[el_j].axis("off")

    plt.savefig(folder_name + "/" + file_basename + "_biased_sampled{}.png".format(i))
    plt.close()

np.save(folder_name + "/" + "samples_" + file_basename + ".npy", np_pure_samples)
np.save(folder_name + "/" + "teacher_forced_samples_" + file_basename + ".npy", np_teacher_forced_samples)
np.save(folder_name + "/" + "teacher_forced_argmax_" + file_basename + ".npy", np_teacher_forced_argmax)
np.save(folder_name + "/" + "groundtruth_" + file_basename + ".npy", teacher_forced_in.cpu().data.numpy().astype("float32"))

print("step")
from IPython import embed; embed(); raise ValueError()

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
