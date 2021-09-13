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
# have to redo training argparse args as well for load setup
parser.add_argument('--axis_splits', type=str, default=None,
                    help='string denoting the axis splits for the model, eg 2121 starting from first split to last\n')
parser.add_argument('--stored_sampled_tier_data', type=str, default=None,
                    help='all previously sampled tier data, in order from beginning tier to previous from left to right. Last array assumed to be the conditioning input. comma separated string, should be path to unnormalized data')

args = parser.parse_args()
"""
python combine_all_outputs.py --axis_splits=2121 --stored_sampled_tier_data=tier0_0/sampled_forced_images/unnormalized_samples.npy,tier0_1_cond0_0/sampled_forced_images/unnormalized_samples.npy,tier1_1_cond1_0/sampled_forced_images/unnormalized_samples.npy,tier2_1_cond2_0/sampled_forced_images/unnormalized_samples.npy,tier3_1_cond3_0/sampled_forced_images/unnormalized_samples.npy,tier4_1_cond4_0/sampled_forced_images/unnormalized_samples.npy
"""
was_none = False
if args.axis_splits is None:
    was_none = True
    args.axis_splits="2121"

if was_none:
    print("axis_splits not passed, using default arguments")
    print("default axis_splits={}".format(args.axis_splits))

was_none = False
if args.stored_sampled_tier_data is None:
    was_none = True
    args.stored_sampled_tier_data = "tier0_0/sampled_forced_images/unnormalized_samples.npy,tier0_1_cond0_0/sampled_forced_images/unnormalized_samples.npy,tier1_1_cond1_0/sampled_forced_images/unnormalized_samples.npy,tier2_1_cond2_0/sampled_forced_images/unnormalized_samples.npy,tier3_1_cond3_0/sampled_forced_images/unnormalized_samples.npy,tier4_1_cond4_0/sampled_forced_images/unnormalized_samples.npy"

if was_none:
    print("stored_sampled_tier_data not passed, using default arguments")
    print("default stored_sampled_tier_data={}".format(args.stored_sampled_tier_data))

input_axis_split_list = [int(args.axis_splits[i]) for i in range(len(args.axis_splits))]

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
                input_real_batch_size = next_conditioning.shape[0]
                comb_dims = (next_conditioning.shape[0],
                             built_conditioning.shape[1] + next_conditioning.shape[1],
                             built_conditioning.shape[2] + next_conditioning.shape[2],
                             next_conditioning.shape[-1])
                # reverse split list since it is done in depth order (shallow to deep)
                # n - 1 because we include first one
                if _n < len(stored_sampled_tier_data_paths) - 1:
                    this_split = input_axis_split_list[::-1][_n - 1]
                else:
                    # if last split was 2, now 1?
                    if this_split == 2:
                        this_split = 1
                    # if last split was 1, now 2?
                    elif this_split == 1:
                        this_split = 2
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
else:
    raise ValueError("Need to pass stored input data to combine into final npy")
fpath = "combined_unnormalized_samples.npy"
print("Saving combined files to {}".format(fpath))
np.save("combined_unnormalized_samples.npy", input_stored_conditioning)
