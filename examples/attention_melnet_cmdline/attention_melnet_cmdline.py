from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F

from kkpthlib.datasets import fetch_mnist
from kkpthlib.datasets import EnglishSpeechCorpus

from kkpthlib import get_logger

from kkpthlib import Linear
from kkpthlib import Dropout
from kkpthlib import BernoulliCrossEntropyFromLogits
from kkpthlib import DiscretizedMixtureOfLogisticsCrossEntropyFromLogits
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
from kkpthlib import SequenceConv1dStack
from kkpthlib import BiLSTMLayer

from kkpthlib import AttentionMelNetTier
from kkpthlib import MelNetTier
from kkpthlib import MelNetFullContextLayer
from kkpthlib import CategoricalCrossEntropyFromLogits
from kkpthlib import relu
from kkpthlib import Embedding

from kkpthlib import space2batch
from kkpthlib import batch2space
from kkpthlib import split
from kkpthlib import interleave
from kkpthlib import scan
from kkpthlib import LSTMCell
from kkpthlib import BiLSTMLayer
from kkpthlib import GaussianAttentionCell

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script {}".format(__file__))
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

    args = parser.parse_args()
else:
    # filthy global hack passed from sampling code
    import builtins
    args = builtins.my_args
    # note here that the exact parser arguments will need to be *repeated* in the sampling code

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

logger = get_logger()

logger.info("sys.argv call {}".format(__file__))
logger.info("{}".format(" ".join(sys.argv)))

logger.info("\ndirect argparse args to script {}".format(__file__))
for arg in vars(args):
    logger.info("{}={}".format(arg, getattr(args, arg)))

mnist = fetch_mnist()

hp = HParams(input_dim=1,
             hidden_dim=input_hidden_size,
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             learning_rate=1E-4,
             clip=3.5,
             n_layers_per_tier=[input_n_layers],
             melnet_init="truncated_normal",
             #melnet_init=None,
             # mnist so we force input symbols
             input_symbols=256,
             n_mix=10,
             output_size=2 * 10 + 10,
             text_input_symbols=194, #len(speech.phone_lookup),
             input_image_size=input_size_at_depth,
             batch_size=input_batch_size,
             n_epochs=1000,
             random_seed=2122)

data_random_state = np.random.RandomState(hp.random_seed)
folder_base = "/usr/local/data/kkastner/robovoice/robovoice_d_25k"
speech = EnglishSpeechCorpus(metadata_csv=folder_base + "/metadata.csv",
                             wav_folder=folder_base + "/wavs/",
                             alignment_folder=folder_base + "/alignment_json/",
                             fixed_minibatch_time_secs=6,
                             train_split=0.9,
                             random_state=data_random_state)



"""
train_el = speech.get_train_utterances(10)
valid_el = speech.get_train_utterances(10)
speech.format_minibatch(train_el)
print("hrr")
from IPython import embed; embed(); raise ValueError()
"""

def get_hparams():
    return hp

def build_model(hp):
    random_state = np.random.RandomState(hp.random_seed)
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            if input_tier_condition_tag is None:
                # handle text attention separately
                self.embed_text = Embedding(hp.text_input_symbols, hp.hidden_dim, random_state=random_state,
                                            name="tier_{}_{}_sz_{}_{}_embed_text".format(input_tier_input_tag[0], input_tier_input_tag[1], hp.input_image_size[0], hp.input_image_size[1]), device=hp.use_device)
                """
                self.conv_text = SequenceConv1dStack([hp.hidden_dim], hp.hidden_dim, n_stacks=3, random_state=random_state,
                                                     name="tier_{}_{}_sz_{}_{}_conv_text".format(input_tier_input_tag[0], input_tier_input_tag[1], hp.input_image_size[0], hp.input_image_size[1]), device=hp.use_device)
                """
                # divided by 2 so the output is hp.hidden_dim
                self.bilstm_text = BiLSTMLayer([hp.hidden_dim], hp.hidden_dim // 2, random_state=random_state,
                                               init=hp.melnet_init,
                                               name="tier_{}_{}_sz_{}_{}_bilstm_text".format(input_tier_input_tag[0], input_tier_input_tag[1], hp.input_image_size[0], hp.input_image_size[1]),
                                               device=hp.use_device)

                self.mn_t = AttentionMelNetTier([hp.input_symbols], hp.input_image_size[0], hp.input_image_size[1],
                                                hp.hidden_dim, hp.output_size, hp.n_layers_per_tier[0],
                                                has_centralized_stack=True,
                                                has_attention=True,
                                                random_state=random_state,
                                                init=hp.melnet_init,
                                                device=hp.use_device,
                                                name="tier_{}_{}_sz_{}_{}_mn".format(input_tier_input_tag[0], input_tier_input_tag[1],
                                                                                     hp.input_image_size[0], hp.input_image_size[1]))
            else:
                self.mn_t = AttentionMelNetTier([hp.input_symbols], hp.input_image_size[0], hp.input_image_size[1],
                                                hp.hidden_dim, hp.output_size, hp.n_layers_per_tier[0],
                                                has_spatial_condition=True,
                                                random_state=random_state,
                                                init=hp.melnet_init,
                                                device=hp.use_device,
                                        name="tier_{}_{}_cond_{}_{}_sz_{}_{}_mn".format(input_tier_input_tag[0], input_tier_input_tag[1],
                                                                                        input_tier_condition_tag[0], input_tier_condition_tag[1],
                                                                                        hp.input_image_size[0], hp.input_image_size[1]))


        def forward(self, x, x_mask=None,
                    spatial_condition=None,
                    memory_condition=None, memory_condition_mask=None, batch_norm_flag=0.):
            # for now we don't use the x_mask in the model itself, only in the loss calculations
            if spatial_condition is None:
                assert memory_condition is not None
                mem, mem_e = self.embed_text(memory_condition)

                #mem_conv = self.conv_text([mem], batch_norm_flag)
                # mask based on the conditioning mask

                #mem_conv = mem_conv * memory_condition_mask[..., None]

                # use mask in BiLSTM
                mem_lstm = self.bilstm_text([mem], input_mask=memory_condition_mask)
                # x currently batch, time, freq, 1
                # mem time, batch, feat
                # feed mask for attention calculations as well
                mn_out, alignment, termination, ksis = self.mn_t([x], memory=mem_lstm, memory_mask=memory_condition_mask)
                self.attention_alignment = alignment
                self.attention_termination = termination
                self.attention_ksis = ksis
            else:
                mn_out = self.mn_t([x], list_of_spatial_conditions=[spatial_condition])
            return mn_out
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
    # N H W C
    loss_function = DiscretizedMixtureOfLogisticsCrossEntropyFromLogits()

    def loop(itr, extras, stateful_args):
        if extras["train"]:
            model.train()
        else:
            model.eval()
        model.zero_grad()

        if extras["train"]:
            train_el = speech.get_train_utterances(hp.batch_size)
            cond_seq_data_batch, cond_seq_mask, data_batch, data_mask = speech.format_minibatch(train_el)
            batch_norm_flag = 0.
        else:
            valid_el = speech.get_valid_utterances(hp.batch_size)
            cond_seq_data_batch, cond_seq_mask, data_batch, data_mask = speech.format_minibatch(valid_el)
            batch_norm_flag = 1.
        torch_data_batch = torch.tensor(data_batch[..., None]).contiguous().to(hp.use_device)
        torch_data_mask = torch.tensor(data_mask).contiguous().to(hp.use_device)
        torch_cond_seq_data_batch = torch.tensor(cond_seq_data_batch[..., None]).contiguous().to(hp.use_device)
        torch_cond_seq_data_mask = torch.tensor(cond_seq_mask).contiguous().to(hp.use_device)
        """
        data_batch, = next(itr)
        # N H W C
        data_batch = data_batch.reshape(data_batch.shape[0], 28, 28, 1)
        # N C H W
        data_batch = data_batch.transpose(0, 3, 1, 2).astype("int32").astype("float32")
        torch_data_batch = torch.tensor(data_batch).contiguous().to(hp.use_device)
        # N H W 1
        torch_data_batch = torch_data_batch[:, 0][..., None]
        """


        all_x_splits = []
        x_t = torch_data_batch
        for aa in input_axis_split_list:
            all_x_splits.append(split(x_t, axis=aa))
            x_t = all_x_splits[-1][0]
        x_in = all_x_splits[::-1][input_tier_input_tag[0]][input_tier_input_tag[1]]

        all_x_mask_splits = []
        # broadcast mask over frequency so we can downsample
        x_mask_t = torch_data_mask[..., None, None] + 0. * torch_data_batch
        for aa in input_axis_split_list:
            all_x_mask_splits.append(split(x_mask_t, axis=aa))
            x_mask_t = all_x_mask_splits[-1][0]
        x_mask_in = all_x_mask_splits[::-1][input_tier_input_tag[0]][input_tier_input_tag[1]]

        if input_tier_condition_tag is None:
            pred_out = model(x_in, x_mask=x_mask_in,
                             memory_condition=torch_cond_seq_data_batch,
                             memory_condition_mask=torch_cond_seq_data_mask,
                             batch_norm_flag=batch_norm_flag)
        else:
            cond = all_x_splits[::-1][input_tier_condition_tag[0]][input_tier_condition_tag[1]]
            pred_out = model(x_in, x_mask=x_mask_in,
                             spatial_condition=cond,
                             batch_norm_flag=batch_norm_flag)

        # x_in comes in discretized between 0 and 256, now scale -1 to 1
        loss1 = loss_function(pred_out, 2 * (x_in / 256.) - 1., n_mix=hp.n_mix)
        # take last trailing 1 off x_mask_in for shapes to match
        loss = ((loss1 * x_mask_in[..., 0]) / (x_mask_in.sum())).sum()
        l = loss.cpu().data.numpy()

        optimizer.zero_grad()
        if extras["train"]:
            loss.backward()
            clipping_grad_value_(model.parameters(), hp.clip)
            #clipping_grad_value_(model.named_parameters(), hp.clip, named_check=True)
            optimizer.step()
        return [l,], None, None

    s = {"model": model,
         "optimizer": optimizer,
         "hparams": hp}

    # the out-of-loop-check
    r = loop(train_itr, {"train": True}, None)
    r2 = loop(train_itr, {"train": True}, None)

    if input_tier_condition_tag is None:
        tag = str(args.experiment_name) + "_tier_{}_{}_sz_{}_{}".format(input_tier_input_tag[0], input_tier_input_tag[1],
                                                                        input_size_at_depth[0], input_size_at_depth[1])
    else:
        tag = str(args.experiment_name) + "_tier_{}_{}_cond_{}_{}_sz_{}_{}".format(input_tier_input_tag[0], input_tier_input_tag[1],
                                                                                   input_tier_condition_tag[0], input_tier_condition_tag[1],
                                                                                   input_size_at_depth[0], input_size_at_depth[1])

    run_loop(loop, train_itr,
             loop, valid_itr,
             s,
             force_tag=tag,
             n_train_steps_per=1000,
             n_valid_steps_per=250)