from __future__ import print_function
import os
import sys

import numpy as np
import torch
from torch import nn
import torch.functional as F

from kkpthlib import AWDTransformerXLDecoderBlock
from kkpthlib import YAMTransformerBlock
from kkpthlib import HParams
from kkpthlib import Linear
from kkpthlib import EmbeddingDropout
from kkpthlib import WordCorpus
from kkpthlib import make_batches_from_list
from kkpthlib import StepIterator
from kkpthlib import CategoricalCrossEntropyFromLogits
from kkpthlib import clipping_grad_norm_
from kkpthlib import RampOpt
from kkpthlib import run_loop

hp = HParams(memory_len=20,
             context_len=70,
             embedding_dropout_keep_prob=.8,
             #transformer_input_dim=380,
             transformer_input_dim=240,
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             learning_rate=1E-5,
             min_learning_rate=1E-5,
             clip=.25,
             batch_size=3,
             n_layers=5,
             max_sequence_length=140,
             max_vocabulary_size=10000,
             random_seed=2122,
             data_storage_dir="penn")

def get_hparams():
    return hp

def build_model(hp):
    random_state = np.random.RandomState(hp.random_seed)
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.embedding = EmbeddingDropout(hp.max_vocabulary_size,
                                              hp.transformer_input_dim,
                                              dropout_keep_prob=hp.embedding_dropout_keep_prob,
                                              random_state=random_state,
                                              device=hp.use_device,
                                              name="embed")
            self.block = YAMTransformerBlock([hp.transformer_input_dim],
                                              10, # vert 
                                              9, # horiz
                                              hp.transformer_input_dim,
                                              hp.transformer_input_dim,
                                              3, # layers
                                              transformer_inner_layers=1,
                                              transformer_n_heads=10,
                                              transformer_head_dim=24,
                                              transformer_inner_dim=900,
                                              has_spatial_condition=False,
                                              spatial_condition_input_size=1,
                                              spatial_condition_n_layers=1,
                                              spatial_condition_n_heads=10,
                                              spatial_condition_head_dim=24,
                                              spatial_condition_inner_dim=900,
                                              device=hp.use_device,
                                              random_state=random_state,
                                              name="yam_block")

            """
            self.transformer = AWDTransformerXLDecoderBlock([hp.transformer_input_dim],
                                                            name="transformer_block",
                                                            random_state=random_state,
                                                            memory_len=hp.memory_len,
                                                            context_len=hp.context_len,
                                                            init="normal",
                                                            scale=0.02,
                                                            device=hp.use_device)
            """
            self.out_proj = Linear([hp.transformer_input_dim],
                                    hp.max_vocabulary_size,
                                    random_state=random_state,
                                    device=hp.use_device,
                                    init="normal",
                                    scale=0.02,
                                    name="model_out")

        def forward(self, x, cond_x, list_of_mems=None):
            xe, de = self.embedding(x)
            #out, l_o_m = self.block([xe], skip_input_embed=True)
            if cond_x is None:
                out = self.block([xe])
            else:
                out = self.block([xe], list_of_spatial_conditions=[cond_x])
            out_shp = out.shape
            out_reshp = out.reshape((-1, out_shp[-1]))
            p = self.out_proj([out_reshp])
            p = p.reshape((out_shp[0], out_shp[1], out_shp[2], -1))
            return p
    return Model().to(hp.use_device)

if __name__ == "__main__":
    train_data_file_path = hp.data_storage_dir + os.sep + "train.txt"
    valid_data_file_path = hp.data_storage_dir + os.sep + "valid.txt"
    test_data_file_path = hp.data_storage_dir + os.sep + "test.txt"
    corpus = WordCorpus(train_data_file_path=train_data_file_path,
                        valid_data_file_path=valid_data_file_path,
                        test_data_file_path=test_data_file_path,
                        cleaner_fn="lower_ascii_keep_standard_punctuation",
                        max_vocabulary_size=hp.max_vocabulary_size,
                        use_eos=False)
    train_batches = make_batches_from_list(corpus.train, batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len)
    valid_batches = make_batches_from_list(corpus.valid, batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len)
    test_batches = make_batches_from_list(corpus.test, batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len)

    train_random_state = np.random.RandomState(hp.random_seed)
    valid_random_state = np.random.RandomState(hp.random_seed + 1)
    test_random_state = np.random.RandomState(hp.random_seed + 2)

    train_itr = StepIterator([train_batches], circular_rotation=True, random_state=train_random_state)
    valid_itr = StepIterator([valid_batches], random_state=valid_random_state)
    test_itr = StepIterator([test_batches], random_state=test_random_state)

    model = build_model(hp)
    loss_fun = CategoricalCrossEntropyFromLogits()

    def get_std_ramp_opt(model):
        return RampOpt(hp.learning_rate, 1, 3000, 3000 + 125 * 450,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1E-9),
                       min_decay_learning_rate=hp.min_learning_rate)

    optimizer = get_std_ramp_opt(model)

    def loop(itr, extras, stateful_args):
        np_data = next(itr)
        np_data = np_data[:90]
        input_data = torch.tensor(np_data).to(hp.use_device)
        #input_data = input_data.reshape((9, 10, -1))
        input_data = input_data.reshape((10, 9, -1))
        input_data = input_data[..., None]

        # convert to batch, vert, horiz, 1
        # 3, 10, 9, 1
        # want to be contiguous on horiz direction
        input_data = input_data.permute((2, 0, 1, 3))

        cond_data = 0. * input_data + 0.5
        cond_data = None

        in_mems = stateful_args
        out = model(input_data, cond_data, list_of_mems=in_mems)
        # use input_data because the yamtransformer handles padding etc internally
        loss = loss_fun(out, input_data)
        loss = loss.sum(axis=2).sum(axis=1).mean()
        #loss = loss.mean()
        l = loss.cpu().data.numpy()
        optimizer.zero_grad()
        if extras["train"]:
            loss.backward()
            clipping_grad_norm_(model.parameters(), hp.clip)
            #clipping_grad_norm_(model.named_parameters(), hp.clip, named_check=True)
            optimizer.step()
        return l, None, None

    # the out-of-loop-check
    r = loop(train_itr, {"train": True}, None)
    r2 = loop(train_itr, {"train": True}, r[-1])
    from IPython import embed; embed(); raise ValueError()

    s = {"model": model,
         "optimizer": optimizer,
         "hparams": hp}

    run_loop(loop, train_itr,
             loop, valid_itr,
             s,
             n_train_steps_per=2000,
             n_valid_steps_per=250)
