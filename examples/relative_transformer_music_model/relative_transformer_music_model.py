from __future__ import print_function
import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F

from kkpthlib.datasets import fetch_jsb_chorales
from kkpthlib.datasets import MusicJSONRasterIterator
from kkpthlib import Linear
from kkpthlib import Embedding
from kkpthlib import CategoricalCrossEntropy
from kkpthlib import LayerNorm
from kkpthlib import relu
from kkpthlib import softmax
from kkpthlib import clipping_grad_norm_
from kkpthlib import clipping_grad_value_
from kkpthlib import ListIterator
from kkpthlib import run_loop
from kkpthlib import HParams
from kkpthlib import get_logger
from kkpthlib import TransformerPositionalEncoding
from kkpthlib import RelativeTransformerAutoregressiveBlock
from kkpthlib import NoamOpt
from kkpthlib import RampOpt

jsb = fetch_jsb_chorales()

# just training on everything seemed to work well, but lets try to form a "correct" validation set
#len([f for f in jsb["files"] if "transposed" not in f])
#from IPython import embed; embed(); raise ValueError()

logger = get_logger()

hp = HParams(#balance token dim and clock dim
             token_embed_dim=64,
             clock_embed_dim=12,
             # maybe change this?
             hidden_dim=512,
             separate_onsets=True,
             init="truncated_normal",
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             dropout_keep_prob=1.,
             learning_rate=0.000125,
             clip=.25,
             batch_size=10,
             clocks=[2, 4, 8, 16, 32, 64],
             n_layers=5,
             max_sequence_length=1024,
             random_seed=2122,
             vocab_storage="jsb_raster_music_transformer_stored_vocab.npz")

def get_hparams():
    return hp

if not os.path.exists(hp.vocab_storage):
    logger.info("Vocab {} not found, calculating...".format(hp.vocab_storage))
    itr = MusicJSONRasterIterator(jsb["files"],
                                  batch_size=hp.batch_size,
                                  max_sequence_length=hp.max_sequence_length,
                                  with_clocks=None,
                                  random_seed=hp.random_seed,
                                  separate_onsets=hp.separate_onsets,
                                  iterate_once=True)
    # set iterate_once=True and find vocabulary size
    # 
    vocab = set()
    try:
        while True:
            roll, mask = next(itr)
            unique_vals = np.unique(roll)
            vocab = vocab | set(unique_vals.astype(int))
    except Exception as e:
        # will raise an error when it terminates
        # debug here
        # from IPython import embed; embed(); raise ValueError()
        pass
    vocab = np.array(sorted(list(vocab)))
    np.savez(hp.vocab_storage, vocab=vocab)
else:
    logger.info("Saved vocab {} found, loading...".format(hp.vocab_storage))
    d = np.load(hp.vocab_storage)
    vocab = d["vocab"]

def build_model(hp):
    random_state = np.random.RandomState(hp.random_seed)
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.clock_embeddings = nn.ModuleList()
            for i in range(len(hp.clocks)):
                e = Embedding(hp.clocks[i], hp.clock_embed_dim,
                              random_state=random_state,
                              name="clock_embed_{}".format(hp.clocks[i]),
                              device=hp.use_device)
                self.clock_embeddings.append(e)
            self.token_embedding = Embedding(len(vocab), hp.token_embed_dim,
                                             random_state=random_state,
                                             name="token_embed",
                                             device=hp.use_device)
            self.positional_encoding = TransformerPositionalEncoding(hp.token_embed_dim, device=hp.use_device, name="position_embed")
            combined_dim = len(hp.clocks) * hp.clock_embed_dim + hp.token_embed_dim

            self.input_dropout = torch.nn.Dropout(p=1. - hp.dropout_keep_prob)
            self.comb_proj = Linear([combined_dim], hp.hidden_dim, init=hp.init,
                                    random_state=random_state, device=hp.use_device, name="hidden_proj")

            self.block = RelativeTransformerAutoregressiveBlock([hp.hidden_dim],
                                                                n_layers=hp.n_layers,
                                                                dropout_keep_prob=hp.dropout_keep_prob,
                                                                random_state=random_state,
                                                                init=hp.init,
                                                                name="block",
                                                                device=hp.use_device)

            self.out_proj = Linear([hp.hidden_dim], len(vocab), init=hp.init, random_state=random_state, device=hp.use_device, name="model_out")

        def forward(self, x, x_mask, clocks):
            te, te_v = self.token_embedding(x)
            pe, pe_v = self.positional_encoding(te)
            ce = []
            for i in range(len(self.clock_embeddings)):
                t_ce, t_ce_v = self.clock_embeddings[i](clocks[i])
                ce.append(t_ce)
            comb = torch.cat([pe] + ce, dim=-1)
            # 0 out the embedding for pad tokens
            comb = comb * x_mask[..., None]
            comb = self.input_dropout(comb)
            # dropout!
            h_proj = self.comb_proj([comb])
            hidden = self.block([h_proj], x_mask)
            o = self.out_proj([hidden])
            return o
    model = Model().to(hp.use_device)
    return model

# hardcoded 1k moves everything "out of the way" for easier vocab mapping
vocab_mapper = {k: v + 1000 for k, v in zip(vocab, range(len(vocab)))}
inverse_vocab_mapper = {v - 1000:k for k, v in vocab_mapper.items()}

if __name__ == "__main__":
    train_itr = MusicJSONRasterIterator(jsb["files"][:-50],
                                        batch_size=hp.batch_size,
                                        max_sequence_length=hp.max_sequence_length,
                                        with_clocks=hp.clocks,
                                        separate_onsets=hp.separate_onsets,
                                        random_seed=hp.random_seed)

    valid_itr = MusicJSONRasterIterator(jsb["files"][-50:],
                                        batch_size=hp.batch_size,
                                        max_sequence_length=hp.max_sequence_length,
                                        with_clocks=hp.clocks,
                                        separate_onsets=hp.separate_onsets,
                                        random_seed=hp.random_seed)

    model = build_model(hp)
    loss_fun = CategoricalCrossEntropy()
    def get_std_noam_opt(model):
        return NoamOpt(hp.hidden_dim, 1, 10000,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1E-9))

    def get_std_ramp_opt(model):
        return RampOpt(hp.learning_rate, 1, 10000, 10000 * 200,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1E-9))

    #optimizer = get_std_noam_opt(model)
    optimizer = get_std_ramp_opt(model)

    def loop(itr, extras, stateful_args):
        piano_roll, mask, clocks = next(itr)

        # bump up by 1000 so we can map them all without conflict
        for k in vocab_mapper.keys():
            piano_roll[piano_roll == k] = vocab_mapper[k]
        # move it down to proper range
        piano_roll = piano_roll - 1000.

        piano_roll = torch.Tensor(piano_roll).to(hp.use_device)
        mask = torch.Tensor(mask).to(hp.use_device)
        clocks = [torch.Tensor(c).to(hp.use_device) for c in clocks]

        div = 4
        # do half overlapping windows for training
        cut = len(piano_roll) // div
        step = len(piano_roll) // int((div / 2))
        total_l = 0
        offset = 0
        for i in range(div):
            start_cut = i * cut
            stop_cut = i * cut + step
            piano_roll_cut = piano_roll[start_cut:stop_cut]
            mask_cut = mask[start_cut:stop_cut]
            clocks_cut = [c[start_cut:stop_cut].to(hp.use_device) for c in clocks]

            # trim off one for AR prediction?
            clocks_cut = [c for c in clocks_cut]
            linear_out = model(piano_roll_cut, mask_cut, clocks_cut)

            #prob_out = softmax(linear_out)
            # only use :-1 preds
            loss = loss_fun(linear_out[:-1], piano_roll_cut[1:])
            from IPython import embed; embed(); raise ValueError()
            #loss = (mask_cut[:-1] * loss).mean()
            loss = ((mask_cut[:-1] * loss) / mask_cut[:-1].sum()).sum()

            #loss = (mask_cut[:-1] * loss).sum(dim=0).mean()
            l = loss.cpu().data.numpy()
            total_l = total_l + l
            optimizer.zero_grad()
            if extras["train"]:
                loss.backward()
                #clipping_grad_norm_(model.parameters(), hp.clip)
                #clipping_grad_norm_(model.named_parameters(), hp.clip, named_check=True)
                #clipping_grad_value_(model.parameters(), hp.clip)
                #clipping_grad_value_(model.named_parameters(), hp.clip, named_check=True)
                optimizer.step()
        return l, None

    s = {"model": model,
         "optimizer": optimizer,
         "hparams": hp}

    # for debugging
    loop(train_itr, {"train": True}, [])
    raise ValueError("finished loop")

    run_loop(loop, train_itr,
             loop, valid_itr,
             s,
             n_train_steps_per=1000,
             n_valid_steps_per=100)
