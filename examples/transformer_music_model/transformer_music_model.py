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
from kkpthlib import TransformerAutoregressiveBlock
from kkpthlib import NoamOpt

jsb = fetch_jsb_chorales()

# just training on everything seemed to work well, but lets try to form a "correct" validation set
#len([f for f in jsb["files"] if "transposed" not in f])
#from IPython import embed; embed(); raise ValueError()

logger = get_logger()

hp = HParams(input_dim=133, #hardcoded from knowledge of vocab size, can temporarily set to 0 if needed
             #balance token dim and clock dim
             token_embed_dim=96,
             clock_embed_dim=16,
             # maybe change this?
             hidden_dim=512,
             init="truncated_normal",
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             dropout_keep_prob=1.,
             learning_rate=0.0001,
             clip=3.,
             batch_size=20,
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
            combined_dim = len(hp.clocks) * hp.clock_embed_dim + hp.token_embed_dim
            self.comb_proj = Linear([combined_dim], hp.hidden_dim, init=hp.init,
                                    random_state=random_state, device=hp.use_device, name="hidden_proj")

            self.block = TransformerAutoregressiveBlock([hp.hidden_dim],
                                                        n_layers=hp.n_layers,
                                                        dropout_keep_prob=hp.dropout_keep_prob,
                                                        random_state=random_state,
                                                        init=hp.init,
                                                        name="block",
                                                        device=hp.use_device)

            self.out_proj = Linear([hp.hidden_dim], len(vocab), init=hp.init, random_state=random_state, device=hp.use_device, name="model_out")

        def forward(self, x, x_mask, clocks, list_of_memories=None, memory_mask=None):
            te, te_v = self.token_embedding(x)
            ce = []
            for i in range(len(self.clock_embeddings)):
                t_ce, t_ce_v = self.clock_embeddings[i](clocks[i])
                ce.append(t_ce)
            comb = torch.cat([te] + ce, dim=-1)
            # 0 out the embedding for pad tokens
            comb = comb * x_mask[..., None]
            h_proj = self.comb_proj([comb])
            hidden, mems = self.block([h_proj], x_mask, list_of_memories, memory_mask)
            o = self.out_proj([hidden])
            return o, mems
    model = Model().to(hp.use_device)
    return model

vocab_mapper = {k: v + 1000 for k, v in zip(vocab, range(len(vocab)))}
inverse_vocab_mapper = {v - 1000:k for k, v in vocab_mapper.items()}

if __name__ == "__main__":
    train_itr = MusicJSONRasterIterator(jsb["files"][:-50],
                                        batch_size=hp.batch_size,
                                        max_sequence_length=hp.max_sequence_length,
                                        with_clocks=hp.clocks,
                                        random_seed=hp.random_seed)

    valid_itr = MusicJSONRasterIterator(jsb["files"][-50:],
                                        batch_size=hp.batch_size,
                                        max_sequence_length=hp.max_sequence_length,
                                        with_clocks=hp.clocks,
                                        random_seed=hp.random_seed)

    model = build_model(hp)
    loss_fun = CategoricalCrossEntropy()
    def get_std_noam_opt(model):
        return NoamOpt(hp.hidden_dim, 1, 4000,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1E-9))
    optimizer = get_std_noam_opt(model)

    def loop(itr, extras, stateful_args):
        piano_roll, mask, clocks = next(itr)

        # bump up by 1000 so we can map them all without conflict
        for k in vocab_mapper.keys():
            piano_roll[piano_roll == k] = vocab_mapper[k]
        # move it down to proper range
        piano_roll = piano_roll - 1000.

        piano_roll = torch.Tensor(piano_roll).to(hp.use_device)
        mask = torch.Tensor(mask).to(hp.use_device)

        list_of_memories = None
        memory_mask = None
        div = 4
        cut = len(piano_roll) // div
        total_l = 0
        for i in range(div):
            piano_roll_cut = piano_roll[i * cut:(i + 1) * cut]
            mask_cut = mask[i * cut:(i + 1) * cut]

            clocks_cut = [torch.Tensor(c)[i * cut:(i + 1) * cut].to(hp.use_device) for c in clocks]
            # trim off one for AR prediction
            clocks_cut = [c[:-1] for c in clocks_cut]

            linear_out, past = model(piano_roll_cut[:-1], mask_cut[:-1], clocks_cut, list_of_memories, memory_mask)
            list_of_memories = past
            memory_mask = mask_cut[:-1]

            prob_out = softmax(linear_out)
            loss = loss_fun(prob_out, piano_roll_cut[1:])
            loss = (mask_cut[:-1] * loss).mean()
            #loss = (mask_cut[:-1] * loss).sum(dim=0).mean()
            l = loss.cpu().data.numpy()
            total_l = total_l + l
            optimizer.zero_grad()
            if extras["train"]:
                loss.backward()
                #clipping_grad_norm_(model.parameters(), hp.clip)
                clipping_grad_value_(model.parameters(), hp.clip)
                optimizer.step()
        return l, None

    s = {"model": model,
         "optimizer": optimizer,
         "hparams": hp}

    run_loop(loop, train_itr,
             loop, valid_itr,
             s,
             n_train_steps_per=1000,
             n_valid_steps_per=50)
