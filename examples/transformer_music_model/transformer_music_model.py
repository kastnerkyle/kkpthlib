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
from kkpthlib import ListIterator
from kkpthlib import run_loop
from kkpthlib import HParams
from kkpthlib import get_logger

from kkpthlib import BasicTransformerBlock

jsb = fetch_jsb_chorales()

logger = get_logger()

hp = HParams(input_dim=133, #hardcoded from knowledge of vocab size, can temporarily set to 0 if needed
             #balance token dim and clock dim
             token_embed_dim=96,
             clock_embed_dim=16,
             # maybe change this?
             hidden_dim=512,
             use_device="cuda",
             learning_rate=0.0001,
             clip=10.,
             batch_size=24,
             clocks=[2, 4, 8, 16, 32, 64],
             n_layers=5,
             max_sequence_length=1000,
             random_seed=2122,
             vocab_storage="jsb_raster_music_transformer_stored_vocab.npz")

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
    except:
        # will raise an Error when it terminates
        pass
    vocab = np.array(sorted(list(vocab)))
    np.savez(hp.vocab_storage, vocab=vocab)
else:
    logger.info("Saved vocab {} found, loading...".format(hp.vocab_storage))
    d = np.load(hp.vocab_storage)
    vocab = d["vocab"]

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
        self.blocks = [BasicTransformerBlock([combined_dim],
                                           hp.max_sequence_length,
                                           random_state=random_state,
                                           name="block_{}".format(i),
                                           device=hp.use_device) for i in range(hp.n_layers)]

        self.ln = LayerNorm(combined_dim, name="model_layer_norm",
                            device=hp.use_device)

        self.out_proj = Linear([combined_dim], len(vocab), random_state=random_state, device=hp.use_device, name="model_out")


    def forward(self, x, x_mask, clocks, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.blocks)
        else:
            past_length = past[0][0].size(-2)

        te, te_v = self.token_embedding(x)
        ce = []
        for i in range(len(self.clock_embeddings)):
            t_ce, t_ce_v = self.clock_embeddings[i](clocks[i])
            ce.append(t_ce)
        comb = torch.cat([te] + ce, dim=-1)
        # 0 out the embedding for pad tokens
        comb = comb * x_mask[..., None]
        hidden_states = comb

        presents = []
        assert len(self.blocks) == len(past)
        for block, layer_past in zip(self.blocks, past):
            hidden_states, present = block([hidden_states], source_mask=x_mask, target_mask=x_mask)
            presents.append(present)
        hidden_states = self.ln(hidden_states)
        o = self.out_proj([hidden_states])
        return o, presents


train_itr = MusicJSONRasterIterator(jsb["files"][:-10],
                                    batch_size=hp.batch_size,
                                    max_sequence_length=hp.max_sequence_length,
                                    with_clocks=hp.clocks,
                                    random_seed=hp.random_seed)

valid_itr = MusicJSONRasterIterator(jsb["files"][10:],
                                    batch_size=hp.batch_size,
                                    max_sequence_length=hp.max_sequence_length,
                                    with_clocks=hp.clocks,
                                    random_seed=hp.random_seed)

vocab_mapper = {k: v + 1000 for k, v in zip(vocab, range(len(vocab)))}
inverse_vocab_mapper = {v - 1000:k for k, v in vocab_mapper.items()}

m = Model().to(hp.use_device)
loss_fun = CategoricalCrossEntropy()
optimizer = torch.optim.Adam(m.parameters(), hp.learning_rate)

def loop(itr, extras, stateful_args):
    piano_roll, mask, clocks = next(itr)
    # bump up by 1000 so we can map them all without conflict
    for k in vocab_mapper.keys():
        piano_roll[piano_roll == k] = vocab_mapper[k]
    # move it down to proper range
    piano_roll = piano_roll - 1000.

    piano_roll = torch.Tensor(piano_roll).to(hp.use_device)
    mask = torch.Tensor(mask).to(hp.use_device)
    # trim off one for AR prediction
    clocks = [torch.Tensor(c)[:-1].to(hp.use_device) for c in clocks]

    linear_out, past = m(piano_roll[:-1], mask[:-1], clocks)
    prob_out = softmax(linear_out)
    loss = loss_fun(prob_out, piano_roll[1:])
    loss = (mask[:-1] * loss).sum(dim=0).mean()
    l = loss.cpu().data.numpy()
    optimizer.zero_grad()
    if extras["train"]:
        loss.backward()
        clipping_grad_norm_(m.parameters(), hp.clip)
        optimizer.step()
    return l, None

s = {"model": m,
     "optimizer": optimizer,
     "hparams": hp}

run_loop(loop, train_itr,
         loop, valid_itr,
         s,
         n_train_steps_per=10000,
         n_valid_steps_per=500)
