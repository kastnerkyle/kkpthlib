from __future__ import print_function
import os
import sys

import numpy as np
import torch
from torch import nn
import torch.functional as F

from kkpthlib import AWDTransformerXLDecoderBlock
from kkpthlib import HParams
from kkpthlib import Linear
from kkpthlib import relu
from kkpthlib import EmbeddingDropout
from kkpthlib import WordCorpus
from kkpthlib import make_batches_from_list
from kkpthlib import StepIterator
from kkpthlib import CategoricalCrossEntropyFromLogits
from kkpthlib import clipping_grad_norm_
from kkpthlib import clipping_grad_value_
from kkpthlib import RampOpt
from kkpthlib import run_loop

from kkpthlib import fetch_jsb_chorales
from kkpthlib import MusicJSONInfillCorpus
from kkpthlib import convert_voice_lists_to_music_json
from kkpthlib import music_json_to_midi
from kkpthlib import write_music_json

hp = HParams(memory_len=0,
             context_len=0,
             sequence_len=64,
             transformer_input_dim=380,
             inner_dim=900,
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             learning_rate=1E-4,
             min_learning_rate=1E-6,
             clip=.25,
             batch_size=10,
             n_layers=16,
             embedding_dropout_keep_prob=0.5,
             attention_dropout_keep_prob=0.8,
             input_dropout_keep_prob=0.3,
             inner_dropout_keep_prob=0.8,
             hidden_dropout_keep_prob=1.0,
             output_dropout_keep_prob=0.5,
             random_seed=2122)


def get_hparams():
    return hp

def build_model(hp, file_corpus):
    random_state = np.random.RandomState(hp.random_seed)
    class Model(nn.Module):
        def __init__(self, hp, this_file_corpus):
            super(Model, self).__init__()

            vocab_size = len(this_file_corpus.dictionary.idx2word)
            inp = hp.transformer_input_dim
            self.emb = EmbeddingDropout(vocab_size,
                                        inp,
                                        dropout_keep_prob=hp.embedding_dropout_keep_prob,
                                        random_state=random_state,
                                        device=hp.use_device,
                                        name="embed")
            self.in_proj = Linear([inp],
                                  hp.transformer_input_dim,
                                  random_state=random_state,
                                  device=hp.use_device,
                                  init="normal",
                                  scale=0.02,
                                  name="model_in")

            self.transformer = AWDTransformerXLDecoderBlock([hp.transformer_input_dim],
                                                            name="transformer_block",
                                                            random_state=random_state,
                                                            memory_len=hp.memory_len,
                                                            context_len=hp.context_len,
                                                            n_layers=hp.n_layers,
                                                            inner_dim=hp.inner_dim,
                                                            input_dropout_keep_prob=hp.input_dropout_keep_prob,
                                                            attention_dropout_keep_prob=hp.attention_dropout_keep_prob,
                                                            inner_dropout_keep_prob=hp.inner_dropout_keep_prob,
                                                            hidden_dropout_keep_prob=hp.hidden_dropout_keep_prob,
                                                            output_dropout_keep_prob=hp.output_dropout_keep_prob,
                                                            init="normal",
                                                            scale=0.02,
                                                            device=hp.use_device)

            self.out_proj = Linear([hp.transformer_input_dim],
                                   len(this_file_corpus.dictionary.idx2word),
                                   random_state=random_state,
                                   device=hp.use_device,
                                   init="normal",
                                   scale=0.02,
                                   name="model_out")

        def forward(self, input_batch, input_batch_masks, list_of_mems=None):
            e, d_e = self.emb(input_batch)
            x1 = relu(self.in_proj([e]))
            # right now assume all batch masks are the same...
            out, l_o_m = self.transformer(x1, input_batch_masks, list_of_mems=list_of_mems)
            p = self.out_proj([out])
            return p, l_o_m
    return Model(hp, file_corpus).to(hp.use_device)

if __name__ == "__main__":
    jsb = fetch_jsb_chorales()

    # sort into minor / major, then by key
    all_transposed = sorted([f for f in jsb["files"] if "original" not in f], key=lambda x:
                            (x.split(os.sep)[-1].split(".")[-2].split("transposed")[0].split("-")[0],
                             x.split(os.sep)[-1].split(".")[-2].split("transposed")[0].split("-")[1]))

    bwv_names = sorted(list(set([f.split(os.sep)[-1].split(".")[0] for f in all_transposed])))
    vrng = np.random.RandomState(144)
    vrng.shuffle(bwv_names)

    # 15 is ~5% of the data if you hold out all 12 transpositions
    # holding out whole songs so actually a pretty hard validation set...
    valid_names = bwv_names[:15]
    train_files = [f for f in all_transposed if all([vn not in f for vn in valid_names])]
    valid_files = [f for f in all_transposed if any([vn in f for vn in valid_names])]
    assert all([vf not in train_files for vf in valid_files])

    # shuffle the train and valid files before we make the flat_measure_corpus
    vrng.shuffle(train_files)
    vrng.shuffle(valid_files)

    infill_corpus = MusicJSONInfillCorpus(train_data_file_paths=train_files,
                                          valid_data_file_paths=valid_files,
                                          raster_scan=True)
    train_itr = infill_corpus.get_iterator(hp.batch_size, hp.random_seed, sequence_len=hp.sequence_len, context_len=hp.context_len)
    valid_itr = infill_corpus.get_iterator(hp.batch_size, hp.random_seed + 1, sequence_len=hp.sequence_len, context_len=hp.context_len,
                                           _type="valid")

    model = build_model(hp, infill_corpus)
    model.train()
    loss_fun = CategoricalCrossEntropyFromLogits()

    def get_std_ramp_opt(model):
        return RampOpt(hp.learning_rate, 1, 3000, 3000 + 125 * 450,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1E-9),
                       min_decay_learning_rate=hp.min_learning_rate)

    optimizer = get_std_ramp_opt(model)

    first_valid = True
    first_train = True

    def loop(itr, extras, stateful_args):
        global first_valid
        global first_train
        batch_np, batch_masks_np, batch_offsets_np, batch_indices_np = next(itr)

        batch = torch.tensor(batch_np)
        pad_batch = torch.cat((0 * batch[:1] + infill_corpus.dictionary.word2idx[infill_corpus.fill_symbol], batch), 0).to(hp.use_device)
        batch_masks = torch.tensor(batch_masks_np).to(hp.use_device)

        input_batch = pad_batch[:-1]
        target_batch = pad_batch[1:].long()

        return_answers, return_offsets, return_positions = infill_corpus.get_answer_groups_from_example(batch_np, batch_offsets_np)
        print("start")
        for i in range(1000000):
            #if i % 1000 == 0:
            print(i)
            batch_np, batch_masks_np, batch_offsets_np, batch_indices_np = next(itr)

            batch = torch.tensor(batch_np)
            pad_batch = torch.cat((0 * batch[:1] + infill_corpus.dictionary.word2idx[infill_corpus.fill_symbol], batch), 0).to(hp.use_device)
            batch_masks = torch.tensor(batch_masks_np).to(hp.use_device)

            input_batch = pad_batch[:-1]
            target_batch = pad_batch[1:].long()

            return_answers, return_offsets, return_positions = infill_corpus.get_answer_groups_from_example(batch_np, batch_offsets_np)
        print("fin")
        from IPython import embed; embed(); raise ValueError()

        in_mems = stateful_args
        if extras["train"]:
            if first_train:
                model.train()
                out, out_mems = model(input_batch, batch_masks, list_of_mems=None)
                first_train = False
                first_valid = True
            else:
                model.train()
                out, out_mems = model(input_batch, batch_masks, list_of_mems=in_mems)
        else:
            if first_valid:
                model.eval()
                out, out_mems = model(input_batch, batch_masks, list_of_mems=None)
                first_valid = False
                first_train = True
            else:
                model.eval()
                out, out_mems = model(input_batch, batch_masks, list_of_mems=in_mems)

        # inputs have already been cut to context length inside transformer
        loss = loss_fun(out, target_batch)
        # masks use transformer convention 0 if valid, 1 if invalid
        loss = loss * (1. - batch_masks)
        loss = (loss / (1. - batch_masks).sum()).sum()

        l = loss.cpu().data.numpy()
        optimizer.zero_grad()
        if extras["train"]:
            loss.backward()
            clipping_grad_norm_(model.parameters(), hp.clip)
            optimizer.step()
        else:
            pass
            # don't carry over valid memories into train, just copy the train ones over
            # this means our validation score in training will be a bit off, but the memory
            # during training should be OK
            #out_mems = in_mems
        return l, None, out_mems

    # the out-of-loop-check
    rs = []
    for i in range(100000):
        print(i)
        if i == 0:
            r = loop(train_itr, {"train": True}, None)
        else:
            r = loop(train_itr, {"train": True}, rs[-1][-1])
        rs.append(r)
    from IPython import embed; embed(); raise ValueError()

    s = {"model": model,
         "optimizer": optimizer,
         "hparams": hp}

    run_loop(loop, train_itr,
             loop, valid_itr,
             s,
             n_train_steps_per=2000,
             n_valid_steps_per=250)
