from __future__ import print_function
import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F
import copy
import sys
import shutil
import time
import itertools

from typing import Generic, TypeVar, Dict, List, Optional
from csp import CSP, Constraint

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
parser.add_argument('--temperature', '-t', type=float, default=1.0,
                    help='sampling temperature to use (default 1.0)')
parser.add_argument('--p_cutoff', '-p', type=float, default=.5,
                    help='cutoff to use in top p sampling (default .5)')

args = parser.parse_args()
if len(args.saved_model_path) < 1:
    if args.direct_saved_model_path is None:
        parser.print_help()
        sys.exit(1)

temperature = args.temperature

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


from kkpthlib import fetch_jsb_chorales
from kkpthlib import MusicJSONInfillCorpus
from kkpthlib import convert_voice_lists_to_music_json
from kkpthlib import write_music_json
from kkpthlib import music_json_to_midi

from importlib import import_module
model_import_path, fname = saved_model_definition_path.rsplit(os.sep, 1)
sys.path.insert(0, model_import_path)
p, _ = fname.split(".", 1)
mod = import_module(p)
get_hparams = getattr(mod, "get_hparams")
build_model = getattr(mod, "build_model")
sys.path.pop(0)

hp = get_hparams()

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
model_dict = torch.load(saved_model_path, map_location=hp.use_device)
model.load_state_dict(model_dict)
model.eval()

from kkpthlib import softmax_np
from kkpthlib import top_k_from_logits_np
from kkpthlib import top_p_from_logits_np

midi_sample_dir = "midi_samples"

if os.path.exists(midi_sample_dir):
    print("Previous midi_samples folder found, removing...")
    shutil.rmtree(midi_sample_dir)

# this should ALWAYS happen
if not os.path.exists(midi_sample_dir):
    os.mkdir(midi_sample_dir)

tstamp = int(time.time())
itr = train_itr

batch_np_o, batch_masks_np_o, batch_offsets_np_o, batch_indices_np_o = next(itr)

# sanity check one at a time
sampling_random_state = np.random.RandomState(args.random_seed)
for t in range(batch_np_o.shape[1]):
    batch_np = copy.deepcopy(batch_np_o)
    batch_masks_np = copy.deepcopy(batch_masks_np_o)
    batch_offsets_np = copy.deepcopy(batch_offsets_np_o)

    return_answers, return_offsets, return_positions = infill_corpus.get_answer_groups_from_example(batch_np, batch_offsets_np)

    # replace -1 with 4 for "special channel"
    for i in range(batch_offsets_np.shape[1]):
        replace_i = np.where(batch_offsets_np[:, i, 0] == -1)[0]
        batch_offsets_np[replace_i, i, 0] = 4

    # one hot batch mask?
    oh_voice_batch_mask_np = [(0. * batch_offsets_np[:, :, 0])[:, :, None] for _ in range(hp.n_voice_channels)]
    oh_voice_batch_mask_np = np.concatenate(oh_voice_batch_mask_np, axis=-1).astype("float32")
    for i in range(batch_offsets_np.shape[1]):
        for _v in range(hp.n_voice_channels):
            match_i = np.where(batch_offsets_np[:, i, 0] == _v)[0]
            oh_voice_batch_mask_np[match_i, i, _v] = 1.

    try:
        assert np.all(oh_voice_batch_mask_np.sum(axis=-1).ravel() == 1.)
    except:
        print("error in oh mask checks")
        from IPython import embed; embed(); raise ValueError()

    oh_voice_batch_mask = torch.tensor(oh_voice_batch_mask_np).to(hp.use_device)
    # target because we never padded with 0s, but only want to work with target 
    oh_target_voice_batch_mask = oh_voice_batch_mask

    batch = torch.tensor(batch_np)
    pad_batch = torch.cat((0 * batch[:1] + infill_corpus.dictionary.word2idx[infill_corpus.fill_symbol], batch), 0).to(hp.use_device)

    offsets = torch.tensor(batch_offsets_np)
    pad_offsets = torch.cat((-1 + 0 * offsets[:1], offsets)).to(hp.use_device)

    batch_masks = torch.tensor(batch_masks_np).to(hp.use_device)

    input_batch = pad_batch[:-1]
    target_batch = pad_batch[1:].long()
    input_batch_offsets = pad_offsets[:-1]
    target_batch_offsets = pad_offsets[1:]

    out, out_mems = model(input_batch, batch_masks, input_batch_offsets, oh_target_voice_batch_mask, list_of_mems=None)

    context_token = infill_corpus.dictionary.word2idx[infill_corpus.end_context_symbol]
    answer_token = infill_corpus.dictionary.word2idx[infill_corpus.answer_symbol]
    mask_token = infill_corpus.dictionary.word2idx[infill_corpus.mask_symbol]

    # use return answers, offsets, etc to do the sampling
    this_return_answers = return_answers[t]
    this_return_offsets = return_offsets[t]
    this_return_positions = return_positions[t]

    all_answers = []
    all_answers_durations = []
    all_answers_voices = []
    context_boundary = int(np.where(input_batch[:, t].cpu().data.numpy() == context_token)[0])
    # +1 to not cut off context mark
    for list_pos, el in enumerate(this_return_offsets):
        total_duration_constraint_per_voice = [0, 0, 0, 0]
        for _vt in range(4):
            voice_match = np.where(this_return_offsets[list_pos][:, 0] == _vt)[0]
            if len(voice_match) == 0:
                this_voice_dur = 0
            else:
                this_voice_dur = this_return_offsets[list_pos][voice_match][:, -1].sum()
            total_duration_constraint_per_voice[_vt] = this_voice_dur

        this_answer = []
        this_answer_durations = []
        this_answer_voices = []
        while_step = 0
        while True:
            this_voice = None

            for _vt in range(4):
                if total_duration_constraint_per_voice[_vt] == 0:
                    continue
                ss = [td for _ii, td in enumerate(this_answer_durations) if this_answer_voices[_ii] == _vt]
                #if total_duration_constraint_per_voice[_vt] - sum(this_answer_durations) == 0:
                if total_duration_constraint_per_voice[_vt] - sum(ss) == 0:
                    continue

                this_voice = _vt
                break

            #if len(this_return_offsets[list_pos][:, 0]) > 1:
            #    if len(this_answer) >= 4:
            #        print("multi")
            #        from IPython import embed; embed(); raise ValueError()
            #        pass

            if this_voice is None:
                # if we get here with no voice assigned, everything has been satisfied
                break

            this_duration_constraint = total_duration_constraint_per_voice[this_voice]
            ss = [td for _ii, td in enumerate(this_answer_durations) if this_answer_voices[_ii] == this_voice]
            this_duration_constraint = this_duration_constraint - sum(ss)

            if this_duration_constraint == 0:
                continue

            # check how many voice answers have been chosen already?
            # + 1 here to include the context token (value 2)
            this_input_batch = input_batch[:context_boundary + 1]
            this_batch_masks = batch_masks[:context_boundary + 1]
            this_input_batch_offsets = input_batch_offsets[:context_boundary + 1]
            # no + 1 because of offset ?
            this_oh_target_voice_batch_mask = oh_target_voice_batch_mask[:context_boundary]

            if len(all_answers) > 0:
                this_return_offsets = return_offsets[t]
                for _n, a in enumerate(all_answers):
                    # add data to the input batch
                    # add mask 
                    # add batch offset
                    # add voice info
                    a_np = np.array(a)[:, None, None]
                    # broadcast the data into a chunk of len l, batch_size, 1
                    a_pt = 0. * this_input_batch[:1] + this_input_batch.new(*a_np.shape).copy_(torch.from_numpy(a_np))
                    a_pt_mask = 0 * this_batch_masks[:len(a_np)]
                    # create all 0s voice batch mask
                    a_pt_voice_batch_mask = 0. * this_oh_target_voice_batch_mask[:len(a_np)]
                    # calculate offset
                    a_pt_batch_offsets = 0. * this_input_batch_offsets[:len(a_np)]

                    # get start times of the gt answer for this mask
                    starts_at_time_offset = this_return_offsets[_n][0, 1]

                    # add this check back?
                    # we check validity at the end regardless
                    #duration_offset = this_return_offsets[_n][0, 2]

                    current_time_into_offset = 0.
                    for _ni, ai in enumerate(a):
                        tup = infill_corpus.dictionary.idx2word[ai]
                        _dur = tup[1]
                        _step = starts_at_time_offset + current_time_into_offset
                        _voice = all_answers_voices[_n][_ni]
                        a_pt_batch_offsets[_ni, :, 0] = _voice
                        a_pt_batch_offsets[_ni, :, 1] = _step
                        a_pt_batch_offsets[_ni, :, 2] = _dur

                        a_pt_voice_batch_mask[_ni, :, _voice] = 1.
                        current_time_into_offset += _dur
                        #try:
                        #    assert duration_offset >= current_time_into_offset
                        #except:
                        #    print("??")
                        #    from IPython import embed; embed(); raise ValueError()

                    # now add all of these answer values onto the minibatch
                    this_input_batch = torch.cat((this_input_batch, a_pt), axis=0)
                    this_batch_masks = torch.cat((this_batch_masks, a_pt_mask), axis=0)
                    this_input_batch_offsets = torch.cat((this_input_batch_offsets, a_pt_batch_offsets), axis=0)
                    this_oh_target_voice_batch_mask = torch.cat((this_oh_target_voice_batch_mask, a_pt_voice_batch_mask), axis=0)

                    # first element of batch offsets is all -1? NEED to fix?

                    # after adding all this stuff, must add "space" value for answers
                    # data value for "answer" is 1
                    # offset is [4., 0., 0.]
                    # voice channel is 4
                    # 4 is also the DATA mask value...
                    ans_pt = 0 * this_input_batch[:1] + 1
                    ans_pt_mask = 0 * this_batch_masks[:1]
                    ans_pt_batch_offsets = 0 * this_input_batch_offsets[:1]
                    ans_pt_batch_offsets[:, 0] = 4.
                    ans_pt_voice_batch_mask = 0 * this_oh_target_voice_batch_mask[:1]
                    ans_pt_voice_batch_mask[:, 4] = 1.

                    this_input_batch = torch.cat((this_input_batch, ans_pt), axis=0)
                    this_batch_masks = torch.cat((this_batch_masks, ans_pt_mask), axis=0)
                    this_input_batch_offsets = torch.cat((this_input_batch_offsets, ans_pt_batch_offsets), axis=0)
                    this_oh_target_voice_batch_mask = torch.cat((this_oh_target_voice_batch_mask, ans_pt_voice_batch_mask), axis=0)

            if len(this_answer) > 0:
                print("this")
                this_return_offsets = return_offsets[t]
                for _n, a in enumerate(this_answer):
                    # add data to the input batch
                    # add mask 
                    # add batch offset
                    # add voice info
                    a_np = np.array(a)[None, None, None]
                    # broadcast the data into a chunk of len l, batch_size, 1
                    a_pt = 0. * this_input_batch[:1] + this_input_batch.new(*a_np.shape).copy_(torch.from_numpy(a_np))
                    a_pt_mask = 0 * this_batch_masks[:len(a_np)]
                    # create all 0s voice batch mask
                    a_pt_voice_batch_mask = 0. * this_oh_target_voice_batch_mask[:len(a_np)]
                    # calculate offset
                    a_pt_batch_offsets = 0. * this_input_batch_offsets[:len(a_np)]

                    # needs to be len(all_answers) + step into this_answer
                    this_offset_idx = len(all_answers)

                    # get start times of the gt answer for this mask
                    starts_at_time_offset = this_return_offsets[this_offset_idx][0, 1]
                    duration_offset = this_return_offsets[this_offset_idx][0, 2]

                    current_time_into_offset = 0.
                    tup = infill_corpus.dictionary.idx2word[a]
                    _dur = tup[1]
                    _step = starts_at_time_offset + current_time_into_offset
                    _voice = this_answer_voices[_n]
                    a_pt_batch_offsets[0, :, 0] = _voice
                    a_pt_batch_offsets[0, :, 1] = _step
                    a_pt_batch_offsets[0, :, 2] = _dur

                    a_pt_voice_batch_mask[0, :, _voice] = 1.
                    current_time_into_offset += _dur
                    #try:
                    #    assert duration_offset >= current_time_into_offset
                    #except:
                    #    from IPython import embed; embed(); raise ValueError()

                    # now add all of these answer values onto the minibatch
                    this_input_batch = torch.cat((this_input_batch, a_pt), axis=0)
                    this_batch_masks = torch.cat((this_batch_masks, a_pt_mask), axis=0)
                    this_input_batch_offsets = torch.cat((this_input_batch_offsets, a_pt_batch_offsets), axis=0)
                    this_oh_target_voice_batch_mask = torch.cat((this_oh_target_voice_batch_mask, a_pt_voice_batch_mask), axis=0)

                    # first element of batch offsets is all -1? NEED to fix?

                    # after adding all this stuff, must add "space" value for answers
                    # data value for "answer" is 1
                    # offset is [4., 0., 0.]
                    # voice channel is 4
                    # 4 is also the DATA mask value...
                    #ans_pt = 0 * this_input_batch[:1] + 1
                    #ans_pt_mask = 0 * this_batch_masks[:1]
                    #ans_pt_batch_offsets = 0 * this_input_batch_offsets[:1]
                    #ans_pt_batch_offsets[:, 0] = 4.
                    #ans_pt_voice_batch_mask = 0 * this_oh_target_voice_batch_mask[:1]
                    #ans_pt_voice_batch_mask[:, 4] = 1.

                    #this_input_batch = torch.cat((this_input_batch, ans_pt), axis=0)
                    #this_batch_masks = torch.cat((this_batch_masks, ans_pt_mask), axis=0)
                    #this_input_batch_offsets = torch.cat((this_input_batch_offsets, ans_pt_batch_offsets), axis=0)
                    #this_oh_target_voice_batch_mask = torch.cat((this_oh_target_voice_batch_mask, ans_pt_voice_batch_mask), axis=0)
                #from IPython import embed; embed(); raise ValueError()

            # need to be sure the oh_target_voice_batch_mask voice input matches _vt
            # add on the current voice as a one hot value...
            blank = 0. * this_oh_target_voice_batch_mask[:1]
            blank[:, :, _vt] = 1.
            this_oh_target_voice_batch_mask = torch.cat((this_oh_target_voice_batch_mask, blank), axis=0)

            #out, out_mems = model(input_batch[:current_boundary], batch_masks[:current_boundary], input_batch_offsets[:current_boundary], oh_target_voice_batch_mask[:current_boundary], list_of_mems=None)
            out, out_mems = model(this_input_batch, this_batch_masks, this_input_batch_offsets, this_oh_target_voice_batch_mask, list_of_mems=None)

            all_opts = [infill_corpus.dictionary.idx2word[aa] for aa in range(out.shape[-1])]
            all_valid_opts = [(n, el) for n, el in enumerate(all_opts) if el[1] > 0 and el[1] <= this_duration_constraint]
            all_invalid_opts = [(n, el) for n, el in enumerate(all_opts) if el[1] <= 0 or el[1] > this_duration_constraint]
            out_np = out.cpu().data.numpy()
            out_np = out_np / temperature

            """
            # try to remove everything outside the constraint
            for aio in all_invalid_opts:
                out_np[:, :, aio[0]] = -1E18

            # take top p candidates
            out_np_p = top_p_from_logits_np(out_np)
            probs = softmax_np(out_np_p[:, t])
            # -1 should be correct because the batch length is relative to the current entry we are predicting
            nonzero_idx = np.where(probs[-1] > 0.)[0]
            poss = [infill_corpus.dictionary.idx2word[aa] for aa in nonzero_idx]
            step_sub_probs = probs[-1][nonzero_idx]
            # renormalize
            step_sub_probs = step_sub_probs / np.sum(step_sub_probs)
            s_idx = sampling_random_state.choice(list(range(len(step_sub_probs))), p=step_sub_probs)
            true_idx = nonzero_idx[s_idx]
            sampled = poss[s_idx]
            sampled_pitch = sampled[0]
            sampled_dur = sampled[1]
            """

            sample_steps = 0
            while True:
                # take top p candidates
                if sample_steps < 20:
                    out_np_p = top_p_from_logits_np(out_np, .9)
                else:
                    out_np_p = copy.deepcopy(out_np)
                probs = softmax_np(out_np_p[:, t])
                # -1 should be correct because the batch length is relative to the current entry we are predicting
                nonzero_idx = np.where(probs[-1] > 0.)[0]
                poss = [infill_corpus.dictionary.idx2word[aa] for aa in nonzero_idx]
                step_sub_probs = probs[-1][nonzero_idx]
                # renormalize
                step_sub_probs = step_sub_probs / np.sum(step_sub_probs)
                s_idx = sampling_random_state.choice(list(range(len(step_sub_probs))), p=step_sub_probs)
                true_idx = nonzero_idx[s_idx]
                sampled = poss[s_idx]
                sampled_pitch = sampled[0]
                sampled_dur = sampled[1]
                print("sample_steps {}".format(sample_steps))
                sample_steps += 1

                if sampled_dur > 0 and sampled_dur <= this_duration_constraint:
                    # check if the pitch is in the set of known notes
                    if sampled_pitch in set([infill_corpus.dictionary.idx2word[el][0] for el in batch_np[:, t, 0]]):
                        break

            sampled_as_index = infill_corpus.dictionary.word2idx[sampled]
            this_answer.append(sampled_as_index)
            this_answer_durations.append(sampled_dur)
            this_answer_voices.append(this_voice)
            while_step += 1
        all_answers.append(this_answer)
        all_answers_durations.append(this_answer_durations)
        all_answers_voices.append(this_answer_voices)
        print(this_answer)

    # check that the answer matches the original constraints
    # be sure we are filling the same number of slots as the original
    assert len(all_answers) == len(this_return_offsets)
    for _ii in range(len(all_answers)):
        aa = all_answers[_ii]
        aav = all_answers_voices[_ii]
        aad = all_answers_durations[_ii]

        aagt = this_return_offsets[_ii]
        assert len(aav) == len(aad)

        # check that the set of voices covered is the same
        vgt = sorted(list(set(aagt[:, 0])))
        setaav = sorted(list(set(aav)))
        assert len(vgt) == len(setaav)
        assert all([i == j for i,j in zip(vgt,setaav)])

        # for every voice, check the covered duration is the same as gt
        for aavi in setaav:
            ss = [aadi for _cc, aadi in enumerate(aad) if aav[_cc] == aavi]
            ssgt = aagt[np.where(aagt[:, 0] == aavi)[0]]
            assert np.sum(ssgt[:, -1]) == np.sum(ss)
    print("fin")
    # reform the original sequence
    # use batch_np because that is the original data, before padding and so on
    # no +1 on context boundary because this is the *unshifted* version
    for sample_type in range(2):
        base_sequence = batch_np[:context_boundary, t, 0]
        base_sequence_offsets = batch_offsets_np[:context_boundary, t]
        print("end")

        pitches_list = []
        durations_list = []
        voices_list = []
        idx = 0
        ans_idx = 0

        #
        if sample_type == 1:
            # replace with gt
            all_answers_durations = [[trai[1] for trai in tra] for tra in this_return_answers]
            all_answers = [[infill_corpus.dictionary.word2idx[trai] for trai in tra] for tra in this_return_answers]
            all_answers_voices = [[int(troi[0]) for troi in tro] for tro in this_return_offsets]

        while True:
            if idx >= len(base_sequence):
                break

            p = base_sequence[idx]
            if p != 0:
                print("copy")
                # for non-zero, just copy it over
                tup = infill_corpus.dictionary.idx2word[p]
                pitches_list.append(tup[0])
                durations_list.append(tup[1])
                o = base_sequence_offsets[idx]
                voices_list.append(int(o[0]))
                idx += 1
            else:
                print("ans")
                # hit a zero, add a while answer chunk
                ans_sub_idx = 0
                while True:
                    # add a whole answer chunk in
                    if ans_sub_idx >= len(all_answers[ans_idx]):
                        break
                    ta = all_answers[ans_idx][ans_sub_idx]
                    tad = all_answers_durations[ans_idx][ans_sub_idx]
                    tav = all_answers_voices[ans_idx][ans_sub_idx]

                    tup = infill_corpus.dictionary.idx2word[ta]
                    pitches_list.append(tup[0])
                    durations_list.append(tup[1])
                    voices_list.append(tav)
                    ans_sub_idx += 1
                ans_idx += 1
                # once we hit a 0, skip all following zeros til next non zero
                while True:
                    print("el")
                    if idx >= len(base_sequence):
                        break

                    if base_sequence[idx] == 0:
                        # skip past 0s til we hit the next
                        idx += 1
                    else:
                        break
        print("hello?")
        if pitches_list[-1] == -3:
            pitches_list = pitches_list[:-1]
            durations_list = durations_list[:-1]
            voices_list = voices_list[:-1]

        assert len(pitches_list) == len(durations_list)
        assert len(pitches_list) == len(voices_list)
        # set voices list to -1 where we have measure marks
        voices_list = [v if p != 99 else 4 for p, v in zip(pitches_list, voices_list)]
        data = convert_voice_lists_to_music_json(pitch_lists=pitches_list, duration_lists=durations_list, voices_list=voices_list)

        if sample_type == 0:
            json_fpath = midi_sample_dir + os.sep + "sampled{}_{}.json".format(t, tstamp)
            fpath = midi_sample_dir + os.sep + "sampled{}_{}.midi".format(t, tstamp)
        elif sample_type == 1:
            json_fpath = midi_sample_dir + os.sep + "gt{}_{}.json".format(t, tstamp)
            fpath = midi_sample_dir + os.sep + "gt{}_{}.midi".format(t, tstamp)
        else:
            raise ValueError("")

        write_music_json(data, json_fpath)

        #a = "harpsichord_preset"
        #b = "woodwind_preset"
        #m = {0: [(a, 0), (b, marked_quarters_context_boundary[0])],
        #     1: [(a, 0), (b, marked_quarters_context_boundary[1])],
        #     2: [(a, 0), (b, marked_quarters_context_boundary[2])],
        #     3: [(a, 0), (b, marked_quarters_context_boundary[3])]}

        #a = "harpsichord_preset"
        a = "woodwind_preset"
        m = {0: [(a, 0),],
             1: [(a, 0),],
             2: [(a, 0),],
             3: [(a, 0),]}
        music_json_to_midi(data, fpath, voice_program_map=m)
        print("Sampled {}".format(fpath))
