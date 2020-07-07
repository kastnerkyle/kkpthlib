from __future__ import print_function
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
parser.add_argument('--context_division', type=int, default=2,
                    help='Amount to divide the default context from training by - helps improve sampling performance, as even though in training you can see "up to" a certain amount, on average you see shorter fill contexts')
parser.add_argument('--generative_rounds', type=int, default=0,
                    help="Number of gibbs sampling passes to do for generating, use 0 for single left-to-right generation")
args = parser.parse_args()
if len(args.saved_model_path) < 1:
    if args.direct_saved_model_path is None:
        parser.print_help()
        sys.exit(1)

if args.direct_saved_model_path is not None:
    saved_model_path = args.direct_saved_model_path
else:
    saved_model_path = args.saved_model_path[0]

if args.generative_rounds is not None:
    generative_rounds = int(args.generative_rounds)
else:
    generative_rounds = None

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

from kkpthlib import softmax_np
from kkpthlib import top_k_from_logits_np
from kkpthlib import top_p_from_logits_np
from kkpthlib import StepIterator
from kkpthlib import make_batches_from_list

from kkpthlib import fetch_jsb_chorales
from kkpthlib import piano_roll_from_music_json_file
from kkpthlib import MusicJSONFlatMeasureCorpus

from kkpthlib import convert_voice_lists_to_music_json
from kkpthlib import music_json_to_midi
from kkpthlib import write_music_json

jsb = fetch_jsb_chorales()

# sort into minor / major, then by key
all_transposed = sorted([f for f in jsb["files"] if "original" not in f], key=lambda x:
                        (x.split(os.sep)[-1].split(".")[-2].split("transposed")[0].split("-")[0],
                         x.split(os.sep)[-1].split(".")[-2].split("transposed")[0].split("-")[1]))

bwv_names = sorted(list(set([f.split(os.sep)[-1].split(".")[0] for f in all_transposed])))
vrng = np.random.RandomState(144)
vrng.shuffle(bwv_names)
# 15 is ~5% of the data
# holding out whole songs so actually a pretty hard validation set...
valid_names = bwv_names[:15]
train_files = [f for f in all_transposed if all([vn not in f for vn in valid_names])]
valid_files = [f for f in all_transposed if any([vn in f for vn in valid_names])]
assert all([vf not in train_files for vf in valid_files])

# shuffle the train and valid files before we make the flat_measure_corpus
vrng.shuffle(train_files)
vrng.shuffle(valid_files)
flat_measure_corpus = MusicJSONFlatMeasureCorpus(train_data_file_paths=train_files,
                                                 valid_data_file_paths=valid_files)

# length of that list is the number of feature groups!
train_batches = None
valid_batches = None
train_cut_points = [p == flat_measure_corpus.pitch_dictionary.word2idx[99] for p in flat_measure_corpus.train[0]]
valid_cut_points = [p == flat_measure_corpus.pitch_dictionary.word2idx[99] for p in flat_measure_corpus.valid[0]]
for v in range(len(flat_measure_corpus.train)):
    # num_batch_steps, time_length_of_batch, batch_size
    this_train_batches = make_batches_from_list(flat_measure_corpus.train[v], batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len, cut_points=train_cut_points, fill_value=len(flat_measure_corpus.pitch_dictionary.idx2word))
    this_valid_batches = make_batches_from_list(flat_measure_corpus.valid[v], batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len, cut_points=valid_cut_points, fill_value=len(flat_measure_corpus.pitch_dictionary.idx2word))
    if train_batches is None:
        train_batches = this_train_batches[..., None]
        valid_batches = this_valid_batches[..., None]
    else:
        # num_batch_steps, time_length_of_batch, batch_size, features (4 - pitch, duration, velocity, voice)
        train_batches = np.concatenate((train_batches, this_train_batches[..., None]), axis=-1)
        valid_batches = np.concatenate((valid_batches, this_valid_batches[..., None]), axis=-1)

train_random_state = np.random.RandomState(hp.random_seed)
valid_random_state = np.random.RandomState(hp.random_seed + 1)
gen_random_state = np.random.RandomState(hp.random_seed + 2)

train_itr = StepIterator([train_batches], circular_rotation=True, random_state=train_random_state)
valid_itr = StepIterator([valid_batches], random_state=valid_random_state)

sampling_random_state = np.random.RandomState(2123)

np_data = next(valid_itr)
quality_cut = hp.context_len + (len(np_data) - hp.context_len) // args.context_division
np_data = np_data[:quality_cut]

true_data = np_data.copy()

midi_sample_dir = "midi_samples"
if not os.path.exists(midi_sample_dir):
    os.mkdir(midi_sample_dir)

for i in range(true_data.shape[1]):
    non_pad = [c < len(flat_measure_corpus.pitch_dictionary.idx2word) for c in true_data[:, i, 0]]
    pitches = [flat_measure_corpus.pitch_dictionary.idx2word[c] for n, c in enumerate(true_data[:, i, 0]) if non_pad[n]]
    durations = [flat_measure_corpus.duration_dictionary.idx2word[c] for n, c in enumerate(true_data[:, i, 1]) if non_pad[n]]
    voices = [vv for n, vv in enumerate(true_data[:, i, -1]) if non_pad[n]]

    data = convert_voice_lists_to_music_json(pitch_lists=pitches, duration_lists=durations, voices_list=voices)

    json_fpath = midi_sample_dir + os.sep + "true{}.json".format(i)
    write_music_json(data, json_fpath)

    fpath = midi_sample_dir + os.sep + "true{}.midi".format(i)
    music_json_to_midi(data, fpath)
    print("Wrote out {}".format(fpath))

gen_random_state = np.random.RandomState(hp.random_seed + 3)

for gibbs_iter in range(generative_rounds + 1):
    if gibbs_iter > 0:
        print("gibbs iter {}".format(gibbs_iter))

    np_pitch_data = np_data[..., 0]
    # at sampling time, use contexts with roughly HALF the training maximum
    np_perm_masks, np_target_mappings, np_target_masks, _, _, _, np_perm_orders = model.transformer.make_inputs_targets_masks_and_mappings(
        np_pitch_data, K=hp.mask_K, max_n_gram=hp.max_n_gram // args.context_division,
        context_cut=hp.context_len, random_state=gen_random_state, sequential_order=True)

    np_targets = np_pitch_data[:-1]
    # will have to split this up
    np_input_ks = np_data[:-1]
    np_input_qs = np_target_masks

    out_mems = None

    num_blanks = np.sum(np_target_masks, axis=0)
    num_filled = 0. * num_blanks
    orig_np_target_masks = np.copy(np_target_masks)

    in_mems = None
    #in_mems = out_mems

    if hp.use_target_mappings:
        old = np.copy(np_targets)
        old_subs = [old[np.where(np_target_masks[:, i])[0], i][None] for i in range(hp.batch_size)]
        np_targets = np.concatenate(old_subs).transpose(1, 0)
        # all 1s mask
        np_target_masks = 0. * np_targets + 1.

        pad_l = np.zeros((hp.context_len, hp.batch_size))
        np_targets = np.concatenate((pad_l, np_targets))
        np_target_masks = np.concatenate((pad_l, np_target_masks))
        # assume len(np_input_ks) == orig len(targets)
        pad_r = np.zeros((len(np_input_ks) - len(np_targets), hp.batch_size))
        np_targets = np.concatenate((np_targets, pad_r))
        np_target_masks = np.concatenate((np_target_masks, pad_r))
    else:
        np_target_mappings = None

    input_ks = torch.tensor(np_input_ks).to(hp.use_device)
    input_qs = torch.tensor(np_input_qs).to(hp.use_device)
    targets = torch.tensor(np_targets).long().to(hp.use_device)

    input_qs = input_qs[..., None]
    targets = targets[..., None]

    perm_masks = torch.tensor(np_perm_masks).to(hp.use_device)
    target_masks = torch.tensor(np_target_masks).to(hp.use_device)

    if np_target_mappings is not None:
        target_mappings = torch.tensor(np_target_mappings).to(hp.use_device)
    else:
        target_mappings = None

    # loop through and UNK out to be assured of no leaks
    for k in range(np_data.shape[1]):
        blank_spots = np.where(orig_np_target_masks[:, k])[0]
        np_data[blank_spots, k, 0] = 67

    filled_sequences = [list() for i in range(np_data.shape[1])]
    finished_sampling = [False for i in range(np_data.shape[1])]
    non_pads = [[c < len(flat_measure_corpus.pitch_dictionary.idx2word) for c in true_data[:, k, 0]] for k in range(np_data.shape[1])]
    while not all(finished_sampling):
        if gibbs_iter > 0:
            print("gibb iter {}, step {} done".format(gibbs_iter, max(num_filled)))
        else:
            print("initial sample step {} done".format(max(num_filled)))

        np_targets = np_pitch_data[:-1]
        # will have to split this up
        np_input_ks = np_data[:-1]
        np_input_qs = np_target_masks

        input_ks = torch.tensor(np_input_ks).to(hp.use_device)

        #input_qs = None

        # modify perm masks and target masks?
        # once we sample, should we recreate the perm masks and target masks with one less

        out_h, out_g, out_mems = model(input_ks, input_qs, perm_masks, target_mappings, target_masks, list_of_mems=in_mems)
        temp = .9

        #unk_index = corpus.dictionary.word2idx["<unk>"]
        # don't let it predict UNK
        #out_g[:, :, unk_index] = -1E30

        reduced = top_p_from_logits_np(out_g.cpu().data.numpy() / temp, .95)
        # don't let it predict unk
        reduced_probs = softmax_np(reduced)

        for k in range(np_data.shape[1]):
            blank_spots = np.where(orig_np_target_masks[:, k])[0]
            si = int(num_filled[k])

            if si >= len(blank_spots):
                finished_sampling[k] = True
                continue

            if np_target_mappings is None:
                print("ARE WE HANDLING NP_TARGET_MAPPINGS CORRECTLY AND INDEXING APPROPRIATELY")
                from IPython import embed; embed(); raise ValueError()

            blank_spot = blank_spots[si]

            # blank out the mask? differs from training but still
            perm_masks[:, blank_spot, k] = 0.
            target_masks[blank_spot, k] = 0.
            input_qs[blank_spot, k, 0] = 0.

            if len(filled_sequences[k]) == 0:
                # true data 0, blanked out data 1
                filled_sequences[k].append([flat_measure_corpus.pitch_dictionary.idx2word[c] for n, c in enumerate(true_data[:, k, 0]) if non_pads[k][n]])
                context_sequence = [flat_measure_corpus.pitch_dictionary.idx2word[c] for n, c in enumerate(np_data[:, k, 0]) if non_pads[k][n]]
                filled_sequences[k].append(context_sequence)

            sampled_i = sampling_random_state.choice(np.arange(reduced_probs.shape[-1]), p=reduced_probs[si, k])

            np_data[blank_spot, k, 0] = sampled_i
            # nonpad
            new_context_sequence = [flat_measure_corpus.pitch_dictionary.idx2word[c] for n, c in enumerate(np_data[:, k, 0]) if non_pads[k][n]]
            filled_sequences[k].append(new_context_sequence)

            num_filled[k] += 1

    for i in range(np_data.shape[1]):
        # use true data to make non_pad!
        non_pad = [c < len(flat_measure_corpus.pitch_dictionary.idx2word) for c in true_data[:, i, 0]]
        pitches = [flat_measure_corpus.pitch_dictionary.idx2word[c] for n, c in enumerate(np_data[:, i, 0]) if non_pads[i][n]]
        durations = [flat_measure_corpus.duration_dictionary.idx2word[c] for n, c in enumerate(np_data[:, i, 1]) if non_pads[i][n]]
        voices = [vv for n, vv in enumerate(true_data[:, i, -1]) if non_pad[n]]

        quality_cut = hp.context_len + (len(pitches) - hp.context_len) // args.context_division
        data = convert_voice_lists_to_music_json(pitch_lists=pitches, duration_lists=durations, voices_list=voices)

        json_fpath = midi_sample_dir + os.sep + "sampled{}_step{}.json".format(i, gibbs_iter)
        write_music_json(data, json_fpath)

        fpath = midi_sample_dir + os.sep + "sampled{}_step{}.midi".format(i, gibbs_iter)
        music_json_to_midi(data, fpath)
        print("Wrote out {}".format(fpath))
