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
from kkpthlib import MusicJSONFlatKeyframeMeasureCorpus

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

filepath = "sampled_trainfiles.txt"
with open(filepath, 'w') as file_handler:
    for item in sorted(train_files):
        file_handler.write("{}\n".format(item))

filepath = "sampled_validfiles.txt"
with open(filepath, 'w') as file_handler:
    for item in sorted(valid_files):
        file_handler.write("{}\n".format(item))

# shuffle the train and valid files before we make the flat_measure_corpus
vrng.shuffle(train_files)
vrng.shuffle(valid_files)
flat_measure_corpus = MusicJSONFlatKeyframeMeasureCorpus(train_data_file_paths=train_files,
                                                         valid_data_file_paths=valid_files)

model = build_model(hp, flat_measure_corpus)

model_dict = torch.load(saved_model_path, map_location=hp.use_device)
model.load_state_dict(model_dict)
model.eval()

from kkpthlib import softmax_np
from kkpthlib import top_k_from_logits_np
from kkpthlib import top_p_from_logits_np
from kkpthlib import StepIterator
from kkpthlib import make_batches_from_list
from kkpthlib import MusicJSONRasterCorpus
from kkpthlib import convert_voice_lists_to_music_json
from kkpthlib import music_json_to_midi
from kkpthlib import write_music_json
from kkpthlib import pitch_duration_velocity_lists_from_music_json_file

# COPY PASTA FROM TRAIN FILE
# need this for now
def make_batches_from_indices(data_lists, indices_batch,
                              fill_value=-1):
    # use transformer convention, 0s unmasked 1s masked
    all_batches = []
    all_batches_masks = []
    for i in range(len(data_lists)):
        # currently works because fill value is -1
        tb = [[[data_lists[i][el] for el in indices_batch[:, b]] for b in range(indices_batch.shape[1])]]
        tb_mask = [[[el for el in indices_batch[:, b]] for b in range(indices_batch.shape[1])]]
        # has dim of 1, slice it temporarily then put it back
        tb = np.array(tb)
        if len(tb.shape) == 4:
            # edge case for 3d features
            tb = tb[0]
            tb = np.transpose(tb, (1, 0, 2))
        elif len(tb.shape) == 3:
            tb = np.transpose(tb, (2, 1, 0))
        elif len(tb.shape) == 2:
            tb = tb.T[:, :, None]
        # zero out everything that was equal to fill_value?
        batch_mask = np.array(tb_mask)
        batch_mask = (batch_mask == fill_value).astype(np.int32)
        # remove spurious axis, swap to seq len, batch
        batch_mask = batch_mask[0].T
        all_batches.append(tb)
        all_batches_masks.append(batch_mask)
    return all_batches, all_batches_masks

# need to handle measure boundaries
break_id = flat_measure_corpus.voices_dictionary.word2idx[9999]
train_cut_points = [el == break_id for el in flat_measure_corpus.train[5]]
valid_cut_points = [el == break_id for el in flat_measure_corpus.valid[5]]

train_batch_indices = make_batches_from_list(np.arange(len(flat_measure_corpus.train[0])), batch_size=hp.batch_size,
                                             sequence_length=hp.max_sequence_length, overlap=hp.context_len,
                                             cut_points=train_cut_points,
                                             fill_value=-1)
valid_batch_indices = make_batches_from_list(np.arange(len(flat_measure_corpus.valid[0])), batch_size=hp.batch_size,
                                             sequence_length=hp.max_sequence_length, overlap=hp.context_len,
                                             cut_points=valid_cut_points,
                                             fill_value=-1)

train_random_state = np.random.RandomState(hp.random_seed)
valid_random_state = np.random.RandomState(hp.random_seed + 1)
gen_random_state = np.random.RandomState(hp.random_seed + 2)

train_itr = StepIterator([train_batch_indices], circular_rotation=True, random_state=train_random_state)
valid_itr = StepIterator([valid_batch_indices], random_state=valid_random_state)

indices = next(valid_itr)

batches_list, batches_masks_list = make_batches_from_indices(flat_measure_corpus.valid, indices)

#indices = next(train_itr)
#batches_list, batches_masks_list = make_batches_from_indices(flat_measure_corpus.train, indices)
'''
[fingerprint_features_zero, : 0
fingerprint_features_one, : 1
duration_features_zero, : 2
duration_features_mid, : 3
duration_features_one, : 4
voices, : 5
centers, : 6
key_zero_base : 7
key_zero, : 8
key_durations_zero, : 9
key_one_base : 10
key_one, : 11
key_durations_one, : 12
key_indicators, : 13
targets : 14]
'''
# cut the features into chunks then feed without indicators
# this is ... messy but necessary

used_features = [0, 1, 2, 3, 4, 5, 7, 8, 9]

#feature_batches_list = [torch.tensor(b).to(hp.use_device) for n, b in enumerate(batches_list) if n in used_features]
#feature_batches_masks_list = [torch.tensor(mb).to(hp.use_device) for n, mb in enumerate(batches_masks_list) if n in used_features]

midi_sample_dir = "midi_samples"

if os.path.exists(midi_sample_dir):
    print("Previous midi_samples folder found, removing...")
    shutil.rmtree(midi_sample_dir)

# this should ALWAYS happen
if not os.path.exists(midi_sample_dir):
    os.mkdir(midi_sample_dir)

tstamp = int(time.time())
file_indices = list(range(30))

used_files = []
for fi in file_indices:
    errored = False
    random_seed = args.random_seed
    sampling_random_state = np.random.RandomState(random_seed)

    path = valid_files[fi]
    used_files.append(path)
    pitches, durations, velocities = pitch_duration_velocity_lists_from_music_json_file(path,
                                                                                        default_velocity=flat_measure_corpus.default_velocity,
                                                                                        n_voices=flat_measure_corpus.n_voices,
                                                                                        measure_value=flat_measure_corpus.measure_value,
                                                                                        fill_value=flat_measure_corpus.fill_value)
    final_all_pitches_list = None
    final_all_durations_list = None
    final_all_voices_list = None
    final_all_marked_quarters_context_boundary = None

    true_all_pitches_list = None
    true_all_durations_list = None
    true_all_voices_list = None
    true_all_marked_quarters_context_boundary = None

    last_voice = -1

    all_pitch_take_index = 0
    pitches_indexes = [0, 0, 0, 0]
    while True:
        print("Sampling {}, step {}".format(fi, all_pitch_take_index))
        try:
            # sometimes predict fingerprint tuples that never existed? might need to handle this in vocabulary
            feats = flat_measure_corpus._features_from_lists(pitches, durations, velocities)
            tokens = flat_measure_corpus._tokenize_features(feats)
            if all_pitch_take_index == 0:
                print(len(tokens[0]))
                if len(tokens[0]) > 300:
                    raise KeyError("")
        except KeyError:
            errored = True
            break

        batches_list = []
        batches_masks_list = []
        # find break points
        break_id = flat_measure_corpus.voices_dictionary.word2idx[9999]
        breaks = np.where(np.array(tokens[5]) == break_id)[0]

        for el in range(len(tokens)):
            tb = np.array(tokens[el])
            inds = np.arange(len(tb))
            inds = np.array([i for i in inds if i not in breaks])
            tb = tb[inds]
            shp = tb.shape
            new_tb = tb.repeat(hp.batch_size).reshape(shp + (hp.batch_size,))
            if len(new_tb.shape) > 2:
                new_tb = new_tb.transpose(0, 2, 1)
            else:
                new_tb = new_tb[..., None]
            batches_list.append(new_tb)
            m_new_tb = new_tb[..., 0] * 0
            m_new_tb[-1] = 1
            batches_masks_list.append(m_new_tb)

        batches_list = [b for n, b in enumerate(batches_list)]
        batches_masks_list = [b for n, b in enumerate(batches_masks_list)]
        feature_batches_list = [torch.tensor(b).to(hp.use_device) for n, b in enumerate(batches_list) if n in used_features]
        feature_batches_masks_list = [torch.tensor(b).to(hp.use_device) for n, b in enumerate(batches_masks_list) if n in used_features]

        # AFTER SANITY CHECK, SHIFTED TARGETS BECOME INPUTS TOO
        # 4 sets of targets, alternate definitions of the same value
        # distance from kp0_0 etc
        np_targets = batches_list[-1]
        list_of_inputs = feature_batches_list
        list_of_input_masks = feature_batches_masks_list
        targets = torch.tensor(np_targets).long().to(hp.use_device)

        # shift by one to account for the fact that keypoint features and targets are identical
        for _i in range(len(list_of_inputs)):
             list_of_inputs[_i] = torch.cat((list_of_inputs[_i][:1] * 0 + 0, list_of_inputs[_i]), 0)
             list_of_input_masks[_i] = torch.cat((list_of_input_masks[_i][:1] * 0 + 0, list_of_input_masks[_i]), 0)

        # shift targets by 1 and pass as inputs as well
        shifted_targets = torch.cat((targets[:1] * 0 + 0, targets), 0)
        # pad by 1 at the back to make targets match
        targets = torch.cat((targets, targets[:1] * 0 + 0), 0)

        if true_all_pitches_list == None:
            # get GT values to write out
            preds = targets[:-1, :, 0].cpu().data.numpy()
            all_pitches_list, all_durations_list, all_voices_list, all_marked_quarters_context_boundary = flat_measure_corpus.pitch_duration_voice_lists_from_preds_and_features(preds, batches_list, batches_masks_list, context_len=hp.context_len)

            true_all_pitches_list = copy.deepcopy(all_pitches_list)
            true_all_durations_list = copy.deepcopy(all_durations_list)
            true_all_voices_list = copy.deepcopy(all_voices_list)
            true_all_marked_quarters_context_boundary = copy.deepcopy(all_marked_quarters_context_boundary)

        list_of_inputs.extend([shifted_targets[:, :, 0][..., None],
                               shifted_targets[:, :, 1][..., None],
                               shifted_targets[:, :, 2][..., None],
                               shifted_targets[:, :, 3][..., None]])

        out0, out1, out2, out3, out_mems = model(list_of_inputs, list_of_input_masks, list_of_mems=None)
        #out0 already cut to [hp.context_len:, :, 0][..., None]
        temperature = args.temperature
        p_cutoff = args.p_cutoff
        reduced = top_p_from_logits_np(out0.cpu().data.numpy() / temperature, p_cutoff)
        reduced_probs = softmax_np(reduced)
        sampled = [[sampling_random_state.choice(np.arange(reduced_probs.shape[-1]), p=reduced_probs[ii, jj]) for jj in range(reduced_probs.shape[1])] for ii in range(len(reduced_probs))]
        #preds = reduced_probs.argmax(axis=-1)
        preds = np.array(sampled)

        #shuf_i = np.arange(len(preds))
        #np.random.RandomState(2122).shuffle(shuf_i)
        #preds = preds[shuf_i]

        #preds = targets[..., 0]
        preds = preds[:-1]
        #preds = preds[hp.context_len-1:]

        all_pitches_list, all_durations_list, all_voices_list, all_marked_quarters_context_boundary = flat_measure_corpus.pitch_duration_voice_lists_from_preds_and_features(preds, batches_list, batches_masks_list, context_len=hp.context_len)

        if all_pitch_take_index >= len(all_pitches_list[0]):
            print("Finished sampling {}".format(path))
            break

        vv = all_voices_list[0][all_pitch_take_index]
        pp = all_pitches_list[0][all_pitch_take_index]

        if durations[vv][pitches_indexes[vv]] == -1:
            # skip the blanks that denote phrase boundary
            pitches_indexes[vv] += 1

        if last_voice != vv:
            # fill in GT values for notes that are keypoints
            pp = pitches[vv][pitches_indexes[vv]]
            last_voice = vv

        if final_all_pitches_list is None:
            final_all_pitches_list = copy.deepcopy(all_pitches_list)
            final_all_durations_list = copy.deepcopy(all_durations_list)
            final_all_voices_list = copy.deepcopy(all_voices_list)
            final_all_marked_quarters_context_boundary = copy.deepcopy(all_marked_quarters_context_boundary)

        marks = all_marked_quarters_context_boundary[0]
        cur_dur = np.sum([d for d in durations[vv][:pitches_indexes[vv] + 1] if d > 0] + [0])
        if cur_dur >= marks[vv]:
            pitches[vv][pitches_indexes[vv]] = pp
            final_all_pitches_list[0][all_pitch_take_index] = pp

        pitches_indexes[vv] += 1
        all_pitch_take_index += 1

        del shifted_targets
        del targets
        del out0
        del out1
        del out2
        del out3
        del out_mems

    i = 0
    lim = -1
    if errored:
        # take a few steps off because errors tend to happen because of "bad choices" in notes before
        lim = all_pitch_take_index - int(.1 * all_pitch_take_index)
        if lim >= 0 and lim < 100:
            print("SKIPPED {}".format(fi))
            continue

    # handle whatever is giving last entry value of 10000
    pitches_list = final_all_pitches_list[i][:lim]
    durations_list = final_all_durations_list[i][:lim]
    voices_list = final_all_voices_list[i][:lim]
    marked_quarters_context_boundary = final_all_marked_quarters_context_boundary[i]

    data = convert_voice_lists_to_music_json(pitch_lists=pitches_list, duration_lists=durations_list, voices_list=voices_list)
    json_fpath = midi_sample_dir + os.sep + "sampled{}_{}.json".format(fi, tstamp)
    write_music_json(data, json_fpath)

    fpath = midi_sample_dir + os.sep + "sampled{}_{}.midi".format(fi, tstamp)
    a = "harpsichord_preset"
    b = "woodwind_preset"
    m = {0: [(a, 0), (b, marked_quarters_context_boundary[0])],
         1: [(a, 0), (b, marked_quarters_context_boundary[1])],
         2: [(a, 0), (b, marked_quarters_context_boundary[2])],
         3: [(a, 0), (b, marked_quarters_context_boundary[3])]}

    a = "harpsichord_preset"
    m = {0: [(a, 0),],
         1: [(a, 0),],
         2: [(a, 0),],
         3: [(a, 0),]}
    music_json_to_midi(data, fpath, voice_program_map=m)
    print("Wrote out {}".format(fpath))

    pitches_list = true_all_pitches_list[i][:lim]
    durations_list = true_all_durations_list[i][:lim]
    voices_list = true_all_voices_list[i][:lim]
    marked_quarters_context_boundary = true_all_marked_quarters_context_boundary[i]

    data = convert_voice_lists_to_music_json(pitch_lists=pitches_list, duration_lists=durations_list, voices_list=voices_list)
    json_fpath = midi_sample_dir + os.sep + "true{}_{}.json".format(fi, tstamp)
    write_music_json(data, json_fpath)

    fpath = midi_sample_dir + os.sep + "true{}_{}.midi".format(fi, tstamp)
    music_json_to_midi(data, fpath, voice_program_map=m)
    print("Wrote out {}".format(fpath))

filepath = "used_files.txt"
with open(filepath, 'w') as file_handler:
    for item in sorted(used_files):
        file_handler.write("{}\n".format(item))

print("Total sampling time: {} seconds".format(time.time() - tstamp))
