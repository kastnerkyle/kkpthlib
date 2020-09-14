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
parser.add_argument('--temperature', '-t', type=float, default=.9,
                    help='sampling temperature to use (default .9)')
parser.add_argument('--p_cutoff', '-p', type=float, default=.95,
                    help='cutoff to use in top p sampling (default .95)')

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

# shuffle the train and valid files before we make the flat_measure_corpus
vrng.shuffle(train_files)
vrng.shuffle(valid_files)
flat_measure_corpus = MusicJSONFlatKeyframeMeasureCorpus(train_data_file_paths=train_files,
                                                         valid_data_file_paths=valid_files)

model = build_model(hp, flat_measure_corpus)

model_dict = torch.load(saved_model_path, map_location=hp.use_device)
model.load_state_dict(model_dict)
#model.eval()

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

used_features = [0, 2, 3, 4, 5, 7, 8, 9]
#feature_batches_list = [torch.tensor(b).to(hp.use_device) for n, b in enumerate(batches_list) if n in used_features]
#feature_batches_masks_list = [torch.tensor(mb).to(hp.use_device) for n, mb in enumerate(batches_masks_list) if n in used_features]

path = valid_files[0]
pitches, durations, velocities = pitch_duration_velocity_lists_from_music_json_file(path,
                                                                                    default_velocity=flat_measure_corpus.default_velocity,
                                                                                    n_voices=flat_measure_corpus.n_voices,
                                                                                    measure_value=flat_measure_corpus.measure_value,
                                                                                    fill_value=flat_measure_corpus.fill_value)
feats = flat_measure_corpus._features_from_lists(pitches, durations, velocities)
tokens = flat_measure_corpus._tokenize_features(feats)

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
targets = torch.cat((targets[:1] * 0 + 0, targets), 0)
# re-adjust to be sure they are the same length
shifted_targets = targets[:-1]
targets = targets[1:]

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
preds = reduced_probs.argmax(axis=-1)

# need to write code to put the audio back together again from given or predictable info
midi_sample_dir = "midi_samples"

if os.path.exists(midi_sample_dir):
    print("Previous midi_samples folder found, removing...")
    shutil.rmtree(midi_sample_dir)

# this should ALWAYS happen
if not os.path.exists(midi_sample_dir):
    os.mkdir(midi_sample_dir)

targets2 = copy.deepcopy(targets.cpu().data.numpy())
preds[:hp.context_len] = targets2[:hp.context_len, :, 0]

#shuf_i = np.arange(len(preds))
#np.random.RandomState(2122).shuffle(shuf_i)
#preds = preds[shuf_i]

#preds = targets[..., 0]
#preds = preds[hp.context_len-1:]

all_pitches_list, all_durations_list, all_voices_list, all_marked_quarters_context_boundary = flat_measure_corpus.pitch_duration_voice_lists_from_preds_and_features(preds, batches_list, batches_masks_list, context_len=hp.context_len)

for i in range(len(all_pitches_list)):
    # handle whatever is giving last entry value of 10000
    pitches_list = all_pitches_list[i][:-1]
    durations_list = all_durations_list[i][:-1]
    voices_list = all_voices_list[i][:-1]
    marked_quarters_context_boundary = all_marked_quarters_context_boundary[i]

    data = convert_voice_lists_to_music_json(pitch_lists=pitches_list, duration_lists=durations_list, voices_list=voices_list)
    tstamp = int(time.time())
    json_fpath = midi_sample_dir + os.sep + "true{}_{}.json".format(i, tstamp)
    write_music_json(data, json_fpath)

    fpath = midi_sample_dir + os.sep + "true{}_{}.midi".format(i, tstamp)
    a = "harpsichord_preset"
    b = "woodwind_preset"
    m = {0: [(a, 0), (b, marked_quarters_context_boundary[0])],
         1: [(a, 0), (b, marked_quarters_context_boundary[1])],
         2: [(a, 0), (b, marked_quarters_context_boundary[2])],
         3: [(a, 0), (b, marked_quarters_context_boundary[3])]}
    music_json_to_midi(data, fpath, voice_program_map=m)
    print("Wrote out {}".format(fpath))
from IPython import embed; embed(); raise ValueError()
