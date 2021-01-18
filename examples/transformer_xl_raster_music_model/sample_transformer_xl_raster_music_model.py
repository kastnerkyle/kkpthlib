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
from kkpthlib import MusicJSONRasterCorpus
from kkpthlib import convert_voice_roll_to_music_json
from kkpthlib import music_json_to_midi
from kkpthlib import write_music_json

jsb = fetch_jsb_chorales()

# sort into minor / major, then by key
all_transposed = sorted([f for f in jsb["files"] if "original" not in f], key=lambda x:
                        (x.split(os.sep)[-1].split(".")[-2].split("transposed")[0].split("-")[0],
                         x.split(os.sep)[-1].split(".")[-2].split("transposed")[0].split("-")[1]))

bwv_names = sorted(list(set([f.split(os.sep)[-1].split(".")[0] for f in all_transposed])))

data_seed = args.data_seed

vrng = np.random.RandomState(data_seed)
vrng.shuffle(bwv_names)

# 15 is ~5% of the data
# holding out whole songs so actually a pretty hard validation set...
valid_names = bwv_names[:15]
train_files = [f for f in all_transposed if all([vn not in f for vn in valid_names])]
valid_files = [f for f in all_transposed if any([vn in f for vn in valid_names])]

# now we want to aggregate "start" seeds from each of the files
batch_size = args.batch_size
assert batch_size <= len(valid_files)

sample_len = args.sample_len

seed_batches = []
for i in range(batch_size):
    corpus = MusicJSONRasterCorpus(train_data_file_paths=train_files,
                                   valid_data_file_paths=[valid_files[i]])
    seed_batches.append(np.array(corpus.valid[:sample_len]))

# make the full corpus to ensure vocabulary matches training
corpus = MusicJSONRasterCorpus(train_data_file_paths=train_files,
                               valid_data_file_paths=valid_files)

min_len = min([sb.shape[0] for sb in seed_batches])
if sample_len < min_len:
    min_len = sample_len

seed_batches = [sb[:min_len] for sb in seed_batches]
np_data = np.array(seed_batches).T
true_data = np_data.copy()

np_data = np_data[:hp.context_len + 1]
# set same seed for all, check variability
np_data_0 = np_data[:, 0]
np_data[:, :] = np_data_0[:, None]

out_mems = None

context_mult = args.context_mult
temperature = args.temperature
p_cutoff = args.p_cutoff

random_seed = args.random_seed
sampling_random_state = np.random.RandomState(random_seed)

for i in range(sample_len - hp.context_len):
    context_sentence = [corpus.dictionary.idx2word[c] for c in np_data[:hp.context_len + 1, 0]]
    sampled_sentence = [corpus.dictionary.idx2word[c] for c in np_data[hp.context_len + 1:, 0]]
    print("==================")
    print("step {}".format(i))
    print("context: {}".format(context_sentence))
    print("sampled: {}".format(sampled_sentence))
    print("==================")
    input_data = torch.tensor(np_data[-(context_mult * (hp.context_len + 1)):]).to(hp.use_device)
    #input_data = torch.tensor(np_data).to(hp.use_device)
    input_data = input_data[..., None]

    in_mems = out_mems
    linear_out, out_mems = model(input_data, list_of_mems=in_mems)

    reduced = top_p_from_logits_np(linear_out.cpu().data.numpy() / temperature, p_cutoff)
    reduced_probs = softmax_np(reduced)
    sample_last = [sampling_random_state.choice(np.arange(reduced_probs.shape[-1]), p=reduced_probs[-1, j]) for j in range(reduced_probs.shape[1])]
    np_data = np.concatenate((np_data, np.array(sample_last)[None]), axis=0)

midi_sample_dir = "midi_samples"
if not os.path.exists(midi_sample_dir):
    os.mkdir(midi_sample_dir)

for i in range(np_data.shape[1]):
    sampled_sentence = [corpus.dictionary.idx2word[c] for c in np_data[:, i]]
    sampled_voice_roll = np.array(sampled_sentence[:len(sampled_sentence) // 4 * 4]).reshape(-1, 4).T
    data = convert_voice_roll_to_music_json(sampled_voice_roll)

    json_fpath = midi_sample_dir + os.sep + "sampled{}.json".format(i)
    write_music_json(data, json_fpath)

    fpath = midi_sample_dir + os.sep + "sampled{}.midi".format(i)
    music_json_to_midi(data, fpath)
    print("Wrote out {}".format(fpath))

for i in range(true_data.shape[1]):
    true_sentence = [corpus.dictionary.idx2word[c] for c in true_data[:, i]]
    true_voice_roll = np.array(true_sentence[:len(true_sentence) // 4 * 4]).reshape(-1, 4).T
    data = convert_voice_roll_to_music_json(true_voice_roll)

    json_fpath = midi_sample_dir + os.sep + "true{}.json".format(i)
    write_music_json(data, json_fpath)

    fpath = midi_sample_dir + os.sep + "true{}.midi".format(i)
    music_json_to_midi(data, fpath)
    print("Wrote out {}".format(fpath))
