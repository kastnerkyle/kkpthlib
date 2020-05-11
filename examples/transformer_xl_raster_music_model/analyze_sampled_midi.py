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
parser.add_argument('sampled_outputs_path', type=str, default="")
parser.add_argument('--data_seed', '-d', type=int, default=144,
                    help='random seed to use when seeding the sampling (default 144)')

args = parser.parse_args()
if args.sampled_outputs_path == "":
    parser.print_help()
    sys.exit(1)

sampled_outputs_path = args.sampled_outputs_path

if not os.path.exists(sampled_outputs_path):
    raise ValueError("Unable to find argument {} for file load! Check your paths, and be sure to remove aliases such as '~'".format(saved_outputs_path))

from kkpthlib import StepIterator
from kkpthlib import make_batches_from_list
from kkpthlib import fetch_jsb_chorales
from kkpthlib import piano_roll_from_music_json_file
from kkpthlib import MusicJSONCorpus
from kkpthlib import convert_voice_roll_to_pitch_duration
from kkpthlib import music_json_to_midi
from kkpthlib import write_music_json
from kkpthlib import get_music21_metadata
from kkpthlib import build_music_plagiarism_checkers
from kkpthlib import evaluate_music_against_checkers

import gc
import time

try:
    import cPickle as pickle
except ImportError:
    import pickle

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

# if it is the very firstrun, assume_cached will be false!
meta = get_music21_metadata(jsb["files"], assume_cached=True)
metafiles = meta["files"]

cached_checkers_path = "cached_checkers.pkl"
if not os.path.exists(cached_checkers_path):
    checkers = build_music_plagiarism_checkers(metafiles)
    # disabling gc can help speed up pickle
    gc.disable()
    print("Caching checkers to {}".format(cached_checkers_path))
    start = time.time()
    with open(cached_checkers_path, 'wb') as f:
        pickle.dump(checkers, f, protocol=-1)
    end = time.time()
    print("Time to cache {}s".format(end - start))
    gc.enable()
else:
    print("Loading cached checkers from {}".format(cached_checkers_path))
    start = time.time()
    with open(cached_checkers_path, 'rb') as f:
        checkers = pickle.load(f)
    end = time.time()
    print("Time to load {}s".format(end - start))

midi_path = sampled_outputs_path + "/sampled0.midi"
evaluate_music_against_checkers(midi_path, checkers)

#corpus = MusicJSONCorpus(train_data_file_paths=jsb["files"])
