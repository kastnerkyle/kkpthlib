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

help_str = "\nUsage:\n{} model_training_or_definition_file.py path_to_checkpoint_file.pth\n".format(sys.argv[0])

if len(sys.argv) < 3:
    print(help_str)
    sys.exit()

if sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print(help_str)
    sys.exit()

if not os.path.exists(sys.argv[1]):
    raise ValueError("Unable to find argument {} for file load! Check your paths".format(sys.argv[1]))

model_file_path = sys.argv[1]
checkpoint_path = sys.argv[2]


from kkpthlib import fetch_jsb_chorales
from kkpthlib import MusicJSONRasterIterator
from kkpthlib import softmax_np
from kkpthlib import get_logger
from kkpthlib import CategoricalCrossEntropy

# longer term, make this a dedicated function in the library?
# loads get_hparams and build_model from the training file
# if that's really a requirement, better check em too
from importlib import import_module
model_import_path, fname = model_file_path.rsplit(os.sep, 1)
sys.path.insert(0, model_import_path)
p, _ = fname.split(".", 1)
mod = import_module(p)
get_hparams = getattr(mod, "get_hparams")
build_model = getattr(mod, "build_model")
sys.path.pop(0)

#from transformer_music_model import get_hparams
#from transformer_music_model import build_model

from minimal_beamsearch import top_k_from_logits_np
from minimal_beamsearch import top_p_from_logits_np

from kkpthlib import music_json_to_midi

jsb = fetch_jsb_chorales()
logger = get_logger()

hp = get_hparams()

logger.info("Saved vocab {} found, loading...".format(hp.vocab_storage))
d = np.load(hp.vocab_storage)
vocab = d["vocab"]

vocab_mapper = {k: v + 1000 for k, v in zip(vocab, range(len(vocab)))}
inverse_vocab_mapper = {v - 1000:k for k, v in vocab_mapper.items()}

model = build_model(hp)
loss_fun = CategoricalCrossEntropy()

# "sampling" code
fake_itr = MusicJSONRasterIterator(jsb["files"][-50:],
                                   batch_size=hp.batch_size,
                                   max_sequence_length=hp.max_sequence_length,
                                   with_clocks=hp.clocks,
                                   random_seed=hp.random_seed)


m_dict = torch.load(checkpoint_path, map_location=hp.use_device)
model.load_state_dict(m_dict)
model.eval()

# just use it for the clocks for now, eventually will need to generate clocks from scratch
_, mask, clocks = next(fake_itr)
# set mask to all 1s
mask = 0. * mask + 1.

# 16 0s to represent 4 blank *start* bars
true_piano_roll = np.zeros((16, hp.batch_size, 1))
partial_piano_roll = copy.deepcopy(true_piano_roll)

sampling_random_state = np.random.RandomState(13)
list_of_memories = None
memory_mask = None
steps = hp.max_sequence_length - len(true_piano_roll)
for t in range(steps):
    print("sampling step {} / {}".format(t, steps))
    piano_roll = copy.deepcopy(true_piano_roll)
    for k in vocab_mapper.keys():
        piano_roll[piano_roll == k] = vocab_mapper[k]
    # move it down to proper range
    piano_roll = piano_roll - 1000.

    this_piano_roll = torch.Tensor(piano_roll).to(hp.use_device)
    this_mask = torch.Tensor(mask[:len(piano_roll)]).to(hp.use_device)
    # trim off one for AR prediction
    this_clocks = [torch.Tensor(c[:len(piano_roll)]).to(hp.use_device) for c in clocks]

    linear_out = model(this_piano_roll, this_mask, this_clocks)
    temp = .1
    reduced = top_p_from_logits_np(linear_out.cpu().data.numpy() / temp, .95)
    reduced_probs = softmax_np(reduced)

    sample_last = [sampling_random_state.choice(np.arange(reduced_probs.shape[-1]), p=reduced_probs[-1, j]) for j in range(reduced_probs.shape[1])]
    true_sample_last = [inverse_vocab_mapper[s] for s in sample_last]
    sampled_arr = np.array(true_sample_last)[None, :, None]
    true_piano_roll = np.concatenate((true_piano_roll, sampled_arr), axis=0)


def convert_voice_roll_to_pitch_duration(voice_roll, duration_step=.25):
    """
    take in voice roll and turn it into a pitch, duration thing again

    currently assume onsets are any notes > 100 , 0 is rest

    example input, where 170, 70, 70, 70 is an onset of pitch 70 (noted as 170), followed by a continuation for 4 steps
    array([[170.,  70.,  70.,  70.],
           [165.,  65.,  65.,  65.],
           [162.,  62.,  62.,  62.],
           [158.,  58.,  58.,  58.]])
    """
    voice_data = {}
    voice_data["parts"] = []
    voice_data["parts_times"] = []
    voice_data["parts_cumulative_times"] = []
    for v in range(voice_roll.shape[0]):
        voice_data["parts"].append([])
        voice_data["parts_times"].append([])
        voice_data["parts_cumulative_times"].append([])
    for v in range(voice_roll.shape[0]):
        ongoing_duration = duration_step
        note_held = 0
        for t in range(len(voice_roll[v])):
            token = int(voice_roll[v][t])
            if voice_roll[v][t] > 100:
                voice_data["parts"][v].append(note_held)
                voice_data["parts_times"][v].append(ongoing_duration)
                ongoing_duration = duration_step
                note_held = token - 100
            elif token != 0:
                if token != note_held:
                    # make it an onset?
                    print("WARNING: got non-onset pitch change, forcing onset token at step {}, voice {}".format(t, v))
                    note_held = token
                    ongoing_duration = duration_step
                else:
                    ongoing_duration += duration_step
            else:
                # just adding 16th note silences?
                ongoing_duration = duration_step
                note_held = 0
                voice_data["parts"][v].append(note_held)
                voice_data["parts_times"][v].append(ongoing_duration)
        voice_data["parts_cumulative_times"][v] = [e for e in np.cumsum(voice_data["parts_times"][v])]
    spq = .5
    ppq = 220
    qbpm = 120
    voice_data["seconds_per_quarter"] = spq
    voice_data["quarter_beats_per_minute"] = qbpm
    voice_data["pulses_per_quarter"] = ppq
    voice_data["parts_names"] = ["Soprano", "Alto", "Tenor", "Bass"]
    j = json.dumps(voice_data, indent=4)
    return j


midi_sample_dir = "midi_samples"
if not os.path.exists(midi_sample_dir):
    os.mkdir(midi_sample_dir)

for mb in range(true_piano_roll.shape[1]):
    reshaped_voice_roll = true_piano_roll[:, mb].ravel().reshape(true_piano_roll.shape[0] // 4, 4).transpose(1, 0)
    data = convert_voice_roll_to_pitch_duration(reshaped_voice_roll)
    fpath = midi_sample_dir + os.sep + "temp{}.midi".format(mb)
    music_json_to_midi(data, fpath)
    print("wrote out {}".format(fpath))
