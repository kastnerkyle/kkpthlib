from __future__ import print_function
import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F
import copy

from kkpthlib import fetch_jsb_chorales
from kkpthlib import MusicJSONRasterIterator
from kkpthlib import softmax_np
from kkpthlib import get_logger
from kkpthlib import CategoricalCrossEntropy

from transformer_music_model import get_hparams
from transformer_music_model import build_model

from minimal_beamsearch import top_k_from_logits_np
from minimal_beamsearch import top_p_from_logits_np

from kkpthlib import music_json_to_midi

import json
import os

jsb = fetch_jsb_chorales()
logger = get_logger()

hp = get_hparams()

logger.info("Saved vocab {} found, loading...".format(hp.vocab_storage))
d = np.load(hp.vocab_storage)
vocab = d["vocab"]

vocab_mapper = {k: v + 1000 for k, v in zip(vocab, range(len(vocab)))}
inverse_vocab_mapper = {v - 1000:k for k, v in vocab_mapper.items()}

m = build_model(hp)
loss_fun = CategoricalCrossEntropy()

# "sampling" code
fake_itr = MusicJSONRasterIterator(jsb["files"][-50:],
                                    batch_size=hp.batch_size,
                                    max_sequence_length=hp.max_sequence_length,
                                    with_clocks=hp.clocks,
                                    random_seed=hp.random_seed)

m_dict = torch.load("model_checkpoint.pth", map_location=hp.use_device)
m.load_state_dict(m_dict)

# just use it for the clocks for now, eventually will need to generate
piano_roll, mask, clocks = next(fake_itr)
# set mask to all 1s
mask = 0. * mask + 1.

# 16 0s to represent 4 blank *start* bars
true_piano_roll = np.zeros((16, hp.batch_size, 1))
sampling_random_state = np.random.RandomState(13)
steps = 1000
assert steps % 4 == 0
for t in range(steps):
    print("sampling_step {}".format(t))
    piano_roll = copy.deepcopy(true_piano_roll)
    for k in vocab_mapper.keys():
        piano_roll[piano_roll == k] = vocab_mapper[k]
    # move it down to proper range
    piano_roll = piano_roll - 1000.

    this_piano_roll = torch.Tensor(piano_roll).to(hp.use_device)
    this_mask = torch.Tensor(mask[:len(piano_roll)]).to(hp.use_device)
    # trim off one for AR prediction
    this_clocks = [torch.Tensor(c[:len(piano_roll)]).to(hp.use_device) for c in clocks]

    linear_out, past = m(this_piano_roll, this_mask, this_clocks)
    reduced = top_p_from_logits_np(linear_out.cpu().data.numpy(), .9)
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
                    print("got non onset, non contiguous continuation token")
                    from IPython import embed; embed(); raise ValueError()
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
    music_json_to_midi(data, midi_sample_dir + os.sep + "temp{}.midi".format(mb))
