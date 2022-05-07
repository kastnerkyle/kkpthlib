from kkpthlib import fetch_jsb_chorales
from kkpthlib import fetch_josquin
from kkpthlib import MusicJSONCorpus
from kkpthlib import convert_voice_lists_to_music_json
from kkpthlib import music_json_to_midi
from kkpthlib import write_music_json
import os
import numpy as np
import re
import json


#jsb = fetch_jsb_chorales()
jrp = fetch_josquin()

"""
all_keys = {}
for _j, fpath in enumerate(jrp["files"]):
    print(_j, len(jrp["files"]))
    with open(fpath, "r") as f:
        jinfo = json.load(f)
        notes_keys = list(sorted(jinfo["notes"].keys(), key=lambda x: int(x.split("_")[-1])))[::-1]
        for k in notes_keys:
            if k not in all_keys:
                all_keys[k] = 1
            else:
                all_keys[k] += 1
"""

fpaths = jrp["files"]
all_base_files = sorted(list(set([fpath.split(os.sep)[-1].split("-")[0] for fpath in fpaths])))
holdout_bases = []
split_random_state = np.random.RandomState(41412)
for comp in ["Tin", "Rue", "Ort", "Ock", "Obr", "Mou", "Mar", "Jos", "Jap", "Isa", "Fva", "Duf", "Com", "Bus", "Bru", "Ano", "Agr"]:
    this_comp_base = sorted([base for base in all_base_files if comp in base])
    split_random_state.shuffle(this_comp_base)
    holdout_bases.append(this_comp_base[0])
holdout_fpaths = []
for hold in holdout_bases:
     holdout_fpaths.extend([fpath for fpath in fpaths if hold in fpath])
holdout_lu = {k: None for k in holdout_fpaths}
full_fpaths = fpaths
fpaths = [fpath for fpath in fpaths if fpath not in holdout_lu]

all_final_measure_dur = {}
all_note_dur = {}
T = 192
total_files = 0
all_arr_chunks = []
all_boundary_chunks = []
global_min_pitch = None
global_max_pitch = None
for fpath in fpaths:
    with open(fpath, "r") as f:
        jinfo = json.load(f)
    notes_keys = list(sorted(jinfo["notes"].keys(), key=lambda x: int(x.split("_")[-1])))[::-1]
    all_max_end = []
    all_max_note = []
    all_min_note = []
    all_last_measure_num = []
    skip_this_entry = False
    for k in notes_keys:
        part = jinfo["notes"][k][0]
        max_end = max([el[-2] + el[-1] for el in part])
        all_max_end.append(max_end)

        note_pitch = [el[0] for el in part if el[0] > 0]
        note_dur = [el[-1] for el in part]
        if len(note_pitch) > 0:
            max_note = max(note_pitch)
            min_note = min(note_pitch)
            all_max_note.append(max_note)
            all_min_note.append(min_note)
        all_last_measure_num.append(part[-1][2])
        if np.sum([e % 0.5 for e in note_dur]) != 0:
            skip_this_entry = True
            break
    final_max_measure_dur = max(all_max_end)
    final_max_note = max(all_max_note)
    final_min_note = min(all_min_note)
    final_measure = max(all_last_measure_num)

    if skip_this_entry:
        continue

    if global_min_pitch is None:
        global_min_pitch = final_min_note
    else:
        global_min_pitch = min(global_min_pitch, final_min_note)

    if global_max_pitch is None:
        global_max_pitch = final_max_note
    else:
        global_max_pitch = max(global_max_pitch, final_max_note)

    note_durs = list(set(note_dur))
    all_note_dur.update([(e, None) for e in note_durs])


    all_final_measure_dur[final_max_measure_dur] = None

    # {12.0: None, 8.0: None, 16.0: None, 24.0: None, 32.0: None, 4.0: None}

    # want 48 or 64 per measure for 3 or 4 beats per measure...
    # means length of 192, might be tight on memory
    if final_max_measure_dur == 4.0:
        # 64
        dur_mult = 16
        stretch_meas = 64
    elif final_max_measure_dur == 8.0:
        dur_mult = 8
        stretch_meas = 64
    elif final_max_measure_dur == 12.0:
        dur_mult = 4
        stretch_meas = 48
    elif final_max_measure_dur == 16.0:
        dur_mult = 4
        stretch_meas = 64
    elif final_max_measure_dur == 24.0:
        dur_mult = 2
        stretch_meas = 48
    elif final_max_measure_dur == 32.0:
        dur_mult = 2
        stretch_meas = 64
    else:
        raise ValueError("Unhandled measure dur")

    overall_arr = np.zeros((4, int((final_measure + 1) * stretch_meas), 100 + 1))
    overall_boundaries = np.zeros((4, int((final_measure + 1) * stretch_meas), 100 + 1))
    # make the full array from 0 to 88, stack then slice later?
    for _n, k in enumerate(notes_keys):
        part = jinfo["notes"][k][0]
        for el in part:
            s = int(el[2] * stretch_meas + el[-2] * dur_mult)
            e = int(el[2] * stretch_meas + el[-2] * dur_mult + el[-1] * dur_mult)
            overall_arr[_n, s:e, el[0]] = 1.
            overall_boundaries[_n, s, el[0]] = 1.
            #overall_boundaries[_n, max(0, e - 1), el[0]] = 1.
    overall_arr = overall_arr.astype("bool")
    overall_boundaries = overall_boundaries.astype("bool")

    cut = 0
    while cut < overall_arr.shape[1] - T:
        all_arr_chunks.append(overall_arr[:, cut:cut+T])
        all_boundary_chunks.append(overall_boundaries[:, cut:cut + T])
        cut += T
    total_files += 1
from IPython import embed; embed(); raise ValueError()
