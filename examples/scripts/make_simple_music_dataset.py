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


jsb = fetch_jsb_chorales()
fpaths = jsb["files"]
#jrp = fetch_josquin()
#fpaths = jrp["files"]
def get_base_file(fpath):
    return fpath.split(os.sep)[-1].split("-")[0].split("_")[0]

all_base_files = sorted(list(set([get_base_file(fpath) for fpath in fpaths])))

# use 95-5 split
split_random_state = np.random.RandomState(1234)
total_bases = len(all_base_files)
split_point = int(.95 * total_bases)
split_random_state.shuffle(all_base_files)
training_bases = all_base_files[:split_point]
holdout_bases = all_base_files[split_point:]
training_fpaths = []
holdout_fpaths = []
for fpath in fpaths:
    base = get_base_file(fpath)
    if base in holdout_bases:
        holdout_fpaths.append(fpath)
    elif base in training_bases:
        training_fpaths.append(fpath)
    else:
        raise ValueError("base not found in either case! {} , fpath {}".format(base, fpath))

for _step in [0, 1]:
    if _step == 0:
        fpaths = training_fpaths
    else:
        fpaths = holdout_fpaths

    all_final_measure_dur = {}
    all_note_dur = {}
    T = 128
    Q = 16
    B = 4.0
    # go for 144 so 3 and 4 will both be on beat
    total_files = 0
    all_arr_chunks = []
    all_boundary_chunks = []
    all_attribution_chunks = []
    all_arr = []
    all_boundary = []
    all_attribution = []
    skipped_files = 0
    # time signatures verified with https://www.bach-chorales.com/BWV0115_6.htm

    # issues: 244.29a is in music21, but 244.29b is in the website. Labeled as 4/4 for now
    # 177.4 is not on the website had to use https://musescore.com/user/17782181/scores/7243656  
    # 248.33-3 does not technically have a pickup but has a strange fermata structure that seems to make music21 parse it weirdly
    # 248.33-3 also shows an example of "editing" - some collections remove the parallel 5th here!
    # bwv284 no pickup but has a fermata measure with only 2 beats technically
    # bwv295 also has fermata 2 beat meas
    # bwv317 has no pickup, looks normal but might be poorly parsed due to repeat structure
    # bwv324 is bizarre I have no idea how to read it
    # bwv325 seems to be normal no pickup
    # bwv363 has a 3 beat fermata at the end
    # bwv248.64 has 32nds

    checks_4_4 = ["bwv10.7_", "bwv115.6_", "bwv145-a_", "bwv146.8_", "bwv154.3_",
                  "bwv157.5_", "bwv159.5_", "bwv162.6-lpz_", "bwv180.7_", "bwv194.6_",
                  "bwv227.11_", "bwv227.1_", "bwv227.7_",  "bwv244.29-a_", "bwv244.40_",
                  "bwv245.14_", "bwv245.15_", "bwv245.28_", "bwv245.37_", "bwv248.53-5_",
                  "bwv25.6_", "bwv259_", "bwv26.6_", "bwv262_", "bwv263_",
                  "bwv283_", "bwv292_", "bwv30.6_", "bwv301_", "bwv318_",
                  "bwv32.6_", "bwv323_", "bwv329_", "bwv330_", "bwv331_",
                  "bwv337_", "bwv339_", "bwv352_", "bwv353_", "bwv354_",
                  "bwv355_", "bwv357_", "bwv358_", "bwv359_", "bwv36.8-2_",
                  "bwv360_", "bwv361_", "bwv373_", "bwv379_", "bwv38.6_",
                  "bwv381_", "bwv39.7_", "bwv40.6_", "bwv40.8_", "bwv405_",
                  "bwv407_", "bwv408_", "bwv423_", "bwv433_", "bwv437_",
                  "bwv55.5_", "bwv56.5_", "bwv60.5_", "bwv64.8_", "bwv78.7_",
                  "bwv81.7_", "bwv87.7_", "bwv99.6_",
                  # begin holdout
                  "bwv310_", "bwv365_", "bwv424_", "bwv65.7_",
                  # begin pickup train
                  "bwv101.7_", "bwv102.7_", "bwv103.6_", "bwv104.6_", "bwv108.6_",
                  "bwv110.7_", "bwv111.6_", "bwv112.5_", "bwv113.8_", "bwv114.7_",
                  "bwv116.6_", "bwv117.4_", "bwv119.9_", "bwv120.6_", "bwv121.6_",
                  "bwv125.6_", "bwv126.6_", "bwv127.5_", "bwv13.6_", "bwv133.6_",
                  "bwv135.6_", "bwv139.6_", "bwv14.5_", "bwv140.7_", "bwv144.3_",
                  "bwv148.6_", "bwv151.5_", "bwv153.5_", "bwv153.1_", "bwv154.8_",
                  "bwv155.5_", "bwv156.6_", "bwv158.4_", "bwv16.6_", "bwv164.6_",
                  "bwv165.6_", "bwv166.6_", "bwv168.6_", "bwv169.7_", "bwv174.5_",
                  "bwv176.6_", "bwv177.4_", "bwv177.5_", "bwv178.7_", "bwv179.6_",
                  "bwv18.5-lz_", "bwv18.5-w_", "bwv183.5_", "bwv184.5_", "bwv188.6_",
                  "bwv190.7_", "bwv197.10_", "bwv197.5_", "bwv197.7-a_", "bwv2.6_",
                  "bwv20.11_", "bwv20.7_", "bwv226.2_", "bwv24.6_", "bwv244.10_",
                  "bwv244.15_", "bwv244.17_", "bwv244.25_", "bwv244.32_", "bwv244.37_",
                  "bwv244.3_", "bwv244.44_", "bwv244.46_", "bwv244.54_", "bwv244.62_",
                  "bwv245.11_", "bwv245.17_", "bwv245.22_", "bwv245.26_", "bwv245.3_",
                  "bwv245.40_", "bwv245.5_", "bwv248.12-2_",  "bwv248.28_", "bwv248.33-3_",
                  "bwv248.35-3_", "bwv248.46-5_", "bwv248.5_", "bwv248.64-s_", "bwv253_",
                  "bwv254_", "bwv255_", "bwv256_", "bwv258_", "bwv260_",
                  "bwv261_", "bwv264_", "bwv265_", "bwv267_", "bwv268_",
                  "bwv270_", "bwv271_", "bwv272_", "bwv273_", "bwv274_",
                  "bwv275_", "bwv276_", "bwv277_", "bwv278_", "bwv28.6_",
                  "bwv280_", "bwv281_", "bwv284_", "bwv285_", "bwv286_",
                  "bwv288_", "bwv290_", "bwv291_", "bwv294_", "bwv296_",
                  "bwv297_", "bwv298_", "bwv3.6_", "bwv300_", "bwv302_",
                  "bwv303_", "bwv305_", "bwv307_", "bwv308_", "bwv309_",
                  "bwv311_", "bwv312_", "bwv313_", "bwv314_", "bwv315_",
                  "bwv316_", "bwv317_", "bwv319_", "bwv322_", "bwv324_",
                  "bwv325_", "bwv328_", "bwv33.6_", "bwv332_", "bwv333_",
                  "bwv334_", "bwv336_", "bwv338_", "bwv340_", "bwv341_",
                  "bwv345_", "bwv346_", "bwv347_", "bwv348_", "bwv350_",
                  "bwv351_", "bwv36.4-2_", "bwv363_", "bwv364_", "bwv367_",
                  "bwv369_", "bwv37.6_", "bwv370_", "bwv371_", "bwv372_",
                  "bwv374_", "bwv375_", "bwv376_", "bwv377_", "bwv378_",
                  "bwv380_", "bwv382_", "bwv383_", "bwv384_", "bwv385_",
                  "bwv386_", "bwv387_", "bwv388_", "bwv389_", "bwv392_",
                  "bwv393_", "bwv394_", "bwv395_", "bwv396_", "bwv398_",
                  "bwv399_", "bwv4.8_", "bwv40.3_", "bwv402_",
                  "bwv406_", "bwv409_", "bwv410_", "bwv411_", "bwv412_",
                  "bwv414_", "bwv415_", "bwv417_", "bwv418_", "bwv419_",
                  "bwv42.7_", "bwv420_", "bwv421_", "bwv422_", "bwv425_",
                  "bwv426_", "bwv427_", "bwv428_", "bwv429_", "bwv430_",
                  "bwv431_", "bwv432_", "bwv434_", "bwv435_", "bwv436_",
                  "bwv438_", "bwv44.7_", "bwv45.7_", "bwv46.6_", "bwv47.5_",
                  "bwv48.3_", "bwv48.7_", "bwv6.6_", "bwv62.6_", "bwv64.2_",
                  "bwv64.4_", "bwv66.6_", "bwv67.7_", "bwv69.6-a", "bwv7.7_"
                  "bwv72.6_", "bwv7.7_", "bwv72.6_", "bwv73.5_", "bwv74.8_",
                  "bwv77.6_", "bwv80.8_", "bwv83.5_", "bwv84.5_", "bwv85.6_",
                  "bwv86.6_", "bwv88.7_", "bwv89.6_", "bwv9.7_", "bwv90.5_",
                  "bwv92.9_", "bwv93.7_", "bwv94.8_", "bwv96.6_", "bwv99.6_",
                  #begin pickup holdout
                  "bwv144.6_", "bwv257_", "bwv279_", "bwv287_", "bwv289_",
                  "bwv293_", "bwv304_", "bwv335_", "bwv362_", "bwv401_",
                  "bwv404_", "bwv416_", "bwv5.7_",
                  ]

    checks_3_4 = ["bwv122.6_", "bwv153.9_", "bwv187.7_", "bwv229.2_", "bwv248.42-s",
                  "bwv320_", "bwv321_", "bwv344_", "bwv349_", "bwv356_",
                  "bwv366_", "bwv391_", "bwv397_", "bwv400_", "bwv413_",
                  "bwv57.8_", "bwv70.7_", "bwv282_",
                  #begin holdout
                  "bwv342_",
                  # begin pickup train
                  "bwv11.6_", "bwv145.5_", "bwv194.12_", "bwv266_", "bwv269_",
                  "bwv295_", "bwv299_", "bwv306_", "bwv326_", "bwv327_",
                  "bwv343_", "bwv368_", "bwv390_", "bwv43.11", "bwv403_",
                  "bwv65.2_", "bwv67.4_",
                  # begin pickup holdout
                  "bwv17.7_",
                  ]

    checks_3_2 = ["bwv123.6_",
                  # begin holdout
                  # begin pickup train
                  # begin pickup holdout
                  ]
    checks_12_8 = [# begin holdout
                   # begin pickup train
                   "bwv248.23-s_",
                   #begin pickup holdout
                   ]

    for fpath in fpaths:
        with open(fpath, "r") as f:
            jinfo = json.load(f)
        notes_keys = ["Soprano", "Alto", "Tenor", "Bass"]
        meas_indices_per = [sorted(jinfo["notes"][k].keys(), key=lambda x: int(x)) for k in notes_keys]
        assert all([meas_indices_per[0] == mi for mi in meas_indices_per])

        # quantized at Qth interval
        # change this based on beat?
        n_meas = len(meas_indices_per[0])
        overall_arr = np.zeros((4, n_meas * Q, 100))
        overall_boundaries = np.zeros((4, n_meas * Q, 100))

        # check measure lengths
        measure_beat_length = [0 for _ in range(n_meas)]
        per_voice_beat_length = [[0] * len(notes_keys) for _ in range(n_meas)]
        per_voice_n_notes = [[0] * len(notes_keys) for _ in range(n_meas)]
        for _n in range(n_meas):
            # check that it is the right length
            # to avoid pickup measure issues
            guessed_beat_length = None
            this_finals = []
            this_notes = []
            for _i, k in enumerate(notes_keys):
                m_step = meas_indices_per[_i][_n]
                meas_notes = jinfo["notes"][k][m_step]
                if len(meas_notes) < 1:
                    continue
                final = meas_notes[-1][-2] + meas_notes[-1][-1]
                this_finals.append(final)
                this_notes.append(len(meas_notes))

            for _i, k in enumerate(notes_keys):
                per_voice_beat_length[_n][_i] = this_finals[_i]
                per_voice_n_notes[_n][_i] = this_notes[_i]

            if len(this_finals) > 0:
                finals_med = np.median(this_finals)
                if all([f == finals_med for f in this_finals]):
                    measure_beat_length[_n] = finals_med
                else:
                    print("diff len")
                    from IPython import embed; embed(); raise ValueError()
            else:
                continue
        beat_exists = [m for m in measure_beat_length if m > 0]
        beat_med = np.median(beat_exists)
        # several possible patterns
        # pickup then short end
        # pickup no short end
        # no pickup short end
        # check if it is only a pickup
        def beat_checksum(beat_exists_in):
            lb = 0
            rb = None
            if beat_exists_in[0] != beat_med:
                lb = 1
            if beat_exists_in[-1] != beat_med:
                rb = -1
            sub = beat_exists_in[lb:rb]
            step_ok = [False for _ in range(len(sub))]
            for step in range(len(sub)):
                if step != (len(sub) - 1):
                    if sub[step] == beat_med:
                        step_ok[step] = True
                    else:
                        # this step and next must sum to beat med
                        if (sub[step] + sub[step + 1]) == beat_med:
                            step_ok[step] = True
                            step_ok[step + 1] = True
            if sub[-1] == beat_med:
                step_ok[-1] = True
            if all(step_ok):
                return all(step_ok)
            else:
                # one more check for random in between fermata - if every voice has a single note in the measure it doesnt matter
                # if we hold for beat_med or another length
                combined = list(zip(per_voice_beat_length, per_voice_n_notes))
                falses = [(_nn, _v) for _nn, _v in enumerate(step_ok) if _v == False]
                for f_i in falses:
                    step_idx_to_comb = f_i[0] + lb
                    notes_in_fail = combined[step_idx_to_comb][1]
                    # only valid if all note counts are 1
                    if sum(notes_in_fail) != len(notes_in_fail):
                        return False
                return True

        if all([beat_med == m for m in beat_exists]):
            global_beat_length = beat_med
            beat_type = "straight"
        elif all([beat_med == m for m in beat_exists[1:]]):
            global_beat_length = beat_med
            beat_type = "pickup_only"
        elif all([beat_med == m for m in beat_exists[:-1]]):
            global_beat_length = beat_med
            beat_type = "fermata_only"
        elif all([beat_med == m for m in beat_exists[1:-1]]):
            global_beat_length = beat_med
            beat_type = "pickup_fermata"
        elif beat_checksum(beat_exists):
            global_beat_length = beat_med
            beat_type = "repeats_fermatas_edge_cases"
        else:
            # if we can't make sense of it, skip it (48 total, ~4 pieces out of the overall)
            skipped_files += 1
            continue

        # can't trust this numerator or denominator... why is it wrong?
        beat_num = jinfo["global_time_signature_numerator"]
        beat_denom = jinfo["global_time_signature_denominator"]

        ts_type = None
        if ts_type is None:
            for _4_4 in checks_4_4:
                if _4_4 in fpath:
                    ts_type = "4/4"
                    break

        if ts_type is None:
            for _3_4 in checks_3_4:
                if _3_4 in fpath:
                    ts_type = "3/4"
                    break

        if ts_type is None:
            for _3_2 in checks_3_2:
                if _3_2 in fpath:
                    ts_type = "3/2"
                    break

        if ts_type is None:
            for _12_8 in checks_12_8:
                if _12_8 in fpath:
                    ts_type = "12/8"
                    break

        if ts_type is None:
            print("didn't find time signature")
            from IPython import embed; embed(); raise ValueError()
        elif ts_type in ["12/8", "3/2", "3/4"]:
            # skipping 3/4 12/8 and 3/2 goes to 480 skipped (~40 pieces)
            # leaving 3672 remaining
            skipped_files += 1
            continue

        # construct the final array measure-wise
        # SHOULD WE SKIP PICKUPS? I VOTE YES, BIG YES
        measure_arrays = []
        measure_boundaries = []
        file_and_measure_attribution = []

        # add to skip meas when we merge 2 measures together for things like repeats
        skip_meas = []

        # a few files have 32nd notes, there is theoretically possible triplets
        cant_quantize = False
        for _n in range(n_meas):
            if cant_quantize:
                break
            # if it is the first measure and off beat (pickup) skip it!
            if _n == 0 and measure_beat_length[_n] != global_beat_length:
                continue

            # if we already marked this measure to skip, skip it
            if _n in skip_meas:
                continue

            if global_beat_length != 4.0:
                print("edge")
                from IPython import embed; embed(); raise ValueError()

            this_arr = np.zeros((4, Q, 100))
            this_boundaries = np.zeros((4, Q, 100))

            for _i, k in enumerate(notes_keys):
                m_step = meas_indices_per[_i][_n]
                meas_notes = jinfo["notes"][k][m_step]
                if measure_beat_length[_n] != global_beat_length:
                    if _n != (n_meas - 1):
                        m_step_next = meas_indices_per[_i][_n + 1]
                        next_meas_notes = jinfo["notes"][k][m_step_next]
                        this_meas_final = int(meas_notes[-1][-2] * global_beat_length) + int(meas_notes[-1][-1] * global_beat_length)
                        next_meas_final = int(next_meas_notes[-1][-2] * global_beat_length) + int(next_meas_notes[-1][-1] * global_beat_length)
                        if this_meas_final + next_meas_final != Q:
                            # if this measure and next don't line up BUT there is only one note in all voices for this measure
                            # it must be a fermata, just extend to full global_beat_length 
                            if sum(per_voice_n_notes[_n]) == len(per_voice_n_notes[_n]):
                                this_meas_final = int(meas_notes[-1][-2] * global_beat_length) + int(meas_notes[-1][-1] * global_beat_length)
                                additional_beats = (Q - this_meas_final) / float(global_beat_length)
                                meas_notes[-1][-1] += additional_beats
                            else:
                                print("measures to combine don't line up...")
                                from IPython import embed; embed(); raise ValueError()
                        else:
                            # make next meas notes
                            for _q in range(len(next_meas_notes)):
                                # offset them
                                next_meas_notes[_q][-2] += this_meas_final / global_beat_length
                            meas_notes = meas_notes + next_meas_notes
                            this_meas_final_final = int(meas_notes[-1][-2] * global_beat_length) + int(meas_notes[-1][-1] * global_beat_length)
                            assert this_meas_final_final == Q
                            # skip the measure since we corrected for it!
                            skip_meas.append(_n + 1)
                    else:
                        # the last measure is not the correct length, just hold it out (assume fermata)
                        this_meas_final = int(meas_notes[-1][-2] * global_beat_length) + int(meas_notes[-1][-1] * global_beat_length)
                        additional_beats = (Q - this_meas_final) / float(global_beat_length)
                        meas_notes[-1][-1] += additional_beats

                for note in meas_notes:
                    # edge case - we know it is 4/4 but got extra beat_med somehow?
                    # assume 16th quantization, and we filtered to always be in 4/4
                    try:
                        assert (note[-2] * global_beat_length) == int(note[-2] * global_beat_length)
                        assert (note[-1] * global_beat_length) == int(note[-1] * global_beat_length)
                    except:
                        cant_quantize = True
                        break
                    start = int(note[-2] * global_beat_length)
                    end = start + int(note[-1] * global_beat_length)
                    p = note[0]
                    this_arr[_i, start:end, p] = 1.
                    this_boundaries[_i, start, p] = 1.
                    this_boundaries[_i, end - 1, p] = -1.

                if cant_quantize:
                    break
            this_arr = this_arr.astype("int")
            this_boundaries = this_boundaries.astype("int")
            measure_arrays.append(this_arr)
            measure_boundaries.append(this_boundaries)
            file_and_measure_attribution.append((fpath, _n))

        if cant_quantize:
            skipped_files += 1
            continue

        overall_arr = np.concatenate(measure_arrays, axis=1)
        overall_boundaries = np.concatenate(measure_boundaries, axis=1)

        overall_arr = overall_arr.argmax(axis=-1)

        all_arr.append(overall_arr)
        all_boundary.append(overall_boundaries)
        all_attribution.append(file_and_measure_attribution)

        cut = 0
        while True:
            if cut < overall_arr.shape[1] - T:
                overall_arr_cut = overall_arr[:, cut:cut + T]
                overall_boundaries_cut = overall_boundaries[:, cut:cut + T]
                file_and_measure_cut = file_and_measure_attribution[cut:cut + T]
                cut += T
                # check that the chunk has no 0 voices
                was_blank = False
                for _i, k in enumerate(notes_keys):
                   line = overall_arr_cut[_i]
                   if any([l == 0 for l in line]):
                       was_blank = True

                if was_blank:
                    continue
                else:
                    all_arr_chunks.append(overall_arr_cut)
                    all_boundary_chunks.append(overall_boundaries_cut)
                    all_attribution_chunks.append(file_and_measure_cut)
            else:
                diff = (cut + T) - overall_arr.shape[1]
                if (diff % Q) == 0 and overall_arr.shape[1] >= T:
                    # always include ending cut
                    overall_arr_cut = overall_arr[:, -T:]
                    overall_boundaries_cut = overall_boundaries[:, -T:]
                    file_and_measure_cut = file_and_measure_attribution[-T:]
                    was_blank = False

                    for _i, k in enumerate(notes_keys):
                       line = overall_arr_cut[_i]
                       if any([l == 0 for l in line]):
                           was_blank = True

                    if not was_blank:
                        all_arr_chunks.append(overall_arr_cut)
                        all_boundary_chunks.append(overall_boundaries_cut)
                        all_attribution_chunks.append(file_and_measure_cut)
                break
        total_files += 1
    assert len(all_arr_chunks) == len(all_boundary_chunks)
    assert len(all_arr_chunks) == len(all_attribution_chunks)
    assert len(all_arr) == len(all_boundary)
    assert len(all_arr) == len(all_attribution)
    if _step == 0:
        train_data = all_arr_chunks
    else:
        test_data = all_arr_chunks

    if _step == 0:
        train_data_full = all_arr
        train_data_boundary_full = all_boundary
        train_data_attribution_full = all_attribution
    else:
        test_data_full = all_arr
        test_data_boundary_full = all_boundary
        test_data_attribution_full = all_attribution

fname = "Jsb16thSeparatedAligned.npz"
#source_files = all_attribution_chunks
#note_boundaries = all_boundary_chunks
np.savez(fname, train_data=train_data, test_data=test_data)#, source_files=source_files, note_boundaries=note_boundaries)

fname = "Jsb16thSeparatedAlignedFull.npz"
#source_files = all_attribution_chunks
#note_boundaries = all_boundary_chunks
# making arr like this prevents np.savez from converting list input
train_data_final = np.empty(len(train_data_full), object)
train_data_final[:] = train_data_full
train_data_full = train_data_final

train_data_boundary_final = np.empty(len(train_data_boundary_full), object)
train_data_boundary_final[:] = train_data_boundary_full
train_data_boundary_full = train_data_boundary_final

train_data_attribution_final = np.empty(len(train_data_attribution_full), object)
train_data_attribution_full = [t[0][0].split(os.sep)[-1] for t in train_data_attribution_full]
train_data_attribution_final[:] = train_data_attribution_full
train_data_attribution_full = train_data_attribution_final


test_data_final = np.empty(len(test_data_full), object)
test_data_final[:] = test_data_full
test_data_full = test_data_final

test_data_boundary_final = np.empty(len(test_data_boundary_full), object)
test_data_boundary_final[:] = test_data_boundary_full
test_data_boundary_full = test_data_boundary_final

test_data_attribution_final = np.empty(len(test_data_attribution_full), object)
test_data_attribution_full = [t[0][0].split(os.sep)[-1] for t in test_data_attribution_full]
test_data_attribution_final[:] = test_data_attribution_full
test_data_attribution_full = test_data_attribution_final

np.savez(fname, train_data_full=train_data_full,
                train_data_attribution_full=train_data_attribution_full,
                test_data_full=test_data_full,
                test_data_attribution_full=test_data_attribution_full)#, source_files=source_files, note_boundaries=note_boundaries)

print("done")
from IPython import embed; embed(); raise ValueError()
