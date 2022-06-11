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

fpaths = training_fpaths

all_final_measure_dur = {}
all_note_dur = {}
T = 144
# go for 144 so 3 and 4 will both be on beat
total_files = 0
all_arr_chunks = []
all_boundary_chunks = []
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
              "bwv399_", "bwv4.8_", "bwv40.3_", "bwv402_", "bwv403_",
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
              "bwv343_", "bwv368_", "bwv390_", "bwv43.11", "bwv65.2_",
              "bwv67.4_",
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

    # quantized at 16th interval
    # change this based on beat?
    n_meas = len(meas_indices_per[0])
    overall_arr = np.zeros((4, n_meas * 16, 100))
    overall_boundaries = np.zeros((4, n_meas * 16, 100))

    # check measure lengths
    measure_beat_length = [0 for _ in range(n_meas)]
    for _n in range(n_meas):
        # check that it is the right length
        # to avoid pickup measure issues
        guessed_beat_length = None
        this_finals = []
        for _i, k in enumerate(notes_keys):
            m_step = meas_indices_per[_i][_n]
            meas_notes = jinfo["notes"][k][m_step]
            if len(meas_notes) < 1:
                continue
            final = meas_notes[-1][-2] + meas_notes[-1][-1]
            this_finals.append(final)
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
    if all([beat_med == m for m in beat_exists]):
        global_beat_length = beat_med
    else:
        pass
        #print("uneven beat")
        #skipped_files += 1
        #continue
        #from IPython import embed; embed(); raise ValueError()

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
        print("didn't find")
        from IPython import embed; embed(); raise ValueError()
    else:
        continue
    # if beat length is anything other than 4 we need to manually handle...

    for _n in range(n_meas):
        for _i, k in enumerate(notes_keys):
            m_step = meas_indices_per[_i][_n]
            meas_notes = jinfo["notes"][k][m_step]
            for note in meas_notes:
                # assume 16th quantization
                start = int(_n * 4 * global_beat_length) + int(note[-2] * global_beat_length)
                end = start + int(note[-1] * global_beat_length)
                p = note[0]
                overall_arr[_i, start:end, p] = 1.
                overall_boundaries[_i, start, p] = 1.
                overall_boundaries[_i, end - 1, p] = -1.

    overall_arr = overall_arr.astype("bool")
    overall_boundaries = overall_boundaries.astype("int")

    cut = 0
    while cut < overall_arr.shape[1] - T:
        overall_arr_cut = overall_arr[:, cut:cut + T]
        overall_boundaries_cut = overall_boundaries[:, cut:cut + T]
        cut += T
        # check that the chunk has no 0 voices
        was_blank = False
        for _i, k in enumerate(notes_keys):
           line = overall_arr[_i].argmax(axis=1)
           if any([l == 0 for l in line]):
               was_blank = True

        if was_blank:
            continue
        all_arr_chunks.append(overall_arr_cut)
        all_boundary_chunks.append(overall_boundaries_cut)
    total_files += 1
print("done")
from IPython import embed; embed(); raise ValueError()
