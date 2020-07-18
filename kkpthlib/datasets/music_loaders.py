from __future__ import print_function
from music21 import corpus, interval, pitch
from music21.midi import MidiTrack, MidiFile, MidiEvent, DeltaTime
import os
import json
import time
import struct
import numpy as np
import itertools
from ..core import get_logger
from ..data import LookupDictionary
from .loaders import get_kkpthlib_dataset_dir
from .midi_instrument_map import midi_instruments_number_to_name
from .midi_instrument_map import midi_instruments_name_to_number
import collections

logger = get_logger()

def music21_parse_and_save_json(p, fpath, tempo_factor=1):
    piece_container = {}
    piece_container["parts"] = []
    piece_container["parts_times"] = []
    piece_container["parts_cumulative_times"] = []
    piece_container["parts_names"] = []
    # we check for multiple timings when loading files, usually
    spq = p.metronomeMarkBoundaries()[0][-1].secondsPerQuarter()
    qbpm = p.metronomeMarkBoundaries()[0][-1].getQuarterBPM()
    # set ppq to 220 to line up with magenta and pretty_midi
    ppq = 220
    #https://stackoverflow.com/questions/2038313/converting-midi-ticks-to-actual-playback-seconds
    piece_container["seconds_per_quarter"] = spq
    piece_container["quarter_beats_per_minute"] = qbpm
    piece_container["pulses_per_quarter"] = ppq
    for i, pi in enumerate(p.parts):
        piece_container["parts"].append([])
        piece_container["parts_times"].append([])
        piece_container["parts_cumulative_times"].append([])
        piece_container["parts_names"].append(pi.id)
        part = []
        part_time = []
        for n in pi.flat.notesAndRests:
            if n.isRest:
                part.append(0)
            else:
                part.append(n.midi)
            part_time.append(n.duration.quarterLength * tempo_factor)
        piece_container["parts"][i] += part
        piece_container["parts_times"][i] += part_time
        piece_container["parts_cumulative_times"][i] += list(np.cumsum(part_time))
    j = json.dumps(piece_container, indent=4)
    with open(fpath, "w") as f:
         print(j, file=f)


def check_fetch_jsb_chorales(only_pieces_with_n_voices=[4], verbose=True):
    if os.path.exists(get_kkpthlib_dataset_dir() + os.sep + "jsb_chorales_json"):
        dataset_path = get_kkpthlib_dataset_dir("jsb_chorales_json")
        # if the dataset already exists, assume the preprocessing is already complete
        return dataset_path
    dataset_path = get_kkpthlib_dataset_dir("jsb_chorales_json")

    all_bach_paths = corpus.getComposer('bach')

    if verbose:
        logger.info("JSB Chorales not yet cached, processing...")
        logger.info("Total number of Bach pieces to process from music21: {}".format(len(all_bach_paths)))
    for it, p_bach in enumerate(all_bach_paths):
        if "riemenschneider" in p_bach:
            # skip certain files we don't care about
            continue
        p = corpus.parse(p_bach)
        if len(p.parts) not in only_pieces_with_n_voices:
            if verbose:
                logger.info("Skipping file {}, {} due to undesired voice count...".format(it, p_bach))
            continue

        if len(p.metronomeMarkBoundaries()) != 1:
            if verbose:
                logger.info("Skipping file {}, {} due to unknown or multiple tempo changes...".format(it, p_bach))
            continue

        if verbose:
            logger.info("Processing {}, {} ...".format(it, p_bach))

        k = p.analyze('key')
        if verbose:
            logger.info("Original key: {}".format(k))
        stripped_extension_name = ".".join(os.path.split(p_bach)[1].split(".")[:-1])
        base_fpath = dataset_path + os.sep + stripped_extension_name
        try:
            if os.path.exists(base_fpath + ".json"):
                if verbose:
                    logger.info("File exists {}, skipping...".format(base_fpath))
            else:
                if 'major' in k.name:
                    kt = "major"
                elif 'minor' in k.name:
                    kt = "minor"
                core_name = base_fpath + ".{}-{}-original.json".format(k.name.split(" ")[0], kt)
                music21_parse_and_save_json(p, core_name, tempo_factor=1)
                logger.info("Writing {}".format(core_name))
            for t in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
                if 'major' in k.name:
                    kt = "major"
                elif 'minor' in k.name:
                    kt = "minor"
                else:
                    raise AttributeError('Unknown key {}'.format(kn.name))

                transpose_fpath = base_fpath + ".{}-{}-transposed.json".format(t, kt)
                if os.path.exists(transpose_fpath):
                    if verbose:
                        logger.info("File exists {}, skipping...".format(transpose_fpath))
                    continue

                i = interval.Interval(k.tonic, pitch.Pitch(t))
                pn = p.transpose(i)
                #kn = pn.analyze('key')
                music21_parse_and_save_json(pn, transpose_fpath, tempo_factor=1)
                if verbose:
                    logger.info("Writing {}".format(transpose_fpath))
        except Exception as e:
            if verbose:
                logger.info(e)
                logger.info("Skipping {} due to unknown error".format(p_bach))
            continue
    return dataset_path


def _populate_track_from_data(data, index, program_changes=None):
    """
    example program change
    https://github.com/cuthbertLab/music21/blob/a78617291ed0aeb6595c71f82c5d398ebe604ef4/music21/midi/__init__.py

    instrument_sequence = [(instrument, ppq_adjusted_time)]
    """
    mt = MidiTrack(index)
    t = 0
    tlast = 0
    pc_counter = 0
    for d, p, v in data:
        # need these "blank" time events
        # between each real event
        # to parse correctly
        dt = DeltaTime(mt)
        dt.time = 0 #t - tLast
        dt.channel = index
        # add to track events
        mt.events.append(dt)

        if program_changes is not None:
            if pc_counter >= len(program_changes):
                pass
            elif t >= program_changes[pc_counter][1] and tlast <= program_changes[pc_counter][1]:
                # crossed a program change event
                pc = MidiEvent(mt)
                if program_changes[pc_counter][0] not in midi_instruments_name_to_number.keys():
                    raise ValueError("Passed program change name {} not found in kkpthlib/datasets/midi_instrument_map.py".format(program_changes[pc_counter][0]))
                inst_num = midi_instruments_name_to_number[program_changes[pc_counter][0]]
                # convert from 1 indexed ala pretty-midi to 0 indexed ala music21...
                inst_num = inst_num - 1
                pc.type = "PROGRAM_CHANGE"
                pc.channel = index
                pc.time = None
                pc.data = inst_num
                mt.events.append(pc)

                # need these "blank" time events
                # between each real event
                # to parse correctly
                dt = DeltaTime(mt)
                dt.time = 0 #t - tLast
                # add to track events
                mt.events.append(dt)

                pc_counter += 1

        me = MidiEvent(mt)
        me.type = "NOTE_ON"
        me.channel = index
        me.time = None #d
        me.pitch = p
        me.velocity = v
        mt.events.append(me)

        # add note off / velocity zero message
        dt = DeltaTime(mt)
        dt.time = d
        dt.channel = index
        # add to track events
        mt.events.append(dt)

        me = MidiEvent(mt)
        me.type = "NOTE_ON"
        me.channel = index 
        me.time = None #d
        me.pitch = p
        me.velocity = 0
        mt.events.append(me)

        tlast = t
        t += d

    # add end of track
    dt = DeltaTime(mt)
    dt.time = 0
    dt.channel = index
    mt.events.append(dt)

    me = MidiEvent(mt)
    me.type = "END_OF_TRACK"
    me.channel = index 
    me.data = '' # must set data to empty string
    mt.events.append(me)
    return mt


def write_music_json(json_data, out_name, default_velocity=120):
    """
    assume data is formatted in "music JSON" format
    """
    data = json.loads(json_data)
    ppq = data["pulses_per_quarter"]
    qbpm = data["quarter_beats_per_minute"]
    spq = data["seconds_per_quarter"]

    parts = data["parts"]
    parts_times = data["parts_times"]
    # https://github.com/cuthbertLab/music21/blob/c6fc39204c16c47d1c540b545d0c9869a9cafa8f/music21/midi/__init__.py#L1471
    if "parts_velocities" not in data:
        # handle rests
        parts_velocities = [[default_velocity if pi != 0 else 0 for pi in p] for p in parts]
    else:
        print("handle velocities in write_music_json")
        from IPython import embed; embed(); raise ValueError()

    with open(out_name, "w") as f:
         json.dump(data, f, indent=4)

_program_presets = {
                    "dreamy_r_preset": [("Sitar", 30),
                                        ("Orchestral Harp", 40),
                                        ("Acoustic Guitar (nylon)", 40),
                                        ("Pan Flute", 20)],
                    "dreamy_preset": [("Pan Flute", 20),
                                      ("Acoustic Guitar (nylon)", 40),
                                      ("Orchestral Harp", 40),
                                      ("Sitar", 30)],
                    "zelda_preset": [("Pan Flute", 10),
                                     ("Acoustic Guitar (nylon)", 25),
                                     ("Acoustic Guitar (nylon)", 16),
                                     ("Acoustic Guitar (nylon)", 20)],
                    "nylon_preset": [("Acoustic Guitar (nylon)", 20),
                                     ("Acoustic Guitar (nylon)", 25),
                                     ("Acoustic Guitar (nylon)", 16),
                                     ("Acoustic Guitar (nylon)", 20)],
                    "organ_preset": [("Church Organ", 50),
                                     ("Church Organ", 30),
                                     ("Church Organ", 30),
                                     ("Church Organ", 40)],
                    "grand_piano_preset": [("Acoustic Grand Piano", 50),
                                           ("Acoustic Grand Piano", 30),
                                           ("Acoustic Grand Piano", 30),
                                           ("Acoustic Grand Piano", 40)],
                    "electric_piano_preset": [("Electric Piano 1", 50),
                                              ("Electric Piano 1", 30),
                                              ("Electric Piano 1", 30),
                                              ("Electric Piano 1", 40)],
                    "harpsichord_preset": [("Harpsichord", 50),
                                           ("Harpsichord", 30),
                                           ("Harpsichord", 30),
                                           ("Harpsichord", 40)],
                    "woodwind_preset": [("Oboe", 50),
                                        ("English Horn", 30),
                                        ("Clarinet", 30),
                                        ("Bassoon", 40)],
                   }

def music_json_to_midi(json_file, out_name, tempo_factor=.5,
                       default_velocity=120,
                       voice_program_map=None):
    """
    string (filepath) or json.dumps object

    tempo factor .5, twice as slow
    tempo factor 2, twice as fast

    voice_program_map {0: [(instrument_name, time_in_quarters)],
                       1: [(instrument_name, time_in_quarters)]}
    voices ordered SATB by default

    instrument names for program changes defined in kkpthlib/datasets/midi_instrument_map.py

    An example program change, doing harpsichord the first 8 quarter notes then a special
    mix as used by Music Transformer, Huang et. al.

    and recommended by
    https://musescore.org/en/node/109121

    a = "Harpsichord"
    b = "Harpsichord"
    c = "Harpsichord"
    d = "Harpsichord"

    e = "Oboe"
    f = "English Horn"
    g = "Clarinet"
    h = "Bassoon"

    # key: voice
    # values: list of tuples (instrument, time_in_quarter_notes_to_start_using) - optionally (instrument, time_in_quarters, global_amplitude)
    # amplitude should be in [0 , 127]
    m = {0: [(a, 0), (e, 8)],
         1: [(b, 0), (f, 8)],
         2: [(c, 0), (g, 8)],
         3: [(d, 0), (h, 8)]}

    or

    m = {0: [(a, 0, 60), (e, 8, 40)],
         1: [(b, 0, 30), (f, 8, 30)],
         2: [(c, 0, 30), (g, 8, 30)],
         3: [(d, 0, 40), (h, 8, 50)]}

    Alternatively, support "auto" groups which set custom voices and amplitudes
    a = "harpsichord_preset"
    b = "woodwind_preset"
    m = {0: [(a, 0), (b, 8)],
         1: [(a, 0), (b, 8)],
         2: [(a, 0), (b, 8)],
         3: [(a, 0), (b, 8)]}

    valid preset values:
                "dreamy_r_preset"
                "dreamy_preset"
                "zelda_preset"
                "nylon_preset"
                "organ_preset"
                "grand_piano_preset"
                "electric_piano_preset"
                "harpsichord_preset"
                "woodwind_preset"
    """
    if json_file.endswith(".json"):
        with open(json_file) as f:
            data = json.load(f)
    else:
        data = json.loads(json_file)
    #[u'parts', u'parts_names', u'parts_cumulative_times', u'parts_times']
    #['parts_velocities'] optional

    ppq = data["pulses_per_quarter"]
    qbpm = data["quarter_beats_per_minute"]
    spq = data["seconds_per_quarter"]

    parts = data["parts"]
    parts_times = data["parts_times"]
    # https://github.com/cuthbertLab/music21/blob/c6fc39204c16c47d1c540b545d0c9869a9cafa8f/music21/midi/__init__.py#L1471
    if "parts_velocities" not in data:
        # handle rests
        parts_velocities = [[default_velocity if pi != 0 else 0 for pi in p] for p in parts]
    else:
        print("handle velocities in json_to_midi")

    all_mt = []
    for i in range(len(parts)):
        assert len(parts[i]) == len(parts_velocities[i])
        assert len(parts[i]) == len(parts_times[i])
        program_changes = voice_program_map[i] if voice_program_map is not None else None
        this_part_velocity = [parts_velocities[i][j] for j in range(len(parts[i]))]
        if program_changes is not None:
            # remap presets
            program_changes_new = []
            for pc in program_changes:
                if pc[0] in _program_presets.keys():
                    p = _program_presets[pc[0]]
                    program_changes_new.append((p[i][0], pc[1], p[i][1]))
                else:
                    program_changes_new.append(pc)
            program_changes = program_changes_new

        if program_changes is not None:
            program_changes = [(pg[0], int(pg[1] * ppq)) if len(pg) < 3 else (pg[0], int(pg[1] * ppq), pg[2]) for pg in program_changes]
            this_part_velocity_new = []
            pg_counter = 0
            last_step_tick = 0
            current_step_tick = 0
            current_velocity = default_velocity
            for j in range(len(parts[i])):
                last_step_tick = current_step_tick
                current_step_tick += int(parts_times[i][j] * ppq)
                if len(program_changes[pg_counter]) < 3:
                    this_part_velocity_new.append(this_part_velocity[j])
                else:
                    # if it is the last program change then we just stay on that
                    if pg_counter < len(program_changes) - 1:
                        # check for tick boundary
                        # if we are exactly on a boundary... change it now
                        if last_step_tick <= int(program_changes[pg_counter + 1][1]) and current_step_tick >= int(program_changes[pg_counter + 1][1]):
                            pg_counter += 1
                    current_velocity = default_velocity if len(program_changes[pg_counter]) < 3 else program_changes[pg_counter][2]
                    this_part_velocity_new.append(current_velocity)
            this_part_velocity = this_part_velocity_new

        track_data = [[int(parts_times[i][j] * ppq), parts[i][j], this_part_velocity[j] if parts[i][j] != 0 else 0] for j in range(len(parts[i]))]

        # do global velocity modulations...
        # + 1 to account for MidiTrack starting at 1
        mt = _populate_track_from_data(track_data, i + 1, program_changes=program_changes)
        all_mt.append(mt)

    mf = MidiFile()
    # multiply by half to get proper feel on bach at least, not sure why...
    # see for example bwv110.7
    # https://www.youtube.com/watch?v=1WWR4PQZdjo
    mf.ticksPerQuarterNote = int(ppq * tempo_factor)
    # ticks (pulses) / quarter * quarters / second 
    mf.ticksPerSecond = int(ppq * (1. / float(spq)))

    for mt in all_mt:
        mf.tracks.append(mt)

    mf.open(out_name, 'wb')
    mf.write()
    mf.close()


def piano_roll_from_music_json_file(json_file, default_velocity=120, quantization_rate=.25, n_voices=4,
                                    separate_onsets=True, onsets_boundary=100, as_numpy=True):
    """
    return list of list [[each_voice] n_voices] or numpy array of shape (time_len, n_voices)
    """
    with open(json_file) as f:
        data = json.load(f)
    ppq = data["pulses_per_quarter"]
    qbpm = data["quarter_beats_per_minute"]
    spq = data["seconds_per_quarter"]

    parts = data["parts"]
    parts_times = data["parts_times"]
    parts_cumulative_times = data["parts_cumulative_times"]
    # https://github.com/cuthbertLab/music21/blob/c6fc39204c16c47d1c540b545d0c9869a9cafa8f/music21/midi/__init__.py#L1471
    if "parts_velocities" not in data:
        default_velocity = default_velocity
        parts_velocities = [[default_velocity] * len(p) for p in parts]
    else:
        parts_velocities = data["parts_velocities"]
    end_in_quarters = max([max(p) for p in parts_cumulative_times])
    # clock is set currently by the fact that "sixteenth" is the only option
    # .25 due to "sixteenth"
    clock = np.arange(0, max(max(parts_cumulative_times)), quantization_rate)
    # 4 * for 4 voices
    raster_end_in_steps = n_voices * len(clock)

    roll_voices = [[] for _ in range(n_voices)]
    # use these for tracking if we cross a change event
    p_i = [0] * n_voices
    for c in clock:
        # voice
        for v in range(len(parts)):
            current_note = parts[v][p_i[v]]
            next_change_time = parts_cumulative_times[v][p_i[v]]
            new_onset = False
            if c >= next_change_time:
                # we hit a boundary, swap notes
                p_i[v] += 1
                current_note = parts[v][p_i[v]]
                next_change_time = parts_cumulative_times[v][p_i[v]]
                new_onset = True
            if c == 0. or new_onset:
                if current_note != 0:
                    if separate_onsets:
                        roll_voices[v].append(current_note + onsets_boundary)
                    else:
                        roll_voices[v].append(current_note)
                else:
                    # rests have no "onset"
                    roll_voices[v].append(current_note)
            else:
               roll_voices[v].append(current_note)
    if as_numpy:
        roll_voices = np.array(roll_voices).T
    return roll_voices


def pitch_duration_velocity_lists_from_music_json_file(json_file, default_velocity=120, n_voices=4,
                                                       add_measure_values=True,
                                                       measure_value=99,
                                                       measure_quarters=4,
                                                       fill_value=-1):
    """
    return list of list [[each_voice] n_voices] or numpy array of shape (time_len, n_voices)
    """
    with open(json_file) as f:
        data = json.load(f)
    ppq = data["pulses_per_quarter"]
    qbpm = data["quarter_beats_per_minute"]
    spq = data["seconds_per_quarter"]

    parts = data["parts"]
    parts_times = data["parts_times"]
    parts_cumulative_times = data["parts_cumulative_times"]
    # https://github.com/cuthbertLab/music21/blob/c6fc39204c16c47d1c540b545d0c9869a9cafa8f/music21/midi/__init__.py#L1471
    if "parts_velocities" not in data:
        default_velocity = default_velocity
        parts_velocities = [[default_velocity] * len(p) for p in parts]
    else:
        parts_velocities = data["parts_velocities"]

    n_steps = max([len(p) for p in parts])

    measure_stops = [[] for p in parts]
    for v in range(len(parts)):
        for s in range(n_steps):
            if s >= len(parts[v]):
                continue
            p_s = parts[v][s]
            d_s = parts_times[v][s]
            p_c_s = parts_cumulative_times[v][s]
            p_v_s = parts_velocities[v][s]

            #new_parts[v].append(p_s)
            #new_parts_times[v].append(d_s)
            #new_parts_cumulative_times[v].append(p_c_s)
            #new_parts_velocities[v].append(p_v_s)
            if p_c_s % measure_quarters == 0:
                measure_stops[v].append((s + 1, p_c_s))

    # the set of stop points consistent within all voices 

    count = collections.Counter([m_i_s[1] for m_i in measure_stops for m_i_s in m_i])
    shared_stop_points = sorted([k for k in count.keys() if count[k] == len(parts)])
    if len(shared_stop_points) < 1:
        raise ValueError("No points where all voices start on the measure start...?")

    cleaned_measure_stops = []
    for v in range(len(parts)):
        r = [mi for mi in measure_stops[v] if mi[1] in shared_stop_points]
        cleaned_measure_stops.append(r)


    new_parts = [[] for p in parts]
    new_parts_times = [[] for p in parts]
    new_parts_cumulative_times = [[] for p in parts]
    new_parts_velocities = [[] for p in parts]
    last_vs = [0 for v in range(len(parts))]
    for v in range(len(parts)):
        cm = cleaned_measure_stops[v]
        for cmi in cm:
            last = last_vs[v]

            new_parts[v].extend(parts[v][last:cmi[0]])
            new_parts_times[v].extend(parts_times[v][last:cmi[0]])
            new_parts_cumulative_times[v].extend(parts_cumulative_times[v][last:cmi[0]])
            new_parts_velocities[v].extend(parts_velocities[v][last:cmi[0]])

            if add_measure_values:
                new_parts[v].append(measure_value)
                new_parts_times[v].append(fill_value)
                new_parts_cumulative_times[v].append(fill_value)
                new_parts_velocities[v].append(fill_value)

            last_vs[v] = cmi[0]
    return new_parts, new_parts_times, new_parts_velocities


class MusicJSONRasterCorpus(object):
    def __init__(self, train_data_file_paths, valid_data_file_paths=None, test_data_file_paths=None,
                 max_vocabulary_size=-1,
                 add_eos=False,
                 tokenization_fn="flatten",
                 default_velocity=120, quantization_rate=.25, n_voices=4,
                 separate_onsets=True, onsets_boundary=100):
        """
        """
        self.dictionary = LookupDictionary()

        self.max_vocabulary_size = max_vocabulary_size
        self.default_velocity = default_velocity
        self.quantization_rate = quantization_rate
        self.n_voices = n_voices
        self.separate_onsets = separate_onsets
        self.onsets_boundary = onsets_boundary
        self.add_eos = add_eos

        if tokenization_fn == "flatten":
            def tk(arr):
                t = [el for el in arr.ravel()]
                if add_eos:
                    # 2 measures of silence are the eos
                    return t + [0] * 32
                else:
                    return t
            self.tokenization_fn = tk
        else:
            raise ValueError("Unknown tokenization_fn {}".format(tokenization_fn))

        base = [fp for fp in train_data_file_paths]
        if valid_data_file_paths is not None:
            base = base + [fp for fp in valid_data_file_paths]
        if test_data_file_paths is not None:
            base = base + [fp for fp in test_data_file_paths]

        self.build_vocabulary(base)

        if self.max_vocabulary_size > -1:
            self.dictionary._prune_to_top_k_counts(self.max_vocabulary_size)

        self.train = self.tokenize(train_data_file_paths)
        if valid_data_file_paths is not None:
            self.valid = self.tokenize(valid_data_file_paths)
        if test_data_file_paths is not None:
            self.test = self.tokenize(test_data_file_paths)

    def build_vocabulary(self, json_file_paths):
        """Tokenizes a text file."""
        for path in json_file_paths:
            assert os.path.exists(path)
            roll = piano_roll_from_music_json_file(path,
                                                   default_velocity=self.default_velocity,
                                                   quantization_rate=self.quantization_rate,
                                                   n_voices=self.n_voices,
                                                   separate_onsets=self.separate_onsets,
                                                   onsets_boundary=self.onsets_boundary,
                                                   as_numpy=True)
            words = self.tokenization_fn(roll)

            for word in words:
                self.dictionary.add_word(word)

    def tokenize(self, paths):
        """Tokenizes a text file."""
        ids = []
        for path in paths:
            assert os.path.exists(path)
            # Add words to the dictionary
            roll = piano_roll_from_music_json_file(path,
                                                   default_velocity=self.default_velocity,
                                                   quantization_rate=self.quantization_rate,
                                                   n_voices=self.n_voices,
                                                   separate_onsets=self.separate_onsets,
                                                   onsets_boundary=self.onsets_boundary,
                                                   as_numpy=True)
            words = self.tokenization_fn(roll)
            for word in words:
                if word in self.dictionary.word2idx:
                    token = self.dictionary.word2idx[word]
                else:
                    token = self.dictionary.word2idx["<unk>"]
                ids.append(token)
        return ids


class MusicJSONFlatKeyframeMeasureCorpus(object):
    def __init__(self, train_data_file_paths, valid_data_file_paths=None, test_data_file_paths=None,
                 max_vocabulary_size=-1,
                 add_measure_marks=True,
                 tokenization_fn="measure_flatten",
                 default_velocity=120, n_voices=4,
                 measure_value=99,
                 transition_value=9999,
                 fill_value=-1,
                 separate_onsets=True, onsets_boundary=100):
        """
        """
        self._make_dictionaries()

        self.max_vocabulary_size = max_vocabulary_size
        self.default_velocity = default_velocity
        self.n_voices = n_voices
        self.measure_value = measure_value
        self.fill_value = fill_value
        self.separate_onsets = separate_onsets
        self.onsets_boundary = onsets_boundary
        self.add_measure_marks = add_measure_marks
        self.transition_value = transition_value

        if tokenization_fn != "measure_flatten":
            raise ValueError("Only default tokenization_fn currently supported")

        base = [fp for fp in train_data_file_paths]
        if valid_data_file_paths is not None:
            base = base + [fp for fp in valid_data_file_paths]
        if test_data_file_paths is not None:
            base = base + [fp for fp in test_data_file_paths]

        self.build_vocabulary(base)

        self.train = self.tokenize(train_data_file_paths)
        if valid_data_file_paths is not None:
            self.valid = self.tokenize(valid_data_file_paths)
        if test_data_file_paths is not None:
            self.test = self.tokenize(test_data_file_paths)

    def _make_dictionaries(self):
        self.fingerprint_features_zero_dictionary = LookupDictionary()
        self.fingerprint_features_one_dictionary = LookupDictionary()
        self.duration_features_zero_dictionary = LookupDictionary()
        self.duration_features_mid_dictionary = LookupDictionary()
        self.duration_features_one_dictionary = LookupDictionary()
        self.voices_dictionary = LookupDictionary()

        self.centers_0_dictionary = LookupDictionary()
        self.centers_1_dictionary = LookupDictionary()
        self.centers_2_dictionary = LookupDictionary()
        self.centers_3_dictionary = LookupDictionary()

        self.keypoint_dictionary = LookupDictionary()
        self.keypoint_durations_dictionary = LookupDictionary()

        self.target_0_dictionary = LookupDictionary()
        self.target_1_dictionary = LookupDictionary()
        self.target_2_dictionary = LookupDictionary()
        self.target_3_dictionary = LookupDictionary()

    def _process(self, path):
        assert os.path.exists(path)
        pitch, duration, velocity = pitch_duration_velocity_lists_from_music_json_file(path,
                                                                                       default_velocity=self.default_velocity,
                                                                                       n_voices=self.n_voices,
                                                                                       measure_value=self.measure_value,
                                                                                       fill_value=self.fill_value)

        def isplit(iterable, splitters):
            return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]

        group = []
        for v in range(len(pitch)):
            # per voice, do split and merge
            s_p = isplit(pitch[v], [self.measure_value])
            s_d = isplit(duration[v], [self.fill_value])
            s_v = isplit(velocity[v], [self.fill_value])
            group.append([s_p, s_d, s_v])

        not_merged = True
        # all should be the same length in terms of measures so we can merge
        try:
            assert len(group[0][0]) == len(group[1][0])
            for g in group:
                assert len(group[0][0]) == len(g[0])
        except:
            raise ValueError("Group check assertion failed in _process of MusicJSONFlatMeasureCorpus")

        # just checked that all have the same number of measures, so now we combine them
        flat_key_zero = []
        flat_key_one = []
        flat_rel_encoded_pitch = []
        flat_lines_encoded = []

        flat_fingerprint_features_zero = []
        flat_fingerprint_features_one = []
        flat_duration_features_zero = []
        flat_duration_features_mid = []
        flat_duration_features_one = []
        flat_voices = []
        flat_centers = []
        flat_targets = []
        flat_key_zero = []
        flat_key_durations_zero = []
        flat_key_one = []
        flat_key_durations_one = []
        flat_key_indicators = []

        # key frame encoding
        bottom_voice = len(group) - 1
        key_counter = 0
        for i in range(len(group[0][0]) - 1):
            lines_offset_from_frame = []
            centers = []

            key_zero = []
            key_durations_zero = []
            key_one = []
            key_durations_one = []

            for v in range(len(group)):
            # for now only top voice
                curr_pitch_set = group[v][0][i]
                curr_dur_set = group[v][1][i]
                next_pitch_set = group[v][0][i + 1]
                next_dur_set = group[v][1][i + 1]
                if curr_pitch_set[0] == 0:
                    # if currently resting, just use next
                    center = next_pitch_set[0]
                elif next_pitch_set[0] == 0:
                    # if next is rest, just use current
                    center = curr_pitch_set[0]
                elif curr_pitch_set[0] == 0 and next_pitch_set[0] == 0:
                    # both resting doesn't occur in jsb... for now
                    print("both rest")
                    from IPython import embed; embed(); raise ValueError()
                    center = (next_pitch_set[0] + curr_pitch_set[0]) / 2.
                else:
                    # if both active, get the centerline between the two
                    center = (next_pitch_set[0] + curr_pitch_set[0]) / 2.

                line_offset_from_frame = center - group[bottom_voice][0][i][0]

                key_zero.append(curr_pitch_set[0])
                key_durations_zero.append(curr_dur_set[0])
                key_one.append(next_pitch_set[0])
                key_durations_one.append(next_dur_set[0])

                lines_offset_from_frame.append(line_offset_from_frame)
                centers.append(center)


            for v in range(len(group)):
                curr_pitch_set = group[v][0][i]
                next_pitch_set = group[v][0][i + 1]
                curr_dur_set = group[v][1][i]
                cumulative_curr_dur_set = np.cumsum(curr_dur_set)

                fingerprint_features_zero = []
                fingerprint_features_one = []
                duration_features_zero = []
                duration_features_mid = []
                duration_features_one = []
                voices = []
                these_centers = []

                # create "fingerprint" features against left and right keyframes
                # create duration offset features
                # annotate with voices
                for l in range(1, len(curr_pitch_set)):
                    print_set_zero = [curr_pitch_set[l] - group[vv][0][i][0] for vv in range(len(group))]
                    print_set_one = [curr_pitch_set[l] - group[vv][0][i + 1][0] for vv in range(len(group))]

                    fingerprint_features_zero.append(print_set_zero)
                    fingerprint_features_one.append(print_set_one)

                    duration_features_zero.append(cumulative_curr_dur_set[l - 1])
                    duration_features_one.append(cumulative_curr_dur_set[-1] - cumulative_curr_dur_set[l - 1])
                    duration_features_mid.append(curr_dur_set[l])
                    voices.append(v)
                    these_centers.append(centers)

                flat_voices.extend(voices)
                flat_duration_features_zero.extend(duration_features_zero)
                flat_duration_features_mid.extend(duration_features_mid)
                flat_duration_features_one.extend(duration_features_one)
                flat_fingerprint_features_zero.extend(fingerprint_features_zero)
                flat_fingerprint_features_one.extend(fingerprint_features_one)
                flat_centers.extend(these_centers)

                # create prediction targets - distance from each "center" line
                # assign unique code for rest
                rel_encoded = [[csi - c if csi != 0 else 100 for c in centers] for csi in curr_pitch_set[1:]]
                flat_targets.extend(rel_encoded)

                flat_key_zero.extend([key_zero for l in range(1, len(curr_pitch_set))])
                flat_key_durations_zero.extend([key_durations_zero for l in range(1, len(curr_pitch_set))])
                flat_key_one.extend([key_one for l in range(1, len(curr_pitch_set))])
                flat_key_durations_one.extend([key_durations_one for l in range(1, len(curr_pitch_set))])
                flat_key_indicators.extend([key_counter for l in range(1, len(curr_pitch_set))])

                if v == 0:
                    # add valid cut point marker at the start
                    # so insert(0)
                    tmp = [flat_fingerprint_features_zero,
                           flat_fingerprint_features_one,
                           flat_duration_features_zero,
                           flat_duration_features_mid,
                           flat_duration_features_one,
                           flat_voices,
                           flat_centers,
                           flat_key_zero,
                           flat_key_durations_zero,
                           flat_key_one,
                           flat_key_durations_one,
                           flat_key_indicators,
                           flat_targets]

                    for z, r in enumerate(tmp):
                        fill = self.transition_value
                        if hasattr(tmp[z][-1], "__len__"):
                            tmp[z].insert(0, [fill for el in tmp[z][-1]])
                        else:
                            tmp[z].insert(0, fill)

                # add valid cut point marker at the boundary
                tmp = [flat_fingerprint_features_zero,
                       flat_fingerprint_features_one,
                       flat_duration_features_zero,
                       flat_duration_features_mid,
                       flat_duration_features_one,
                       flat_voices,
                       flat_centers,
                       flat_key_zero,
                       flat_key_durations_zero,
                       flat_key_one,
                       flat_key_durations_one,
                       flat_key_indicators,
                       flat_targets]

                for z, r in enumerate(tmp):
                    fill = self.transition_value
                    if hasattr(tmp[z][-1], "__len__"):
                        tmp[z].append([fill for el in tmp[z][-1]])
                    else:
                        tmp[z].append(fill)
            key_counter += 1

        ret = [flat_fingerprint_features_zero,
               flat_fingerprint_features_one,
               flat_duration_features_zero,
               flat_duration_features_mid,
               flat_duration_features_one,
               flat_voices,
               flat_centers,
               flat_key_zero,
               flat_key_durations_zero,
               flat_key_one,
               flat_key_durations_one,
               flat_key_indicators,
               flat_targets]
        return ret


    def build_vocabulary(self, json_file_paths):
        """Tokenizes a text file."""
        for path in json_file_paths:
            ret = self._process(path)
            (flat_fingerprint_features_zero,
             flat_fingerprint_features_one,
             flat_duration_features_zero,
             flat_duration_features_mid,
             flat_duration_features_one,
             flat_voices,
             flat_centers,
             flat_key_zero,
             flat_key_durations_zero,
             flat_key_one,
             flat_key_durations_one,
             flat_key_indicators,
             flat_targets) = ret

            for fpz in flat_fingerprint_features_zero:
                self.fingerprint_features_zero_dictionary.add_word(tuple(fpz))

            for fpo in flat_fingerprint_features_one:
                self.fingerprint_features_one_dictionary.add_word(tuple(fpo))

            for durfz in flat_duration_features_zero:
                self.duration_features_zero_dictionary.add_word(durfz)

            for durfm in flat_duration_features_mid:
                self.duration_features_mid_dictionary.add_word(durfm)

            for durfo in flat_duration_features_one:
                self.duration_features_one_dictionary.add_word(durfo)

            for v in flat_voices:
                self.voices_dictionary.add_word(v)

            for c in flat_centers:
                self.centers_0_dictionary.add_word(c[0])
                self.centers_1_dictionary.add_word(c[1])
                self.centers_2_dictionary.add_word(c[2])
                self.centers_3_dictionary.add_word(c[3])

            for kz in flat_key_zero:
                self.keypoint_dictionary.add_word(tuple(kz))

            for kdz in flat_key_durations_zero:
                self.keypoint_durations_dictionary.add_word(tuple(kdz))

            for ko in flat_key_one:
                self.keypoint_dictionary.add_word(tuple(ko))

            for kdo in flat_key_durations_one:
                self.keypoint_durations_dictionary.add_word(tuple(kdo))

            for t in flat_targets:
                self.target_0_dictionary.add_word(t[0])
                self.target_1_dictionary.add_word(t[1])
                self.target_2_dictionary.add_word(t[2])
                self.target_3_dictionary.add_word(t[3])

    def tokenize(self, paths):
        """Tokenizes

        [fingerprint_features_zero,
         fingerprint_features_one,
         duration_features_zero,
         duration_features_one,
         voices,
         centers,
         key_zero,
         key_durations_zero,
         key_one,
         key_durations_one,
         targets]
        """
        fingerprint_features_zero = []
        fingerprint_features_one = []
        duration_features_zero = []
        duration_features_mid = []
        duration_features_one = []
        voices = []
        centers = []
        key_zero = []
        key_durations_zero = []
        key_one = []
        key_durations_one = []
        key_indicators = []
        targets = []

        for path in paths:
            ret = self._process(path)
            (flat_fingerprint_features_zero,
             flat_fingerprint_features_one,
             flat_duration_features_zero,
             flat_duration_features_mid,
             flat_duration_features_one,
             flat_voices,
             flat_centers,
             flat_key_zero,
             flat_key_durations_zero,
             flat_key_one,
             flat_key_durations_one,
             flat_key_indicators,
             flat_targets) = ret

            for fpz in flat_fingerprint_features_zero:
                fpz_token = self.fingerprint_features_zero_dictionary.word2idx[tuple(fpz)]
                fingerprint_features_zero.append(fpz_token)

            for fpo in flat_fingerprint_features_one:
                fpo_token = self.fingerprint_features_one_dictionary.word2idx[tuple(fpo)]
                fingerprint_features_one.append(fpo_token)

            for durfz in flat_duration_features_zero:
                durfz_token = self.duration_features_zero_dictionary.word2idx[durfz]
                duration_features_zero.append(durfz_token)

            for durfm in flat_duration_features_mid:
                durfm_token = self.duration_features_mid_dictionary.word2idx[durfm]
                duration_features_mid.append(durfm_token)

            for durfo in flat_duration_features_one:
                durfo_token = self.duration_features_one_dictionary.word2idx[durfo]
                duration_features_one.append(durfo_token)

            for v in flat_voices:
                 v_token = self.voices_dictionary.word2idx[v]
                 voices.append(v_token)

            for c in flat_centers:
                center_0_token = self.centers_0_dictionary.word2idx[c[0]]
                center_1_token = self.centers_1_dictionary.word2idx[c[1]]
                center_2_token = self.centers_2_dictionary.word2idx[c[2]]
                center_3_token = self.centers_3_dictionary.word2idx[c[3]]
                centers.append([center_0_token,
                                center_1_token,
                                center_2_token,
                                center_3_token])

            for kz in flat_key_zero:
                kz_token = self.keypoint_dictionary.word2idx[tuple(kz)]
                key_zero.append(kz_token)

            for kdz in flat_key_durations_zero:
                kdz_token = self.keypoint_durations_dictionary.word2idx[tuple(kdz)]
                key_durations_zero.append(kdz_token)

            for ko in flat_key_one:
                ko_token = self.keypoint_dictionary.word2idx[tuple(ko)]
                key_one.append(ko_token)

            for kdo in flat_key_durations_one:
                kdo_token = self.keypoint_durations_dictionary.word2idx[tuple(kdo)]
                key_durations_one.append(kdo_token)

            for fki in flat_key_indicators:
                key_indicators.append(fki)

            for t in flat_targets:
                target_0_token = self.target_0_dictionary.word2idx[t[0]]
                target_1_token = self.target_1_dictionary.word2idx[t[1]]
                target_2_token = self.target_2_dictionary.word2idx[t[2]]
                target_3_token = self.target_3_dictionary.word2idx[t[3]]
                targets.append([target_0_token, target_1_token, target_2_token, target_3_token])

        outs = [fingerprint_features_zero,
                fingerprint_features_one,
                duration_features_zero,
                duration_features_mid,
                duration_features_one,
                voices,
                centers,
                key_zero,
                key_durations_zero,
                key_one,
                key_durations_one,
                key_indicators,
                targets]
        assert len(outs[0]) == len(outs[1])
        for i in range(2, len(outs)):
            assert len(outs[0]) == len(outs[i])
        return outs



class MusicJSONFlatMeasureCorpus(object):
    def __init__(self, train_data_file_paths, valid_data_file_paths=None, test_data_file_paths=None,
                 max_vocabulary_size=-1,
                 add_measure_marks=True,
                 tokenization_fn="measure_flatten",
                 default_velocity=120, n_voices=4,
                 measure_value=99,
                 fill_value=-1,
                 separate_onsets=True, onsets_boundary=100):
        """
        """
        self.pitch_dictionary = LookupDictionary()
        self.duration_dictionary = LookupDictionary()
        self.velocity_dictionary = LookupDictionary()
        self.voice_dictionary = LookupDictionary()
        
        self.max_vocabulary_size = max_vocabulary_size
        self.default_velocity = default_velocity
        self.n_voices = n_voices
        self.measure_value = measure_value
        self.fill_value = fill_value
        self.separate_onsets = separate_onsets
        self.onsets_boundary = onsets_boundary
        self.add_measure_marks = add_measure_marks

        if tokenization_fn != "measure_flatten":
            raise ValueError("Only default tokenization_fn currently supported")

        base = [fp for fp in train_data_file_paths]
        if valid_data_file_paths is not None:
            base = base + [fp for fp in valid_data_file_paths]
        if test_data_file_paths is not None:
            base = base + [fp for fp in test_data_file_paths]

        self.build_vocabulary(base)

        if self.max_vocabulary_size > -1:
            self.dictionary._prune_to_top_k_counts(self.max_vocabulary_size)

        self.train = self.tokenize(train_data_file_paths)
        if valid_data_file_paths is not None:
            self.valid = self.tokenize(valid_data_file_paths)
        if test_data_file_paths is not None:
            self.test = self.tokenize(test_data_file_paths)

    def _process(self, path):
        assert os.path.exists(path)
        pitch, duration, velocity = pitch_duration_velocity_lists_from_music_json_file(path,
                                                                                       default_velocity=self.default_velocity,
                                                                                       n_voices=self.n_voices,
                                                                                       measure_value=self.measure_value,
                                                                                       fill_value=self.fill_value)

        def isplit(iterable, splitters):
            return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]

        group = []
        for v in range(len(pitch)):
            # per voice, do split and merge
            s_p = isplit(pitch[v], [self.measure_value])
            s_d = isplit(duration[v], [self.fill_value])
            s_v = isplit(velocity[v], [self.fill_value])
            group.append([s_p, s_d, s_v])

        not_merged = True
        # all should be the same length in terms of measures so we can merge
        try:
            assert len(group[0][0]) == len(group[1][0])
            for g in group:
                assert len(group[0][0]) == len(g[0])
        except:
            raise ValueError("Group check assertion failed in _process of MusicJSONFlatMeasureCorpus")

        # just checked that all have the same number of measures, so now we combine them
        flat_pitch = []
        flat_duration = []
        flat_velocity = []
        flat_voice = []

        flat_pitch.append(self.measure_value)
        flat_duration.append(self.fill_value)
        flat_velocity.append(self.fill_value)
        flat_voice.append(len(pitch))

        for i in range(len(group[0][0])):
            for v in range(len(group)):
                m_p = group[v][0][i]
                m_d = group[v][1][i]
                m_v = group[v][2][i]
                m_vv = [v for el in m_p]

                flat_pitch.extend(m_p)
                flat_duration.extend(m_d)
                flat_velocity.extend(m_v)
                flat_voice.extend(m_vv)
            flat_pitch.append(self.measure_value)
            flat_duration.append(self.fill_value)
            flat_velocity.append(self.fill_value)
            flat_voice.append(len(pitch))
        return flat_pitch, flat_duration, flat_velocity, flat_voice


    def build_vocabulary(self, json_file_paths):
        """Tokenizes a text file."""
        for path in json_file_paths:
            pitch, duration, velocity, voice = self._process(path)

            for p in pitch:
                self.pitch_dictionary.add_word(p)

            for d in duration:
                self.duration_dictionary.add_word(d)

            for v in velocity:
                self.velocity_dictionary.add_word(v)

            for vv in voice:
                self.voice_dictionary.add_word(vv)


    def tokenize(self, paths):
        """Tokenizes a text file."""
        pitches = []
        durations = []
        velocities = []
        voices = []
        for path in paths:
            pitch, duration, velocity, voice = self._process(path)

            # do we even bother to check for unknown words
            for p in pitch:
                p_token = self.pitch_dictionary.word2idx[p]
                pitches.append(p_token)

            for d in duration:
                d_token = self.duration_dictionary.word2idx[d]
                durations.append(d_token)

            for v in velocity:
                v_token = self.velocity_dictionary.word2idx[v]
                velocities.append(v_token)

            for vv in voice:
                vv_token = self.voice_dictionary.word2idx[vv]
                voices.append(vv_token)

        return pitches, durations, velocities, voices


class MusicJSONCorpus(object):
    def __init__(self, train_data_file_paths, valid_data_file_paths=None, test_data_file_paths=None,
                 max_vocabulary_size=-1,
                 add_eos=False,
                 eos_amount=32,
                 eos_symbol=0,
                 tokenization_fn="flatten",
                 default_velocity=120, n_voices=4,
                 separate_onsets=True, onsets_boundary=100):
        """
        """
        self.dictionary = LookupDictionary()

        self.max_vocabulary_size = max_vocabulary_size
        self.default_velocity = default_velocity
        self.quantization_rate = quantization_rate
        self.n_voices = n_voices
        self.separate_onsets = separate_onsets
        self.onsets_boundary = onsets_boundary
        self.add_eos = add_eos
        self.eos_amount = eos_amount
        self.eos_symbol = eos_symbol

        if tokenization_fn == "flatten":
            def tk(arr):
                t = [el for el in arr.ravel()]
                if add_eos:
                    # 2 measures of silence are the eos
                    return t + [self.eos_symbol] * self.eos_amount
                else:
                    return t
            self.tokenization_fn = tk
        else:
            raise ValueError("Unknown tokenization_fn {}".format(tokenization_fn))

        base = [fp for fp in train_data_file_paths]
        if valid_data_file_paths is not None:
            base = base + [fp for fp in valid_data_file_paths]
        if test_data_file_paths is not None:
            base = base + [fp for fp in test_data_file_paths]

        self.build_vocabulary(base)

        if self.max_vocabulary_size > -1:
            self.dictionary._prune_to_top_k_counts(self.max_vocabulary_size)

        self.train = self.tokenize(train_data_file_paths)
        if valid_data_file_paths is not None:
            self.valid = self.tokenize(valid_data_file_paths)
        if test_data_file_paths is not None:
            self.test = self.tokenize(test_data_file_paths)

    def build_vocabulary(self, json_file_paths):
        """Tokenizes a text file."""
        for path in json_file_paths:
            assert os.path.exists(path)
            roll = piano_roll_from_music_json_file(path,
                                                   default_velocity=self.default_velocity,
                                                   quantization_rate=self.quantization_rate,
                                                   n_voices=self.n_voices,
                                                   separate_onsets=self.separate_onsets,
                                                   onsets_boundary=self.onsets_boundary,
                                                   as_numpy=True)
            words = self.tokenization_fn(roll)

            for word in words:
                self.dictionary.add_word(word)

    def tokenize(self, paths):
        """Tokenizes a text file."""
        ids = []
        for path in paths:
            assert os.path.exists(path)
            # Add words to the dictionary
            roll = piano_roll_from_music_json_file(path,
                                                   default_velocity=self.default_velocity,
                                                   quantization_rate=self.quantization_rate,
                                                   n_voices=self.n_voices,
                                                   separate_onsets=self.separate_onsets,
                                                   onsets_boundary=self.onsets_boundary,
                                                   as_numpy=True)
            words = self.tokenization_fn(roll)
            for word in words:
                if word in self.dictionary.word2idx:
                    token = self.dictionary.word2idx[word]
                else:
                    token = self.dictionary.word2idx["<unk>"]
                ids.append(token)
        return ids


class MusicJSONRasterIterator(object):
    def __init__(self, list_of_music_json_files,
                 batch_size,
                 max_sequence_length,
                 random_seed,
                 n_voices=4,
                 iterate_once=False,
                 with_clocks=[2, 4, 8, 16, 32, 64],
                 separate_onsets=False,
                 #with_clocks=None,
                 resolution="sixteenth"):
        super(MusicJSONRasterIterator, self).__init__()
        self.list_of_music_json_files = list_of_music_json_files
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(random_seed)
        self.file_list_indices_ = list(range(len(self.list_of_music_json_files)))
        self.iterate_once = iterate_once
        self.iterate_at_ = 0
        self.batch_size = batch_size
        self.separate_onsets = separate_onsets
        if self.iterate_once:
            pass
        else:
            self.random_state.shuffle(self.file_list_indices_)
        self.file_list_indices_ = self.file_list_indices_[self.iterate_at_:self.iterate_at_ + self.batch_size]
        self.resolution = resolution
        self.max_sequence_length = max_sequence_length
        self.n_voices = n_voices
        if self.resolution != "sixteenth":
            raise ValueError("Currently only support 16th note resolution")
        if self.n_voices != 4:
            raise ValueError("Currently only support 4 voices")
        self.with_clocks = with_clocks
        # build vocabularies now?

    def next(self):
        return self.__next__()

    def __iter__(self):
        while True:
            yield next(self)

    def __next__(self):
        # -1 value for padding - will convert to 0s but mask in the end
        all_roll_voices = []
        for fli in self.file_list_indices_:
            json_file = self.list_of_music_json_files[fli]
            with open(json_file) as f:
                data = json.load(f)
            ppq = data["pulses_per_quarter"]
            qbpm = data["quarter_beats_per_minute"]
            spq = data["seconds_per_quarter"]

            parts = data["parts"]
            parts_times = data["parts_times"]
            parts_cumulative_times = data["parts_cumulative_times"]
            # https://github.com/cuthbertLab/music21/blob/c6fc39204c16c47d1c540b545d0c9869a9cafa8f/music21/midi/__init__.py#L1471
            if "parts_velocities" not in data:
                default_velocity = 120
                parts_velocities = [[default_velocity] * len(p) for p in parts]
            else:
                parts_velocities = data["parts_velocities"]
            end_in_quarters = max([max(p) for p in parts_cumulative_times])
            # clock is set currently by the fact that "sixteenth" is the only option
            # .25 due to "sixteenth"
            clock = np.arange(0, max(max(parts_cumulative_times)), .25)
            # 4 * for 4 voices
            raster_end_in_steps = 4 * len(clock)
            if raster_end_in_steps > self.max_sequence_length:
                pass
                # need to randomly slice out a chunk that fits? or just take the first steps?
                # let it go for now

            # start with 1 quarter note (4 16ths) worth of pure rest
            roll_voices = [[0] * 4, [0] * 4, [0] * 4, [0] * 4]
            # use these for tracking if we cross a change event
            p_i = [0, 0, 0, 0]
            for c in clock:
                # voice
                for v in range(len(parts)):
                    current_note = parts[v][p_i[v]]
                    next_change_time = parts_cumulative_times[v][p_i[v]]
                    new_onset = False
                    if c >= next_change_time:
                        # we hit a boundary, swap notes
                        p_i[v] += 1
                        current_note = parts[v][p_i[v]]
                        next_change_time = parts_cumulative_times[v][p_i[v]]
                        new_onset = True
                    if c == 0. or new_onset:
                        if current_note != 0:
                            if self.separate_onsets:
                                roll_voices[v].append(current_note + 100)
                            else:
                                roll_voices[v].append(current_note)
                        else:
                            # rests have no "onset"
                            roll_voices[v].append(current_note)
                    else:
                       roll_voices[v].append(current_note)
            all_roll_voices.append(roll_voices)

        raster_roll_voices = np.zeros((self.max_sequence_length, self.batch_size, 1)) - 1.

        if self.with_clocks is not None:
            all_clocks = [np.zeros((self.max_sequence_length, self.batch_size, 1)) for ac in self.with_clocks]

        for n, rv in enumerate(all_roll_voices):
            # transpose from 4 long sequences, to a long sequence of 4 "tuples"
            i_rv = [[t_rv[i] for t_rv in rv] for i in range(len(rv[0]))]
            raster_i_rv = [r for step in i_rv for r in step]
            if self.with_clocks is not None:
                # create clock signals, by taking time index modulo each value
                clock_base = [[t for t_rv in rv] for t in range(len(rv[0]))]
                clock_base = [clk for step in clock_base for clk in step]
                lcl_clocks = []
                for cl_i in self.with_clocks:
                    this_clock = [cb % cl_i for cb in clock_base]
                    lcl_clocks.append(this_clock)

            slicer = 0
            if len(raster_i_rv) > self.max_sequence_length:
                # find a point to start where either all voices rest, or all voices are onsets!
                # guaranteed to have at least 1 start point due to beginning, avoid those if we can
                proposed_cuts = [all([i_rv_ii > 100 or i_rv_ii == 0 for i_rv_ii in i_rv_i]) for i_rv_i in i_rv]
                proposed_cuts_i = [i for i in range(len(proposed_cuts)) if proposed_cuts[i] is True]

                # prune to only the cuts that give us a full self.max_sequence_length values after rasterizing
                proposed_cuts_i = [pci for pci in proposed_cuts_i if pci * self.n_voices + self.max_sequence_length <= len(raster_i_rv)]

                if len(proposed_cuts_i) == 0:
                    # edge case if none qualify
                    proposed_cuts_i = [0]
                # shuffle to get one at random - shuffle is in place so we choose the first one
                self.random_state.shuffle(proposed_cuts_i)

                step_slicer = proposed_cuts_i[0]

                # turn it into a raster pointer instead of a "voice tuple" pointer
                slicer = step_slicer * self.n_voices
            subslice = raster_i_rv[slicer:slicer + self.max_sequence_length]
            raster_roll_voices[:len(subslice), n, 0] = subslice
            # we broadcast it to self.batch_size soon
            if self.with_clocks is not None:
                for _i, ac in enumerate(lcl_clocks):
                    all_clocks[_i][:len(subslice), n, 0] = ac[slicer:slicer + self.max_sequence_length]
        # take off trailing 1 from shape
        mask = np.array(raster_roll_voices >= 0.).astype(np.float32)[..., 0]
        # np.abs to get rid of -0. , is annoying to me
        raster_roll_voices = np.abs(raster_roll_voices * mask[..., None])

        # setup new file_list_indices - use a new song for each batch element
        self.file_list_indices_ = list(range(len(self.list_of_music_json_files)))
        if self.iterate_once:
            self.iterate_at_ += self.batch_size
        else:
            self.random_state.shuffle(self.file_list_indices_)
        self.file_list_indices_ = self.file_list_indices_[self.iterate_at_:self.iterate_at_ + self.batch_size]
        if len(self.file_list_indices_) != self.batch_size:
            if self.iterate_once and len(self.file_list_indices_) > 0:
                # let the last batch through for iterate_once / vocabulary and statistics checks, etc
                pass
            else:
                raise ValueError("Unknown error, not enough file list indices to iterate! Current indices {}".format(self.file_list_indices_))
        if self.with_clocks is None:
            return raster_roll_voices, mask
        else:
            return raster_roll_voices, mask, [ac.astype(np.float32) * mask[..., None] for ac in all_clocks]


def convert_voice_roll_to_music_json(voice_roll, quantization_rate=.25, onsets_boundary=100):
    """
    take in voice roll and turn it into a pitch, duration thing again

    currently assume onsets are any notes > 100 , 0 is rest

    example input, where 170, 70, 70, 70 is an onset of pitch 70 (noted as 170), followed by a continuation for 4 steps
    array([[170.,  70.,  70.,  70.],
           [165.,  65.,  65.,  65.],
           [162.,  62.,  62.,  62.],
           [158.,  58.,  58.,  58.]])
    """
    duration_step = quantization_rate
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
            if voice_roll[v][t] > onsets_boundary:
                voice_data["parts"][v].append(note_held)
                voice_data["parts_times"][v].append(ongoing_duration)
                ongoing_duration = duration_step
                note_held = token - onsets_boundary
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


def convert_voice_lists_to_music_json(pitch_lists, duration_lists, velocity_lists=None, voices_list=None,
                                      default_velocity=120,
                                      measure_value=99,
                                      onsets_boundary=100):
    """
    can either work by providing a list of lists input for pitch_lists and duration_lists (optionally velocity lists)

    or

    1 long list for pitch, 1 long list for duration (optionally velocity), and the voices_list argument which has
    indicators for each voice and how it maps
    """
    voice_data = {}
    voice_data["parts"] = []
    voice_data["parts_times"] = []
    voice_data["parts_cumulative_times"] = []
    if voices_list is not None:
        assert len(pitch_lists) == len(duration_lists)
        voices = sorted(list(set(voices_list)))
        for v in voices:
            selector = [v_i == v for v_i in voices_list]
            parts = [pitch_lists[i] for i in range(len(pitch_lists)) if selector[i]]
            parts_times = [duration_lists[i] for i in range(len(pitch_lists)) if selector[i]]
            if velocity_lists is not None:
                parts_velocities = [velocity_lists[i] for i in range(len(pitch_lists)) if selector[i]]
            else:
                parts_velocities = [default_velocity for i in range(len(pitch_lists)) if selector[i]]
            # WE ASSUME MEASURE SELECTOR IS A unIQue ONE
            if any([p == measure_value for p in parts]):
                continue
            else:
                voice_data["parts"].append(parts)
                voice_data["parts_times"].append(parts_times)
                voice_data["parts_cumulative_times"].append([0.] + [e for e in np.cumsum(parts_times)])
    else:
        from IPython import embed; embed(); raise ValueError()
    spq = .5
    ppq = 220
    qbpm = 120
    voice_data["seconds_per_quarter"] = spq
    voice_data["quarter_beats_per_minute"] = qbpm
    voice_data["pulses_per_quarter"] = ppq
    voice_data["parts_names"] = ["Soprano", "Alto", "Tenor", "Bass"]
    j = json.dumps(voice_data, indent=4)
    return j


class MusicJSONVoiceIterator(object):
    """
            return pitch_batch, time_batch, voice_batch, cumulative_time_batch, mask
        else:
            return pitch_batch, time_batch, voice_batch, cumulative_time_batch, mask, clock_batches
    """
    def __init__(self, list_of_music_json_files,
                 batch_size,
                 max_sequence_length,
                 random_seed,
                 n_voices=4,
                 rest_marked_durations=True,
                 iterate_once=False,
                 with_clocks=[2, 4, 8, 16, 32, 64],
                 resolution="sixteenth"):
        super(MusicJSONVoiceIterator, self).__init__()
        self.list_of_music_json_files = list_of_music_json_files
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(random_seed)
        self.file_list_indices_ = list(range(len(self.list_of_music_json_files)))
        self.iterate_once = iterate_once
        self.iterate_at_ = 0
        self.batch_size = batch_size
        self.rest_marked_durations = rest_marked_durations
        if self.iterate_once:
            pass
        else:
            self.random_state.shuffle(self.file_list_indices_)
        self.file_list_indices_ = self.file_list_indices_[self.iterate_at_:self.iterate_at_ + self.batch_size]
        self.resolution = resolution
        self.max_sequence_length = max_sequence_length
        self.n_voices = n_voices
        if self.resolution != "sixteenth":
            raise ValueError("Currently only support 16th note resolution")
        if self.n_voices != 4:
            raise ValueError("Currently only support 4 voices")
        self.with_clocks = with_clocks
        # build vocabularies now?

    def next(self):
        return self.__next__()

    def __iter__(self):
        while True:
            yield next(self)

    def __next__(self):
        # -1 value for padding - will convert to 0s but mask in the end
        all_roll_voice_times = []
        all_roll_voice_pitches = []
        for fli in self.file_list_indices_:
            json_file = self.list_of_music_json_files[fli]
            with open(json_file) as f:
                data = json.load(f)
            ppq = data["pulses_per_quarter"]
            qbpm = data["quarter_beats_per_minute"]
            spq = data["seconds_per_quarter"]

            parts = data["parts"]
            parts_times = data["parts_times"]
            parts_cumulative_times = data["parts_cumulative_times"]
            # https://github.com/cuthbertLab/music21/blob/c6fc39204c16c47d1c540b545d0c9869a9cafa8f/music21/midi/__init__.py#L1471
            if "parts_velocities" not in data:
                default_velocity = 120
                parts_velocities = [[default_velocity] * len(p) for p in parts]
            else:
                parts_velocities = data["parts_velocities"]

            all_roll_voice_times.append(parts_times)
            all_roll_voice_pitches.append(parts)

        all_flat_pitch = []
        all_flat_voice = []
        all_flat_time = []
        all_flat_cumulative_step = []
        all_flat_cumulative_time = []
        for n in range(len(all_roll_voice_times)):
            flat_pitch = []
            flat_voice = []
            flat_time = []
            flat_cumulative_step = []
            flat_cumulative_time = []
            # multi-voice, should be 4
            this_time = all_roll_voice_times[n]
            this_pitch = all_roll_voice_pitches[n]
            this_cumulative_time_start = [[0] + [int(el) for el in np.cumsum(vv)] for vv in all_roll_voice_times[n]]
            finished = False
            # track which step 
            n_voices = len(this_pitch)
            voice_time_counter = [0] * n_voices
            voice_step_counter = [0] * n_voices
            last_event_time = -1
            next_event_time = -1
            keep_voices = [0, 1, 2, 3]
            # semi-dynamic program to make a flat sequence out of a stacked event sequence
            while True:
                if len(keep_voices) == 0:
                    #print("terminal")
                    # we need to be sure we got to the end! but how...
                    break

                if last_event_time < 0:
                    # frist
                    for v in range(n_voices):
                        flat_pitch.append(this_pitch[v][0])
                        flat_voice.append(v)
                        flat_time.append(this_time[v][0])
                        flat_cumulative_step.append(voice_step_counter[v])
                        flat_cumulative_time.append(voice_time_counter[v])
                        voice_time_counter[v] += this_time[v][0]
                        voice_step_counter[v] += 1
                    last_event_time = 0
                    next_event_time = min([min(cts[1:]) for cts in this_cumulative_time_start])
                    # need to do something about if it was a rest or not?
                else:
                    # now
                    for v in range(n_voices):
                        if v not in keep_voices:
                            continue

                        if this_cumulative_time_start[v][voice_step_counter[v]] == next_event_time:
                            flat_pitch.append(this_pitch[v][voice_step_counter[v]])
                            flat_voice.append(v)
                            flat_time.append(this_time[v][voice_step_counter[v]])
                            flat_cumulative_step.append(voice_step_counter[v])
                            flat_cumulative_time.append(voice_time_counter[v])
                            voice_time_counter[v] += this_time[v][voice_step_counter[v]]
                            voice_step_counter[v] += 1
                    last_event_time = next_event_time
                    next_event_time = min([min(cts[voice_step_counter[vi]:]) for vi, cts in enumerate(this_cumulative_time_start) if vi in keep_voices])
                # check if we hit the end of 1 voice
                keep_voices = []
                for v in range(n_voices):
                    if voice_step_counter[v] >= len(this_time[v]):
                        pass
                        #print("dawhkj")
                        #from IPython import embed; embed(); raise ValueError()
                    else:
                        keep_voices.append(v)
                #print(keep_voices)
            all_flat_pitch.append(flat_pitch)
            all_flat_voice.append(flat_voice)
            all_flat_time.append(flat_time)
            all_flat_cumulative_step.append(flat_cumulative_step)
            all_flat_cumulative_time.append(flat_cumulative_time)

        maxlen = max([len(tv) for tv in all_flat_voice])
        pitch_batch = np.zeros((maxlen, self.batch_size, 1))
        voice_batch = np.zeros((maxlen, self.batch_size, 1))
        time_batch = np.zeros((maxlen, self.batch_size, 1))
        # make this one but it seems poitnless to return it
        cumulative_step_batch = np.zeros((maxlen, self.batch_size, 1))
        cumulative_time_batch = np.zeros((maxlen, self.batch_size, 1))
        mask = np.zeros((maxlen, self.batch_size, 1))

        for i in range(self.batch_size):
            l = len(all_flat_pitch[i])
            pitch_batch[:l, i, 0] = all_flat_pitch[i]
            voice_batch[:l, i, 0] = all_flat_voice[i]
            if self.rest_marked_durations:
                # we give special duration marks to rest durations, adding 500 is a quick hack for that
                tft = all_flat_time[i]
                tfp = all_flat_pitch[i]
                tft = [tft[jj] if tfp[jj] != 0 else tft[jj] + 500 for jj in range(len(tft))]
                time_batch[:l, i, 0] = tft
            else:
                time_batch[:l, i, 0] = all_flat_time[i]
            cumulative_step_batch[:l, i, 0] = all_flat_cumulative_step[i]
            cumulative_time_batch[:l, i, 0] = all_flat_cumulative_time[i]
            mask[:l, i, 0] = 1.

        if self.with_clocks is not None:
            # create clock signals, by taking time index modulo each value
            clock_batches = [np.zeros((maxlen, self.batch_size, 1)) for c in self.with_clocks]
            for ii, c in enumerate(self.with_clocks):
                clock_batches[ii] = cumulative_time_batch % c


        # take off trailing 1 from shape
        mask = mask.astype(np.float32)[..., 0]

        # setup new file_list_indices - use a new song for each batch element
        self.file_list_indices_ = list(range(len(self.list_of_music_json_files)))
        if self.iterate_once:
            self.iterate_at_ += self.batch_size
        else:
            self.random_state.shuffle(self.file_list_indices_)
        self.file_list_indices_ = self.file_list_indices_[self.iterate_at_:self.iterate_at_ + self.batch_size]
        if len(self.file_list_indices_) != self.batch_size:
            if self.iterate_once and len(self.file_list_indices_) > 0:
                # let the last batch through for iterate_once / vocabulary and statistics checks, etc
                pass
            else:
                raise ValueError("Unknown error, not enough file list indices to iterate! Current indices {}".format(self.file_list_indices_))
        if self.with_clocks is None:
            return pitch_batch, time_batch, voice_batch, cumulative_time_batch, mask
        else:
            return pitch_batch, time_batch, voice_batch, cumulative_time_batch, mask, clock_batches


def fetch_jsb_chorales():
    jsb_dataset_path = check_fetch_jsb_chorales()
    json_files = [jsb_dataset_path + os.sep + fname for fname in sorted(os.listdir(jsb_dataset_path)) if ".json" in fname]
    return {"files": json_files}
