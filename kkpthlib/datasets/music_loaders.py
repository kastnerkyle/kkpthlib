from __future__ import print_function
from music21 import corpus, interval, pitch
from music21.midi import MidiTrack, MidiFile, MidiEvent, DeltaTime
import os
import json
import time
import numpy as np
from ..core import get_logger
from ..data import LookupDictionary
from .loaders import get_kkpthlib_dataset_dir
import collections

logger = get_logger()

def _music21_parse_and_save_json(p, fpath):
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
            part_time.append(n.duration.quarterLength)
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
                _music21_parse_and_save_json(p, core_name)
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
                _music21_parse_and_save_json(pn, transpose_fpath)
                if verbose:
                    logger.info("Writing {}".format(transpose_fpath))
        except Exception as e:
            if verbose:
                logger.info(e)
                logger.info("Skipping {} due to unknown error".format(p_bach))
            continue
    return dataset_path


def _populate_track_from_data(data, instrument=None):
    """
    example program change
    https://github.com/cuthbertLab/music21/blob/a78617291ed0aeb6595c71f82c5d398ebe604ef4/music21/midi/__init__.py
    >>> me2 = midi.MidiEvent(mt)
    >>> rem = me2.parseChannelVoiceMessage(to_bytes([0xC0, 71]))
    # program change to instrument 71
    """
    mt = MidiTrack(1)
    t = 0
    tLast = 0
    for d, p, v in data:
        dt = DeltaTime(mt)
        dt.time = t - tLast
        # add to track events
        mt.events.append(dt)

        me = MidiEvent(mt)
        me.type = "NOTE_ON"
        me.channel = 1
        me.time = None #d
        me.pitch = p
        me.velocity = v
        mt.events.append(me)

        # add note off / velocity zero message
        dt = DeltaTime(mt)
        dt.time = d
        # add to track events
        mt.events.append(dt)

        me = MidiEvent(mt)
        me.type = "NOTE_ON"
        me.channel = 1
        me.time = None #d
        me.pitch = p
        me.velocity = 0
        mt.events.append(me)

        tLast = t + d # have delta to note off
        t += d # next time

    # add end of track
    dt = DeltaTime(mt)
    dt.time = 0
    mt.events.append(dt)

    me = MidiEvent(mt)
    me.type = "END_OF_TRACK"
    me.channel = 1
    me.data = '' # must set data to empty string
    mt.events.append(me)
    return mt


def music_json_to_midi(json_file, out_name, tempo_factor=.5):
    """
    string (filepath) or json.dumps object

    tempo factor .5, twice as slow
    tempo factor 2, twice as fast
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
        default_velocity = 120
        parts_velocities = [[default_velocity] * len(p) for p in parts]
    else:
        print("handle velocities in json_to_midi")
        from IPython import embed; embed(); raise ValueError()

    all_mt = []
    for i in range(len(parts)):
        assert len(parts[i]) == len(parts_velocities[i])
        assert len(parts[i]) == len(parts_times[i])
        track_data = [[int(parts_times[i][j] * ppq), parts[i][j], parts_velocities[i][j]] for j in range(len(parts[i]))]
        mt = _populate_track_from_data(track_data)
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


class MusicJSONCorpus(object):
    def __init__(self, train_data_file_paths, valid_data_file_paths=None, test_data_file_paths=None,
                 max_vocabulary_size=-1,
                 add_eos=True,
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
