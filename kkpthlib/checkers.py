from __future__ import print_function
from collections import defaultdict
import numpy as np
import os
from music21 import roman, stream, chord, midi, corpus, interval, pitch
import json
import shutil
from operator import itemgetter
from itertools import groupby
import cPickle as pickle
import gc
import time
from collections import OrderedDict
from .core import get_cache_dir
from .datasets import music_json_to_midi

class GroupDict(OrderedDict):
    def __init__(self):
        super(GroupDict, self).__init__()

    def groupby_tag(self):
        """
        'pivot' so keys are reduced, and matches

        turns:

        GroupDict([(('76:74:73:78:80:81:81:80->78', 1, 55),
                    [('bwv145-a.A-major-transposed.metajson', 0, 51),
                     ('bwv145-a.A-major-transposed.metajson', 18, 51)])])

        into

        OrderedDict([('bwv145-a.A-major-transposed.metajson',
                      [(0, 51, '76:74:73:78:80:81:81:80->78', 1, 55),
                       (18, 51, '76:74:73:78:80:81:81:80->78', 1, 55)])])

        where the first 2 indices are the step into key (bwv145) and its length
              the match key
              then the index into the example and its length

        """
        all_v = []
        for k, v in self.items():
            all_v.extend([vi[0] for vi in v])

        o = OrderedDict()

        done = []
        for p in all_v:
            if p in done:
                continue
            # key -> list 
            # [(('76:74:73:78:80:81:81:80->78', 1, 55),
            #  [('bwv145-a.A-major-transposed.metajson', 0, 51),
            #   ('bwv145-a.A-major-transposed.metajson', 18, 51)])]
            temp = [(k, [vi for vi in v if p in vi]) for k, v in self.items()]
            # prune empties
            temp = [t for t in temp if len(t[1]) > 0]
            # name of tag
            name = temp[0][1][0][0]
            for el in temp:
               o[name] = [(e[1], e[2]) + el[0] for e in el[1]]
            #o[name]
            done.append(p)
        return o


class Trie(object):
    """
    order_insert
    order_search

    are the primary methods
    trie = Trie()

    trie.order_insert(3, "string")

    or

    trie.order_insert(3, [<hashable_obj_instance1>, <hashable_obj_instance2>, ...]

    can optionally pass a tag in with_attribution_tag arguemnt
    """
    def __init__(self):
        self.root = defaultdict()
        self._end = "_end"
        self.orders = []
        self.attribution_tags = {}

    def insert(self, list_of_items):
        current = self.root
        for item in list_of_items:
            current = current.setdefault(item, {})
        current.setdefault(self._end)
        self.orders = sorted(list(set(self.orders + [len(list_of_items)])))

    def order_insert(self, order, list_of_items, with_attribution_tag=None):
        s = 0
        e = order
        while e < len(list_of_items):
            # + 1 due to numpy slicing
            e = s + order + 1
            el = list_of_items[s:e]
            self.insert(el)
            if with_attribution_tag:
                tk_seq = [str(eel).encode("ascii", "ignore") for eel in el]
                tk = ":".join(tk_seq[:-1]) + "->" + tk_seq[-1]
                if tk not in self.attribution_tags:
                    self.attribution_tags[tk] = []
                # tag is name of file, number of steps into sequence associated with that file, total number of items in the file
                self.attribution_tags[tk].append((with_attribution_tag, s, len(list_of_items)))
            s += 1

    def search(self, list_of_items):
        # items of the list should be hashable
        # returns True if item in Trie, else False
        if len(list_of_items) not in self.orders:
            raise ValueError("item {} has invalid length {} for search, only {} supported".format(list_of_items, len(list_of_items), self.orders))
        current = self.root
        for item in list_of_items:
            if item not in current:
                return False
            current = current[item]
        if self._end in current:
            return True
        return False

    def order_search(self, order, list_of_items, return_attributions=False):
        # returns true if subsequence at offset is found
        s = 0
        e = order
        searches = []
        attributions = GroupDict()
        while e < len(list_of_items):
            # + 1 due to numpy slicing
            e = s + order + 1
            el = list_of_items[s:e]
            ss = self.search(el)
            if ss and return_attributions:
                if not ss:
                    attributions.append(None)
                else:
                    tk_seq = [str(eel).encode("ascii", "ignore") for eel in el]
                    tk = ":".join(tk_seq[:-1]) + "->" + tk_seq[-1]
                    attributions[(tk, s, len(list_of_items))] = self.attribution_tags[tk]
            searches.append(ss)
            s += 1
        if return_attributions:
            return searches, attributions
        else:
            return searches


class MaxOrder(object):
    def __init__(self, max_order):
        assert max_order >= 2
        self.orders = list(range(1, max_order + 1))
        self.order_tries = [Trie() for n in self.orders]
        self.max_order = max_order

    def insert(self, list_of_items, with_attribution_tag=None):
        """
        a string, or a list of elements
        """
        if len(list_of_items) - 1 < self.max_order:
            raise ValueError("item {} to insert shorter than max_order!".format(list_of_items))

        for n, i in enumerate(self.orders):
            self.order_tries[n].order_insert(i, list_of_items, with_attribution_tag=with_attribution_tag)

    def included_at_index(self, list_of_items, return_attributions=False):
        """
        return a list of list values in [None, True, False]
        where None is pre-padding, True means the subsequence of list_of_items at that point is included
        False means not included

        attributions returned as custom OrderedDict of OrderedDict

        with extra methods for grouping / gathering
        e.g. for attr returned

        attr["order_8"].keys() will show all the keys with matches in the data
        attr["order_8"][key_name] will show the files, places, and total length of the match for all attribute tags
        attr["order_8"].groupby_tag() will "pivot" the matches to show all matches, with a list of the tag names and positions
        """
        longest = len(list_of_items) - 1
        all_res = []
        all_attr = OrderedDict()
        for n, i in enumerate(self.orders):
            if return_attributions:
                res, attr = self.order_tries[n].order_search(i, list_of_items, return_attributions=return_attributions)
            else:
                res = self.order_tries[n].order_search(i, list_of_items, return_attributions=return_attributions)
            # need to even these out with padding
            if len(res) < longest:
                res = [None] * (longest - len(res)) + res
            all_res.append(res)
            if return_attributions:
                all_attr["order_{}".format(i)] = attr
        if return_attributions:
            return all_res, all_attr
        else:
            return all_res

    def satisfies_max_order(self, list_of_items):
        if len(list_of_items) - 1 < self.max_order:
            return True
        matched = self.included_at_index(list_of_items)
        true_false_order = [any([mi for mi in m if mi is not None]) for m in matched]
        if all(true_false_order[:-1]):
            # if all the previous are conained, guarantee the last one IS contained
            return not true_false_order[-1]
        else:
            # if some of the previous ones were false, max order is satisfied
            return True

# Following functions from 
# https://www.kaggle.com/wfaria/midi-music-data-extraction-using-music21
def note_count(measure, count_dict):
    bass_note = None
    for chord in measure.recurse().getElementsByClass('Chord'):
        # All notes have the same length of its chord parent.
        note_length = chord.quarterLength
        for note in chord.pitches:
            # If note is "C5", note.name is "C". We use "C5"
            # style to be able to detect more precise inversions.
            note_name = str(note)
            if (bass_note is None or bass_note.ps > note.ps):
                bass_note = note
            if note_name in count_dict:
                count_dict[note_name] += note_length
            else:
                count_dict[note_name] = note_length
    return bass_note


def simplify_roman_name(roman_numeral):
    # Chords can get nasty names as "bII#86#6#5",
    # in this method we try to simplify names, even if it ends in
    # a different chord to reduce the chord vocabulary and display
    # chord function clearer.
    ret = roman_numeral.romanNumeral
    inversion_name = None
    inversion = roman_numeral.inversion()
    # Checking valid inversions.
    if ((roman_numeral.isTriad() and inversion < 3) or
            (inversion < 4 and
                 (roman_numeral.seventh is not None or roman_numeral.isSeventh()))):
        inversion_name = roman_numeral.inversionName()
    if (inversion_name is not None):
        ret = ret + str(inversion_name)
    elif (roman_numeral.isDominantSeventh()): ret = ret + "M7"
    elif (roman_numeral.isDiminishedSeventh()): ret = ret + "o7"
    return ret

def harmonic_reduction(part):
    ret_roman = []
    ret_chord = []
    temp_midi = stream.Score()
    temp_midi_chords = part.chordify()
    temp_midi.insert(0, temp_midi_chords)
    music_key = temp_midi.analyze('key')
    max_notes_per_chord = 4
    # bug in music21? chordify can return a thing without measures...
    if len(temp_midi_chords.measures(0, None)) == 0:
        print("chordify returned 0 measure stream, attempting to fix...")
        tt = stream.Stream()
        for tmc in temp_midi_chords:
            mm = stream.Measure()
            mm.insert(tmc)
            tt.append(mm)
        temp_midi_chords = tt

    for m in temp_midi_chords.measures(0, None): # None = get all measures.
        if (type(m) != stream.Measure):
            continue
        # Here we count all notes length in each measure,
        # get the most frequent ones and try to create a chord with them.
        count_dict = dict()
        bass_note = note_count(m, count_dict)
        if (len(count_dict) < 1):
            ret_roman.append("-") # Empty measure
            ret_chord.append(chord.Chord(["C0", "C0", "C0", "C0"]))
            continue
        sorted_items = sorted(count_dict.items(), key=lambda x:x[1])
        sorted_notes = [item[0] for item in sorted_items[-max_notes_per_chord:]]
        measure_chord = chord.Chord(sorted_notes)
        # Convert the chord to the functional roman representation
        # to make its information independent of the music key.
        roman_numeral = roman.romanNumeralFromChord(measure_chord, music_key)
        ret_roman.append(simplify_roman_name(roman_numeral))
        ret_chord.append(measure_chord)
    return ret_roman, ret_chord

def music21_from_midi(midi_path):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    return midi.translate.midiFileToStream(mf)

def get_metadata(p):
    piece_container = {}
    piece_container["parts"] = []
    piece_container["parts_times"] = []
    piece_container["parts_cumulative_times"] = []
    piece_container["parts_names"] = []
    roman_names, chord_objects = harmonic_reduction(p)
    # list of list of chord pitches in SATB order 
    chord_pitch_names = [[str(cpn) for cpn in cn.pitches][::-1] for cn in chord_objects]
    functional_names = [cn.commonName for cn in chord_objects]
    pitched_functional_names = [cn.pitchedCommonName for cn in chord_objects]
    piece_container["chord_pitches"] = chord_pitch_names
    piece_container["functional_names"] = functional_names
    piece_container["pitched_functional_names"] = pitched_functional_names
    piece_container["roman_names"] = roman_names

    for i, pi in enumerate(p.parts):
        piece_container["parts"].append([])
        piece_container["parts_times"].append([])
        piece_container["parts_cumulative_times"].append([])
        piece_container["parts_names"].append(pi.id)
        part = []
        part_time = []
        for n in pi.flat.notesAndRests:
            if n.isChord:
                continue

            if n.isRest:
                part.append(0)
            else:
                part.append(n.midi)
            part_time.append(n.duration.quarterLength)
        piece_container["parts"][i] += part
        piece_container["parts_times"][i] += part_time
        piece_container["parts_cumulative_times"][i] += list(np.cumsum(part_time))
    return piece_container


def save_metadata_to_json(piece_container, fpath):
    j = json.dumps(piece_container, indent=4)
    with open(fpath, "w") as f:
         print(j, file=f)


def get_music21_metadata(list_of_musicjson_or_midi_files, only_pieces_with_n_voices=[4], assume_cached=False,
                         metapath="_kkpthlib_cache", verbose=False):
    """
    build metadata for plagiarism checks
    """
    metapath = get_cache_dir() + "_{}_metadata".format(abs(hash(tuple(list_of_musicjson_or_midi_files))) % 10000)
    if assume_cached:
        print("Already cached!")
        files_list = [metapath + os.sep + f for f in os.listdir(metapath) if ".metajson" in f]
        if not os.path.exists(metapath):
            raise ValueError("{} does not exist, cannot assume cached!".format(metapath))
        return {"files": files_list}
    midi_cache_path = get_cache_dir() + "_{}_midicache".format(abs(hash(tuple(list_of_musicjson_or_midi_files))) % 10000)
    piece_paths = []
    for f in list_of_musicjson_or_midi_files:
        if f.endswith(".json"):
            if not os.path.exists(midi_cache_path):
                os.mkdir(midi_cache_path)
            basename = f.split(os.sep)[-1].split(".json")[0]
            out_midi = midi_cache_path + os.sep + basename + ".midi"
            if not os.path.exists(out_midi):
                music_json_to_midi(f, out_midi)
            piece_paths.append(out_midi)
        elif f.endswith(".midi") or f.endswith(".mid"):
            if not os.path.exists(midi_cache_path):
                os.mkdir(midi_cache_path)
            basename = f.split(os.sep)[-1].split(".mid")[0]
            out_path = midi_cache_path + os.sep + basename + ".midi"
            if not os.path.exists(out_path):
                shutil.copy2(f, out_path)
            piece_paths.append(out_path)
        else:
            raise ValueError("Unknown file type for file {}, expected .json (MusicJSON) or .midi/.mid".format(f))
    if not os.path.exists(metapath):
        os.mkdir(metapath)

    print("Not yet cached, processing...")
    print("Total number of pieces to process from music21: {}".format(len(piece_paths)))

    for it, piece in enumerate(piece_paths):
        p = music21_from_midi(piece)
        if len(p.parts) not in only_pieces_with_n_voices:
            print("Skipping file {}, {} due to undesired voice count...".format(it, p_bach))
            continue

        if len(p.metronomeMarkBoundaries()) != 1:
            print("Skipping file {}, {} due to unknown or multiple tempo changes...".format(it, p_bach))
            continue

        print("Processing {}, {} ...".format(it, piece))
        stripped_extension_name = ".".join(os.path.split(piece)[1].split(".")[:-1])
        base_fpath = metapath + os.sep + stripped_extension_name
        skipped = False
        k = p.analyze('key')
        dp = get_metadata(p)
        if os.path.exists(base_fpath + ".metajson"):
            pass
        else:
            save_metadata_to_json(dp, base_fpath + ".metajson")

    files_list = [metapath + os.sep + f for f in os.listdir(metapath) if ".metajson" in f]
    return {"files": files_list}


def build_music_plagiarism_checkers(metajson_files):
    """
    given list of metajson files, builds all the tries for checking plagiarism
    """
    roman_reduced_max_order = 5
    roman_reduced_checker = MaxOrder(roman_reduced_max_order)

    roman_checker_max_order = 5
    roman_checker = MaxOrder(roman_checker_max_order)

    pitched_functional_max_order = 5
    pitched_functional_checker = MaxOrder(pitched_functional_max_order)

    functional_max_order = 5
    functional_checker = MaxOrder(functional_max_order)

    pitches_max_order = 12
    soprano_pitch_checker = MaxOrder(pitches_max_order)
    alto_pitch_checker = MaxOrder(pitches_max_order)
    tenor_pitch_checker = MaxOrder(pitches_max_order)
    bass_pitch_checker = MaxOrder(pitches_max_order)

    for n, jf in enumerate(metajson_files):
        print("growing plagiarism checker {}/{}".format(n + 1, len(metajson_files)))
        with open(jf) as f:
            data = json.load(f)
        tag = jf.split(os.sep)[-1]
        roman_names = data["roman_names"]
        roman_reduced_names = [x[0] for x in groupby(roman_names)]
        roman_checker.insert(roman_names, with_attribution_tag=tag)

        pitched_functional_names = data["pitched_functional_names"]
        functional_names = data["functional_names"]

        roman_reduced_checker.insert(roman_reduced_names, with_attribution_tag=tag)
        pitched_functional_checker.insert(pitched_functional_names, with_attribution_tag=tag)
        functional_checker.insert(functional_names, with_attribution_tag=tag)
        soprano_pitch_checker.insert(data["parts"][0], with_attribution_tag=tag)
        alto_pitch_checker.insert(data["parts"][1], with_attribution_tag=tag)
        tenor_pitch_checker.insert(data["parts"][2], with_attribution_tag=tag)
        bass_pitch_checker.insert(data["parts"][3], with_attribution_tag=tag)
    return {"roman_names_checker": roman_checker,
            "roman_reduced_names_checker": roman_reduced_checker,
            "pitched_functional_checker": pitched_functional_checker,
            "functional_checker": functional_checker,
            "soprano_pitch_checker": soprano_pitch_checker,
            "alto_pitch_checker": alto_pitch_checker,
            "tenor_pitch_checker": tenor_pitch_checker,
            "bass_pitch_checker": bass_pitch_checker}


def evaluate_music_against_checkers(midi_or_musicjson_file_path, checkers):
    tmp_midi_path = "_tmp.midi"
    if os.path.exists(tmp_midi_path):
        os.remove(tmp_midi_path)

    f = midi_or_musicjson_file_path
    if f.endswith(".json"):
        music_json_to_midi(f, tmp_midi_path)
    elif f.endswith(".midi") or f.endswith(".mid"):
        shutil.copy2(f, tmp_midi_path)

    p = music21_from_midi(tmp_midi_path)
    dp = get_metadata(p)
    roman_names = dp["roman_names"]
    roman_reduced_names = [x[0] for x in groupby(roman_names)]
    functional_names = dp["functional_names"]
    pitched_functional_names = dp["pitched_functional_names"]

    roman_names_max_order_ok = checkers["roman_names_checker"].satisfies_max_order(roman_names)
    roman_names_matrix, roman_names_attr = checkers["roman_names_checker"].included_at_index(roman_names, return_attributions=True)
    print("Roman names checker status {}".format(roman_names_max_order_ok))

    roman_reduced_names_max_order_ok = checkers["roman_reduced_names_checker"].satisfies_max_order(roman_reduced_names)
    roman_reduced_names_matrix, roman_reduced_names_attr = checkers["roman_reduced_names_checker"].included_at_index(roman_reduced_names, return_attributions=True)
    print("Roman reduced names checker status {}".format(roman_reduced_names_max_order_ok))

    functional_names_max_order_ok = checkers["functional_checker"].satisfies_max_order(functional_names)
    functional_names_matrix, functional_names_attr = checkers["functional_checker"].included_at_index(functional_names, return_attributions=True)
    print("Functional names checker status {}".format(functional_names_max_order_ok))

    pitched_functional_names_max_order_ok = checkers["pitched_functional_checker"].satisfies_max_order(pitched_functional_names)
    pitched_functional_names_matrix, pitched_functional_names_attr = checkers["pitched_functional_checker"].included_at_index(pitched_functional_names, return_attributions=True)
    print("Pitched functional names checker status {}".format(pitched_functional_names_max_order_ok))

    soprano_parts = dp["parts"][0]
    alto_parts = dp["parts"][1]
    tenor_parts = dp["parts"][2]
    bass_parts = dp["parts"][3]

    soprano_pitch_max_order_ok = checkers["soprano_pitch_checker"].satisfies_max_order(soprano_parts)
    soprano_pitch_matrix, soprano_pitch_attr = checkers["soprano_pitch_checker"].included_at_index(soprano_parts, return_attributions=True)
    print("Soprano pitch checker status {}".format(soprano_pitch_max_order_ok))

    alto_pitch_max_order_ok = checkers["alto_pitch_checker"].satisfies_max_order(soprano_parts)
    alto_pitch_matrix, alto_pitch_attr = checkers["alto_pitch_checker"].included_at_index(alto_parts, return_attributions=True)
    print("Alto pitch checker status {}".format(alto_pitch_max_order_ok))

    tenor_pitch_max_order_ok = checkers["tenor_pitch_checker"].satisfies_max_order(tenor_parts)
    tenor_pitch_matrix, tenor_pitch_attr = checkers["tenor_pitch_checker"].included_at_index(tenor_parts, return_attributions=True)
    print("Tenor pitch checker status {}".format(tenor_pitch_max_order_ok))

    bass_pitch_max_order_ok = checkers["bass_pitch_checker"].satisfies_max_order(bass_parts)
    bass_pitch_matrix, bass_pitch_attr = checkers["bass_pitch_checker"].included_at_index(bass_parts, return_attributions=True)
    print("Bass pitch checker status {}".format(bass_pitch_max_order_ok))

    if os.path.exists(tmp_midi_path):
        os.remove(tmp_midi_path)

    from IPython import embed; embed(); raise ValueError()


'''
if __name__ == "__main__":
    # these two samples are identical for a while, not a bad idea to use these to test
    #path1 = "midi_samples_2142/temp20.midi"
    #path2 = "midi_samples_13/temp0.midi"
    #p1 = music21_from_midi(path1)
    #p2 = music21_from_midi(path2)
    #dp1 = get_metadata(p1)
    #dp2 = get_metadata(p2)

    # if it is the first time running this scripts, set assume_cached to false!
    meta = get_music21_bach_metadata(assume_cached=True)
    metafiles = meta["files"]
    skip = False
    if not skip:
        cached_checkers_path = "cached_checkers.pkl"
        if not os.path.exists(cached_checkers_path):
            checkers = build_plagiarism_checkers(metafiles)
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
    else:
        checkers = build_plagiarism_checkers(metafiles)

    midi_path = "midi_samples/temp0.midi"
    evaluate_midi_against_checkers(midi_path, checkers)
    from IPython import embed; embed(); raise ValueError()

    #corpus = ["random", "randint", "randnight"]
    #max_order = 4
    #checker = MaxOrder(max_order)
    #[checker.insert(c) for c in corpus]

    #checker.insert("purple")

    #a = checker.satisfies_max_order("purp")
    #b = checker.satisfies_max_order("purpt")
    #c = checker.satisfies_max_order("purpl")
    #d = checker.satisfies_max_order("purple")
    #e = checker.satisfies_max_order("purplez")
    #f = checker.satisfies_max_order("purplezz")
    #print(a)
    #print(b)
    #print(c)
    #print(d)
    #print(e)
    #print(f)
    #from IPython import embed; embed(); raise ValueError()
'''
