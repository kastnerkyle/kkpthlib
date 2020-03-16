from __future__ import print_function
from collections import defaultdict
import numpy as np
import os
from music21 import roman, stream, chord, midi, corpus, interval, pitch
import json
from operator import itemgetter
from itertools import groupby
import cPickle as pickle
import gc
import time

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
        attributions = []
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
                    attributions.append((tk, self.attribution_tags[tk]))
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
        """
        longest = len(list_of_items) - 1
        all_res = []
        all_attr = []
        for n, i in enumerate(self.orders):
            if return_attributions:
               res, attr = self.order_tries[n].order_search(i, list_of_items, return_attributions=return_attributions)
            else:
               res = self.order_tries[n].order_search(i, list_of_items, return_attributions=return_attributions)
            # need to even these out with padding
            if len(res) < longest:
                res = [None] * (longest - len(res)) + res
                if return_attributions:
                    attr = [None] * (longest - len(res)) + attr
            all_res.append(res)
            if return_attributions:
                all_attr.append(attr)
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

def get_music21_bach_metadata(only_pieces_with_n_voices=[4], assume_cached=False):
    """
    build metadata for plagiarism checks
    """
    metapath = "jsb_metadata"
    if assume_cached:
        print("JSB Chorales cached!")
        files_list = [metapath + os.sep + f for f in os.listdir(metapath) if ".metajson" in f]
        return {"files": files_list}
    all_bach_paths = corpus.getComposer('bach')
    print("JSB Chorales not yet cached, processing...")
    print("Total number of Bach pieces to process from music21: {}".format(len(all_bach_paths)))
    if not os.path.exists(metapath):
        os.mkdir(metapath)

    for it, p_bach in enumerate(all_bach_paths):
        if "riemenschneider" in p_bach:
            # skip certain files we don't care about
            continue
        p = corpus.parse(p_bach)
        if len(p.parts) not in only_pieces_with_n_voices:
            print("Skipping file {}, {} due to undesired voice count...".format(it, p_bach))
            continue

        if len(p.metronomeMarkBoundaries()) != 1:
            print("Skipping file {}, {} due to unknown or multiple tempo changes...".format(it, p_bach))
            continue

        print("Processing {}, {} ...".format(it, p_bach))
        stripped_extension_name = ".".join(os.path.split(p_bach)[1].split(".")[:-1])
        base_fpath = metapath + os.sep + stripped_extension_name
        skipped = False
        k = p.analyze('key')
        dp = get_metadata(p)
        if os.path.exists(base_fpath + ".metajson"):
            pass
        else:
            save_metadata_to_json(dp, base_fpath + ".metajson")

        for t in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
            if 'major' in k.name:
                kt = "major"
            elif 'minor' in k.name:
                kt = "minor"
            else:
                raise AttributeError('Unknown key {}'.format(kn.name))
            transpose_fpath = base_fpath + ".{}-{}-transposed.metajson".format(t, kt)
            if os.path.exists(transpose_fpath):
                print("file already exists for {}, skipping".format(transpose_fpath))
            else:
                i = interval.Interval(k.tonic, pitch.Pitch(t))
                pn = p.transpose(i)
                dpn = get_metadata(pn)
                save_metadata_to_json(dpn, transpose_fpath)
    files_list = [metapath + os.sep + f for f in os.listdir(metapath) if ".metajson" in f]
    return {"files": files_list}


def build_plagiarism_checkers(files):
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

    for n, jf in enumerate(files):
        print("growing plagiarism checker {}/{}".format(n + 1, len(files)))
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


def evaluate_midi_against_checkers(midi_path, checkers):
    p = music21_from_midi(midi_path)
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

    from IPython import embed; embed(); raise ValueError()

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
