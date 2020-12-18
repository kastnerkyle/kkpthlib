from __future__ import print_function
import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F
import copy
import sys
import shutil
import time
import itertools
import functools

from typing import Generic, TypeVar, Dict, List, Optional
from csp import CSP, Constraint

import json
import os
import glob

parser = argparse.ArgumentParser(description="script {}".format(__file__))
parser.add_argument('saved_model_path', nargs='*', type=str)
parser.add_argument('--direct_saved_model_path', type=str, default=None,
                    help='location of the saved model, used the same as positional argument')
saved_model_def_default = "../{tag}.py"
parser.add_argument('--direct_saved_model_definition', type=str, default=saved_model_def_default,
                    help='location of saved model definition, a .py file')
parser.add_argument('--random_seed', '-r', type=int, default=2133,
                    help='random seed to use when sampling (default 2133)')
parser.add_argument('--data_seed', '-d', type=int, default=144,
                    help='random seed to use when seeding the sampling (default 144)')
parser.add_argument('--context_mult', '-c', type=int, default=3,
                    help='default context multiplier to combined with context size from training (default 3)')
parser.add_argument('--batch_size', '-b', type=int, default=10,
                    help='batch size for sample (default 10)')
parser.add_argument('--sample_len', '-s', type=int, default=1024,
                    help='how long of a sequence to sample (default 1024)')
parser.add_argument('--temperature', '-t', type=float, default=1.0,
                    help='sampling temperature to use (default 1.0)')
parser.add_argument('--p_cutoff', '-p', type=float, default=.5,
                    help='cutoff to use in top p sampling (default .5)')

args = parser.parse_args()
if len(args.saved_model_path) < 1:
    if args.direct_saved_model_path is None:
        parser.print_help()
        sys.exit(1)

if args.direct_saved_model_path is not None:
    saved_model_path = args.direct_saved_model_path
else:
    saved_model_path = args.saved_model_path[0]

if not os.path.exists(saved_model_path):
    raise ValueError("Unable to find argument {} for file load! Check your paths, and be sure to remove aliases such as '~'".format(saved_model_path))

if args.direct_saved_model_definition == saved_model_def_default:
    # here we search out the saved model file from ../
    saved_model_root = os.sep.join(os.path.split(saved_model_path)[:-1])
    if not saved_model_root.endswith("saved_models"):
        raise ValueError("Default directory structure not found - explicitly pass model def to --direct_saved_model_definition !")
    model_def_root = os.sep.join(os.path.split(saved_model_root)[:-1])
    py_files = sorted(glob.glob(model_def_root + os.sep + "*.py"))
    # find the first file that has a model def
    has_model_def = []
    for pf in py_files:
        with open(pf) as f:
            t = f.read()
            if "def build_model" in t and "def get_hparams" in t:
                has_model_def.append(True)
            else:
                has_model_def.append(False)
    if not any(has_model_def):
        raise ValueError("None of the detected model files {} has 'def build_model' or 'def get_hparams', normal functions for autoload!".format(py_files))
    comb = list(zip(py_files, has_model_def))
    temp = [c for c in comb if c[1]]
    saved_model_definition_path = temp[0][0]
else:
    saved_model_definition_path = args.direct_saved_model_definition


from kkpthlib import fetch_jsb_chorales
from kkpthlib import MusicJSONInfillCorpus
from kkpthlib import convert_voice_lists_to_music_json
from kkpthlib import write_music_json
from kkpthlib import music_json_to_midi

from importlib import import_module
model_import_path, fname = saved_model_definition_path.rsplit(os.sep, 1)
sys.path.insert(0, model_import_path)
p, _ = fname.split(".", 1)
mod = import_module(p)
get_hparams = getattr(mod, "get_hparams")
build_model = getattr(mod, "build_model")
sys.path.pop(0)

hp = get_hparams()

jsb = fetch_jsb_chorales()

# sort into minor / major, then by key
all_transposed = sorted([f for f in jsb["files"] if "original" not in f], key=lambda x:
                        (x.split(os.sep)[-1].split(".")[-2].split("transposed")[0].split("-")[0],
                         x.split(os.sep)[-1].split(".")[-2].split("transposed")[0].split("-")[1]))

bwv_names = sorted(list(set([f.split(os.sep)[-1].split(".")[0] for f in all_transposed])))
vrng = np.random.RandomState(144)
vrng.shuffle(bwv_names)

# 15 is ~5% of the data if you hold out all 12 transpositions
# holding out whole songs so actually a pretty hard validation set...
valid_names = bwv_names[:15]
train_files = [f for f in all_transposed if all([vn not in f for vn in valid_names])]
valid_files = [f for f in all_transposed if any([vn in f for vn in valid_names])]
assert all([vf not in train_files for vf in valid_files])

filepath = "sampled_trainfiles.txt"
with open(filepath, 'w') as file_handler:
    for item in sorted(train_files):
        file_handler.write("{}\n".format(item))

filepath = "sampled_validfiles.txt"
with open(filepath, 'w') as file_handler:
    for item in sorted(valid_files):
        file_handler.write("{}\n".format(item))

# shuffle the train and valid files before we make the flat_measure_corpus
vrng.shuffle(train_files)
vrng.shuffle(valid_files)

infill_corpus = MusicJSONInfillCorpus(train_data_file_paths=train_files,
                                      valid_data_file_paths=valid_files,
                                      raster_scan=True)
train_itr = infill_corpus.get_iterator(hp.batch_size, hp.random_seed)
valid_itr = infill_corpus.get_iterator(hp.batch_size, hp.random_seed + 1, _type="valid")

model = build_model(hp, infill_corpus)

model_dict = torch.load(saved_model_path, map_location=hp.use_device)
model.load_state_dict(model_dict)
model.eval()

from kkpthlib import softmax_np
from kkpthlib import top_k_from_logits_np
from kkpthlib import top_p_from_logits_np

midi_sample_dir = "midi_samples"

if os.path.exists(midi_sample_dir):
    print("Previous midi_samples folder found, removing...")
    shutil.rmtree(midi_sample_dir)

# this should ALWAYS happen
if not os.path.exists(midi_sample_dir):
    os.mkdir(midi_sample_dir)

tstamp = int(time.time())
itr = valid_itr

batch_np, batch_masks_np, batch_offsets_np, batch_file_indices = next(itr)

# sanity check one at a time
sampling_random_state = np.random.RandomState(111111)
for t in range(batch_np.shape[1]):
    # 1 runs forever never terminates
    # 3 has edge case boundary issue
    if t in [1,3,5]:
        continue
    batch_np = copy.deepcopy(batch_np)
    batch_masks_np = copy.deepcopy(batch_masks_np)
    batch_offsets_np = copy.deepcopy(batch_offsets_np)

    batch = torch.tensor(batch_np)
    pad_batch = torch.cat((0 * batch[:1] + infill_corpus.dictionary.word2idx[infill_corpus.fill_symbol], batch), 0).to(hp.use_device)
    batch_masks = torch.tensor(batch_masks_np).to(hp.use_device)

    pad_batch_offsets_np = np.concatenate((0 * batch_offsets_np[:1] + np.array([-1, 0, 0])[None, None, :], batch_offsets_np), axis=0)

    input_batch = pad_batch[:-1]
    target_batch = pad_batch[1:].long()
    target_aligned_offsets = pad_batch_offsets_np[1:]
    input_aligned_offsets = pad_batch_offsets_np[:-1]

    context_token = infill_corpus.dictionary.word2idx[infill_corpus.end_context_symbol]
    answer_token = infill_corpus.dictionary.word2idx[infill_corpus.answer_symbol]
    mask_token = infill_corpus.dictionary.word2idx[infill_corpus.mask_symbol]

    # +1 to include context boundary
    context_boundary = np.where(input_batch[:, t, 0] == context_token)[0][0] + 1
    mask_positions = np.where(input_batch[:context_boundary, t, 0] == mask_token)[0]

    t1 = input_batch[context_boundary:]
    t2 = pad_batch_offsets_np[context_boundary:]
    a1 = [infill_corpus.dictionary.idx2word[ts] for ts in t1[:, t, 0]]
    assert len(a1) == len(t2)
    _a1_durations = [aa1[1] for aa1 in a1]
    _t2_durations = t2[:, t, -1]
    # ensure the sequences line up
    assert all([_a == _t for _a, _t in zip(_a1_durations, _t2_durations) if _a > 0])
    assert all([_t == 0 for _a, _t in zip(_a1_durations, _t2_durations) if _a <= 0])

    # 2 separate checks because < 0 is "special" symbols in the iterator, should have gt duration of 0

    # get expected duration total for each answer, for each voice!
    gt_answers = []
    gt_offsets = []
    _pre = [_aa1 if _aa1 > 0 else None for _aa1 in _a1_durations]
    # collapse consequentive None values?
    split_idx = [_n for _n in range(len(_pre)) if _pre[_n] == None]
    if split_idx[0] != 0:
        split_idx.insert(0, 0)
    # if the last split idx isn't at the last element, add a dummy splitter
    if split_idx[-1] != (len(_pre) - 1):
         # list grabs aren't inclusive, so we add 1 so that l[a:b] grabs elem b-1 (the last value in our data)
         split_idx.append(len(_pre))

    boundaries = list(zip(split_idx[:-1], split_idx[1:]))
    for b in boundaries:
        if b[0] + 1 < b[1]:
            if b[0] > 0:
                # +1 to skip the boundary value of None
                gt_answers.append(a1[b[0] + 1:b[1]])
                gt_offsets.append(t2[b[0] + 1:b[1], t])
            else:
                gt_answers.append(a1[b[0]:b[1]])
                gt_offsets.append(t2[b[0]:b[1], t])
        else:
            if b[1] == (len(_pre) - 1):
                # edge case where the last dummy chunk only has 1 elem
                # skip it? last value is end indicator
                #print("example 2")
                #from IPython import embed; embed(); raise ValueError()
                pass
                #gt_answers.append(a1[b[0] + 1:])
                #gt_offsets.append(t2[b[0] + 1:, t])
            else:
                raise ValueError("boundary issue")

    # now that we have gathered the answer groups, make sure there are the same number as mask positions
    try:
        assert len(gt_answers) == len(mask_positions)
        assert len(gt_offsets) == len(mask_positions)
    except:
        from IPython import embed; embed(); raise ValueError()
    from IPython import embed; embed(); raise ValueError()


    class SelfConstraint(Constraint):
        def __init__(self, voice1: int, notes: List[tuple], total_duration_constraint_per_voice: List[float]) -> None:
            super().__init__([voice1])
            self.voice1 = voice1
            self.notes = notes
            self.total_duration_constraint_per_voice = total_duration_constraint_per_voice
            self.active_voice_set = [_n for _n, d in enumerate(total_duration_constraint_per_voice) if d > 0]

        def satisfied(self, assignment: Dict[int, List[tuple]]) -> bool:
            notes_assigned = assignment[self.voice1]
            these_durations = [el[1][1] for el in notes_assigned]
            # do all-but-last relaxation here?
            # > 0 so that we don't care where "marker" symbols like end of file, etc go
            if sum([t for t in these_durations if t > 0]) != self.total_duration_constraint_per_voice[self.voice1]:
                return False
            return True
            """
            voice_sums = [-1 for v in range(len(self.total_duration_constraint_per_voice))]
            voice_sums_before_last = [-1 for v in range(len(self.total_duration_constraint_per_voice))]
            for v in sorted(list(set(assignment.values()))):
                # for every voice which has been assigned notes, be sure the duration constraint per voice
                # can be satisfied using the 'cutoff trick'
                # has not yet been violated
                # use order of notes in assignment
                subset = {k: v_i for k, v_i in assignment.items() if v_i == v}
                ordered_subset_keys = sorted(subset.keys(), key=lambda x: x[0])

                voice_sum = 0
                for k in ordered_subset_keys:
                    # skip marker symbols
                    if k[1][1] <= 0:
                        assert k[1][0] < 0
                    voice_sum += k[1][1] if k[1][1] > 0 else 0
                voice_sums[v] = voice_sum
                voice_sums_before_last[v] = voice_sum - ordered_subset_keys[-1][1][1]
            print(assignment)
            from IPython import embed; embed(); raise ValueError()

            for v in sorted(list(set(assignment.values()))):
                # make sure one more note could fit, even if it has to be cut by hand
                # -1 denotes a non used voice, skip it
                if voice_sums_before_last[v] > -1 and voice_sums_before_last[v] > (self.total_duration_constraint_per_voice[v] - min(durations)):
                    return False
            return True
            """

    class PairConstraint(Constraint):
        def __init__(self, voice1: int, voice2: int, notes: List[tuple], total_duration_constraint_per_voice: List[float]) -> None:
            super().__init__([voice1, voice2])
            self.voice1 = voice1
            self.voice2 = voice2
            self.notes = notes
            self.total_duration_constraint_per_voice = total_duration_constraint_per_voice
            self.active_voice_set = [_n for _n, d in enumerate(total_duration_constraint_per_voice) if d > 0]

        def satisfied(self, assignment: Dict[int, List[tuple]]) -> bool:
            if self.voice1 not in assignment or self.voice2 not in assignment:
                # cant check this one unless 2 voices in assignment
                return True
            notes1 = assignment[self.voice1]
            notes2 = assignment[self.voice2]
            for notes2_i in notes2:
                if notes2_i in notes1:
                    return False
            return True

    class AllConstraint(Constraint):
        def __init__(self, list_of_voices: List[int], notes: List[tuple], total_duration_constraint_per_voice: List[float]) -> None:
            super().__init__(list_of_voices)
            self.list_of_voices = list_of_voices
            self.notes = notes
            self.total_duration_constraint_per_voice = total_duration_constraint_per_voice
            self.active_voice_set = [_n for _n, d in enumerate(total_duration_constraint_per_voice) if d > 0]

        def satisfied(self, assignment: Dict[int, List[tuple]]) -> bool:
            for _v in self.list_of_voices:
                 if _v not in assignment:
                     # can't check til we hit max depth
                     return True
            # check that all notes have been assigned
            # should be correct by construction but eh
            all_assigned = {}
            for _k, _v in assignment.items():
                for _n in _v:
                    all_assigned[_n] = None
            # build lookup dict for this check
            # maybe overkill
            for _n in self.notes:
                if _n not in all_assigned:
                    return False
            return True

    def sub_search_depth(infill_corpus,
                         current_batch, current_batch_masks,
                         current_log_probs,
                         current_voice_possibility_map,
                         total_duration_constraint_per_voice,
                         context_boundary,
                         gt_offsets,
                         gt_answers,
                         list_pos,
                         mask_pos,
                         temperature,
                         p_cutoff,
                         depth=0):

        print("WRONGO")
        from IPython import embed; embed(); raise ValueError()
        # check constraint right here
        # if we match the relaxed constraint, return the batch as is!
        if depth >= len(current_voice_possibility_map.keys()):
            # look at tentative voice assignment to get duration constraint
            active_voice_set = [int(v) for v in list(set(gt_offsets[list_pos][:, 0]))]
            # just -depth because the first token is the context token
            this_answer = [a for a in current_batch[-(depth):][:, 0, 0].cpu().data.numpy()]

            variables = copy.deepcopy(active_voice_set)
            # note index as a unique entity, note tuple of pitch, duration
            notes = [(_n, infill_corpus.dictionary.idx2word[t_a]) for _n, t_a in enumerate(this_answer)]
            # voices are variables
            # domains are notes
            # add constraints that check every voice against every other voice to ensure:
            # no reused notes
            # duration constraints
            # style?
            # generate all combinations of notes... gonna be rough
            domains = {}
            for variable in variables:
                all_combs = []
                for _l in range(len(notes) - len(active_voice_set) + 1):
                    # range starts from 0, combinations start from 1
                    combs = list(itertools.combinations(notes, _l + 1))
                    all_combs.extend(combs)
                domains[variable] = all_combs

            partial_csp = CSP(variables, domains)
            # one against 2 3 4
            # two against 3 4
            # three against 4
            for _n, _v in enumerate(active_voice_set):
                partial_csp.add_constraint(SelfConstraint(_v, notes, total_duration_constraint_per_voice))

            for _n1, _v1 in enumerate(active_voice_set):
                # check 0 against 1, 2, 3
                # 1 against 2, 3
                # 2 against 3
                comparisons = active_voice_set
                comparisons = [_v2 for _v2 in active_voice_set if _v2 > _v1]
                for _v2 in comparisons:
                    partial_csp.add_constraint(PairConstraint(_v1, _v2, notes, total_duration_constraint_per_voice))

            partial_csp.add_constraint(AllConstraint(active_voice_set, notes, total_duration_constraint_per_voice))
            solution = partial_csp.backtracking_search()
            if solution is not None:
                print("solve work")
                return current_batch, solution, depth

        # TODO: un hard code, get from corpus max ngram value
        if depth >= 7:
            return None
            #return current_batch

        # get them ordered from most probable to least
        # stop at top k?
        # should this be beam prob or single step...
        # should this be *full* probs?
        # here do depth + 1 because we do care about making the "separator" token more likely
        sumlogprob = np.sum(current_log_probs[-(depth + 1):], axis=0)
        sumlogprob_ordered_indices = np.argsort(sumlogprob[0, :])[::-1]

        # TODO: batch this in chunks of batch_size!
        #print("depth")
        #print(depth)
        # do the 10x speedup here?
        for el in sumlogprob_ordered_indices:
            symbol = infill_corpus.dictionary.idx2word[el]
            if symbol in infill_corpus.special_symbols:
                # skip special tokens...
                continue

            min_check = sumlogprob[0, el] < -1E9
            if min_check:
                print("min prob occured at depth {}".format(depth))
                return None

            # set uniform probs for now?
            sub_batch = torch.cat((current_batch, current_batch[:1] * 0 + el), axis=0)
            sub_batch_masks = torch.cat((current_batch_masks, current_batch_masks[:1] * 0), axis=0)

            out, out_mems = model(sub_batch, sub_batch_masks, list_of_mems=None)

            log_p = out.cpu().data.numpy()

            sub_log_probs = copy.deepcopy(log_p)

            # apply temperature here to avoid numerical trouble dividing neginf
            log_p_t = log_p

            # create set of all possible durations left in possibles...
            # don't do this
            #duration_constraint_set = functools.reduce(lambda a,b: a | b, [set([eli for el in current_voice_possibility_map[v_i] for eli in el]) for v_i in current_voice_possibility_map.keys()])

            #invalid_indices = [_s for _s in range(len(infill_corpus.dictionary.idx2word))
                               #if infill_corpus.dictionary.idx2word[_s][1] not in duration_constraint_set]
            # also block answer token?
            #log_p_t[:, :, invalid_indices] = np.min(log_p_t)
            #
            # top_p reduction as normal, but we cannot do top_p if we catch a loop where answer is way more probable
            # than continuation
            # long term mebbe switch from top_p to top_k for this case
            # top p is eliminating everything but 1 element
            #reduced, reduced_mask = top_p_from_logits_np(log_p_t, 0.00001, return_indices=True)

            reduced = log_p_t * (1. / temperature)
            reduced_probs = softmax_np(reduced)

            # create new voice possibility map...
            sub_voice_possibility_map = copy.deepcopy(current_voice_possibility_map)

            sub_res = sub_search(infill_corpus,
                                 sub_batch, sub_batch_masks,
                                 sub_log_probs,
                                 sub_voice_possibility_map,
                                 total_duration_constraint_per_voice,
                                 context_boundary,
                                 gt_offsets,
                                 gt_answers,
                                 list_pos,
                                 mask_pos,
                                 temperature,
                                 p_cutoff,
                                 depth=depth + 1)

            if sub_res is not None:
                return sub_res
        print("failed subset search")
        from IPython import embed; embed(); raise ValueError()
        raise ValueError("No solutions found in recusive sub_search...")

    def sub_search_pruned_depth(infill_corpus,
                                current_batch, current_batch_masks,
                                current_log_probs,
                                current_voice_possibility_map,
                                total_duration_constraint_per_voice,
                                context_boundary,
                                gt_offsets,
                                gt_answers,
                                list_pos,
                                mask_pos,
                                temperature,
                                p_cutoff,
                                depth=0,
                                debug=False):
        #print(list_pos)
        _i = list_pos

        for vi in sorted(set(gt_offsets[_i][:, 0])):
            this_gt_offset = gt_offsets[_i][np.where(gt_offsets[_i][:, 0] == vi)[0]]
            start_offset_t = this_gt_offset[0, 1]
            last_offset_t = this_gt_offset[-1, 1]
            last_dur_t = this_gt_offset[-1, -1]
            #print("gt", vi, start_offset_t, last_offset_t, last_dur_t, last_offset_t + last_dur_t)

        if list_pos > 0:
            pass
            #print(total_duration_constraint_per_voice)
            #from IPython import embed; embed(); raise ValueError()

        # need to check *before* doing the sub_res part
        # check all possibles before going deeper!
        # only keep those that could possibly be solutions
        # delete intermediates instantly

        # check constraint right here
        # if we match the relaxed constraint, return the batch as is!
        # do constraint check inside loop...
        # that way we terminate immediately if any proposal is a solution
        if depth >= len(current_voice_possibility_map.keys()):
            # look at tentative voice assignment to get duration constraint
            active_voice_set = [int(v) for v in list(set(gt_offsets[list_pos][:, 0]))]
            # just -depth because the first token is the context token
            this_answer = [int(a) for a in current_batch[-(depth):][:, 0, 0].cpu().data.numpy()]

            variables = copy.deepcopy(active_voice_set)
            # note index as a unique entity, note tuple of pitch, duration
            notes = [(_n, infill_corpus.dictionary.idx2word[t_a]) for _n, t_a in enumerate(this_answer)]
            # voices are variables
            # domains are notes
            # add constraints that check every voice against every other voice to ensure:
            # no reused notes
            # duration constraints
            # style?
            # generate all combinations of notes... gonna be rough
            domains = {}
            for variable in variables:
                all_combs = []
                for _l in range(len(notes) - len(active_voice_set) + 1):
                    # range starts from 0, combinations start from 1
                    combs = list(itertools.combinations(notes, _l + 1))
                    all_combs.extend(combs)
                domains[variable] = all_combs

            partial_csp = CSP(variables, domains)
            # one against 2 3 4
            # two against 3 4
            # three against 4
            for _n, _v in enumerate(active_voice_set):
                partial_csp.add_constraint(SelfConstraint(_v, notes, total_duration_constraint_per_voice))

            for _n1, _v1 in enumerate(active_voice_set):
                # check 0 against 1, 2, 3
                # 1 against 2, 3
                # 2 against 3
                comparisons = active_voice_set
                comparisons = [_v2 for _v2 in active_voice_set if _v2 > _v1]
                for _v2 in comparisons:
                    partial_csp.add_constraint(PairConstraint(_v1, _v2, notes, total_duration_constraint_per_voice))

            partial_csp.add_constraint(AllConstraint(active_voice_set, notes, total_duration_constraint_per_voice))
            solution = partial_csp.backtracking_search()

            if solution is not None:
                print("solve work")
                return current_batch, solution, depth

        # TODO: un hard code, get from corpus max ngram value
        if depth >= 9:
            return None
            #return current_batch

        # get them ordered from most probable to least
        # stop at top k?
        # should this be beam prob or single step...
        # should this be *full* probs?
        # here do depth + 1 because we do care about making the "separator" token more likely
        sumlogprob = np.sum(current_log_probs[-(depth + 1):], axis=0)
        sumlogprob_ordered_indices = np.argsort(sumlogprob[0, :])[::-1]

        # remove some indices...
        reduced_indices = []
        for el in sumlogprob_ordered_indices:
            symbol = infill_corpus.dictionary.idx2word[el]

            active_voice_set = [(_n, av) for _n, av in enumerate(total_duration_constraint_per_voice) if av > 0]
            # skip special tokens
            # skip tokens which clearly violate constraints
            if depth > 0:
                # skip context token, by not doing +1
                answer_so_far_tokens = current_batch[-(depth):, 0]
                answer_so_far_tokens = [a for a in answer_so_far_tokens.cpu().data.numpy().astype("int32").flat]
                answer_so_far_symbols = [infill_corpus.dictionary.idx2word[a] for a in answer_so_far_tokens]

            # can skip a lot of pitches by adding smoothness constraint
            # aka, dont use pitches that are 2 octaves above or below the maximum extent

            if symbol[1] > max(total_duration_constraint_per_voice):
                # if the duration is larger than any possible assignment can hold, already wrong
                pass
            elif symbol in infill_corpus.special_symbols:
                # don't do special symbols for now
                pass
            elif depth > len(active_voice_set):
                existing_durations = np.array([tup[1] for tup in answer_so_far_symbols])
                # if we cannot add this note onto ANY other note, and there are more notes than voices
                # it cannot work (pigeonhole principle)
                if min(existing_durations + symbol[1]) > max(total_duration_constraint_per_voice):
                    pass
                else:
                    reduced_indices.append(el)
            else:
                reduced_indices.append(el)

        #if debug:
            #print("JTKAEJFKA")
            #print("bdepth")
            #print(depth)
        #    if depth >= 5:
        #        tt = [infill_corpus.dictionary.idx2word[el] for el in reduced_indices]
        #        print("debug in prune")
        #        from IPython import embed; embed(); raise ValueError()
        print("depth", depth)

        # use reduced indices instead of originals
        sumlogprob_ordered_indices = np.array(reduced_indices)
        extra_sz = sumlogprob_ordered_indices.shape[-1] % current_log_probs.shape[1]

        if extra_sz != 0:
            pad_sz = current_log_probs.shape[1] - extra_sz
            # set pad to -1
            sumlogprob_ordered_indices = np.concatenate((sumlogprob_ordered_indices, sumlogprob_ordered_indices[:pad_sz]))

        # pad so we can evaluate probs in batches
        batched_indices = sumlogprob_ordered_indices.reshape((sumlogprob_ordered_indices.shape[0] // current_log_probs.shape[1], current_log_probs.shape[1]))

        results_data = {}
        for _bi in batched_indices:

            """
            # skip this...
            min_check = sumlogprob[0, el] < -1E9
            if min_check:
                print("min prob occured at depth {}".format(depth))
                return None
            """

            # set uniform probs for now?
            sub_batch = torch.cat((current_batch, current_batch[:1] * 0 + _bi[None, :, None]), axis=0)
            sub_batch_masks = torch.cat((current_batch_masks, current_batch_masks[:1] * 0), axis=0)
            out, out_mems = model(sub_batch, sub_batch_masks, list_of_mems=None)

            log_p = out.cpu().data.numpy()

            sub_log_probs = copy.deepcopy(log_p)

            # apply temperature here to avoid numerical trouble dividing neginf
            log_p_t = log_p

            # create set of all possible durations left in possibles...
            # don't do this
            #duration_constraint_set = functools.reduce(lambda a,b: a | b, [set([eli for el in current_voice_possibility_map[v_i] for eli in el]) for v_i in current_voice_possibility_map.keys()])

            #invalid_indices = [_s for _s in range(len(infill_corpus.dictionary.idx2word))
                               #if infill_corpus.dictionary.idx2word[_s][1] not in duration_constraint_set]
            # also block answer token?
            #log_p_t[:, :, invalid_indices] = np.min(log_p_t)
            #
            # top_p reduction as normal, but we cannot do top_p if we catch a loop where answer is way more probable
            # than continuation
            # long term mebbe switch from top_p to top_k for this case
            # top p is eliminating everything but 1 element
            #reduced, reduced_mask = top_p_from_logits_np(log_p_t, 0.00001, return_indices=True)

            reduced = log_p_t * (1. / temperature)
            reduced_probs = softmax_np(reduced)

            # create new voice possibility map...
            sub_voice_possibility_map = copy.deepcopy(current_voice_possibility_map)

            for el_index, el in enumerate(_bi):
                if el in results_data:
                    # if already seen, was a pad element
                    continue
                symbol = infill_corpus.dictionary.idx2word[el]
                #scuffed, but these should be references NOT copies. so we can keep them around
                results_data[el] = [sub_log_probs, el_index, el, sub_voice_possibility_map, sub_batch]#, sub_batch_masks]#, out, out_mems]
            del out
            del out_mems

        for _bi in batched_indices:
            for el_index, el in enumerate(_bi):
                if el not in results_data:
                    # if it is in the batched_indices but not the results (because we deleted it!), was a pad element
                    continue
                sub_log_probs_out, el_index_a, el_a, sub_voice_possibility_map, sub_batch_out = results_data[el]
                #reduced_probs, el_index, el, sub_voice_possibility_map, sub_batch, sub_batch_masks, out, out_mems = results_data[el]
                # rebroadcast it for each element
                sub_batch_fixed = 0. * sub_batch_out + sub_batch_out[:, el_index, :][:, None]
                sub_log_probs_fixed = 0. * sub_log_probs_out + sub_log_probs_out[:, el_index, :][:, None]
                sub_batch_masks_fixed = 0. * sub_batch_fixed[..., 0]

                if _i > 0:
                    tmp_answer_so_far_tokens = current_batch[context_boundary:, 0]
                    tmp_answer_so_far_tokens = [a for a in tmp_answer_so_far_tokens.cpu().data.numpy().astype("int32").flat]
                    tmp_answer_so_far_symbols = [infill_corpus.dictionary.idx2word[a] for a in tmp_answer_so_far_tokens]

                sub_res = sub_search_pruned_depth(infill_corpus,
                                                  sub_batch_fixed, sub_batch_masks_fixed,
                                                  sub_log_probs_fixed,
                                                  sub_voice_possibility_map,
                                                  total_duration_constraint_per_voice,
                                                  context_boundary,
                                                  gt_offsets,
                                                  gt_answers,
                                                  list_pos,
                                                  mask_pos,
                                                  temperature,
                                                  p_cutoff,
                                                  depth=depth + 1,
                                                  debug=debug)

                if sub_res is not None:
                    return sub_res
                del sub_log_probs_out
                del sub_batch_fixed
                del sub_log_probs_fixed
                del sub_batch_out
                del results_data[el]

        #print("failed subset search")
        #from IPython import embed; embed(); raise ValueError()
        #raise ValueError("No solutions found in recusive sub_search...")

    def sub_search_pruned_breadth(infill_corpus,
                                  current_batch, current_batch_masks,
                                  current_log_probs,
                                  current_voice_possibility_map,
                                  total_duration_constraint_per_voice,
                                  context_boundary,
                                  gt_offsets,
                                  gt_answers,
                                  list_pos,
                                  mask_pos,
                                  temperature,
                                  p_cutoff,
                                  depth=0,
                                  max_depth=9,
                                  debug=False):
        #print(list_pos)
        _i = list_pos

        for vi in sorted(set(gt_offsets[_i][:, 0])):
            this_gt_offset = gt_offsets[_i][np.where(gt_offsets[_i][:, 0] == vi)[0]]
            start_offset_t = this_gt_offset[0, 1]
            last_offset_t = this_gt_offset[-1, 1]
            last_dur_t = this_gt_offset[-1, -1]
            #print("gt", vi, start_offset_t, last_offset_t, last_dur_t, last_offset_t + last_dur_t)

        if list_pos > 0:
            pass
            #print(total_duration_constraint_per_voice)
            #from IPython import embed; embed(); raise ValueError()

        # need to check *before* doing the sub_res part
        # check all possibles before going deeper!
        # only keep those that could possibly be solutions
        # delete intermediates instantly

        # check constraint right here
        # if we match the relaxed constraint, return the batch as is!
        # do constraint check inside loop...
        # that way we terminate immediately if any proposal is a solution
        # TODO: un hard code, get from corpus max ngram value
        if depth >= max_depth:
            return None
            #return current_batch

        # get them ordered from most probable to least
        # stop at top k?
        # should this be beam prob or single step...
        # should this be *full* probs?
        # here do depth + 1 because we do care about making the "separator" token more likely
        sumlogprob = np.sum(current_log_probs[-(depth + 1):], axis=0)
        sumlogprob_ordered_indices = np.argsort(sumlogprob[0, :])[::-1]

        # remove some indices...
        reduced_indices = []
        for el in sumlogprob_ordered_indices:
            symbol = infill_corpus.dictionary.idx2word[el]

            active_voice_set = [(_n, av) for _n, av in enumerate(total_duration_constraint_per_voice) if av > 0]
            # skip special tokens
            # skip tokens which clearly violate constraints
            if depth > 0:
                # skip context token, by not doing +1
                answer_so_far_tokens = current_batch[-(depth):, 0]
                answer_so_far_tokens = [a for a in answer_so_far_tokens.cpu().data.numpy().astype("int32").flat]
                answer_so_far_symbols = [infill_corpus.dictionary.idx2word[a] for a in answer_so_far_tokens]

            # can skip a lot of pitches by adding smoothness constraint
            # aka, dont use pitches that are 2 octaves above or below the maximum extent

            if symbol[1] > max(total_duration_constraint_per_voice):
                # if the duration is larger than any possible assignment can hold, already wrong
                pass
            elif symbol in infill_corpus.special_symbols:
                # don't do special symbols for now
                pass
            elif depth > len(active_voice_set):
                existing_durations = np.array([tup[1] for tup in answer_so_far_symbols])
                # if we cannot add this note onto ANY other note, and there are more notes than voices
                # it cannot work (pigeonhole principle)
                if min(existing_durations + symbol[1]) > max(total_duration_constraint_per_voice):
                    pass
                else:
                    reduced_indices.append(el)
            else:
                reduced_indices.append(el)


        # do the check here for each value, if any of them hit return
        active_voice_set = [_n for _n, av in enumerate(total_duration_constraint_per_voice) if av > 0]
        if depth >= len(active_voice_set):
            # look at tentative voice assignment to get duration constraint
            # just -depth because the first token is the context token
            this_answer_global = [int(a) for a in current_batch[-(depth):][:, 0, 0].cpu().data.numpy()]
            not_exhausted = True
            ii = 0
            while not_exhausted:
                el = reduced_indices[ii]
                this_answer = this_answer_global + [el]
                variables = copy.deepcopy(active_voice_set)
                # note index as a unique entity, note tuple of pitch, duration
                notes = [(_n, infill_corpus.dictionary.idx2word[t_a]) for _n, t_a in enumerate(this_answer)]
                # voices are variables
                # domains are notes
                # add constraints that check every voice against every other voice to ensure:
                # no reused notes
                # duration constraints
                # style?
                # generate all combinations of notes... gonna be rough
                domains = {}
                for variable in variables:
                    all_combs = []
                    for _l in range(len(notes) - len(active_voice_set) + 1):
                        # range starts from 0, combinations start from 1
                        combs = list(itertools.combinations(notes, _l + 1))
                        all_combs.extend(combs)
                    domains[variable] = all_combs

                partial_csp = CSP(variables, domains)
                # one against 2 3 4
                # two against 3 4
                # three against 4
                for _n, _v in enumerate(active_voice_set):
                    partial_csp.add_constraint(SelfConstraint(_v, notes, total_duration_constraint_per_voice))

                for _n1, _v1 in enumerate(active_voice_set):
                    # check 0 against 1, 2, 3
                    # 1 against 2, 3
                    # 2 against 3
                    comparisons = active_voice_set
                    comparisons = [_v2 for _v2 in active_voice_set if _v2 > _v1]
                    for _v2 in comparisons:
                        partial_csp.add_constraint(PairConstraint(_v1, _v2, notes, total_duration_constraint_per_voice))

                partial_csp.add_constraint(AllConstraint(active_voice_set, notes, total_duration_constraint_per_voice))
                solution = partial_csp.backtracking_search()

                if solution is not None:
                    # print("solve work")
                    res_batch = torch.cat((current_batch, current_batch[:1] * 0 + el))
                    return res_batch, solution, depth
                else:
                    # if this duration failed, we can do more pruning
                    el_sym = infill_corpus.dictionary.idx2word[el]
                    #dur_matches = [_j for _j in range(len(reduced_indices)) if infill_corpus.dictionary.idx2word[reduced_indices[_j]] == el_sym[1]]
                    # this is really aggressive pruning, only use the most likely candidate for a duration!
                    dur_matches = [infill_corpus.dictionary.idx2word[reduced_indices[_j]] for _j in range(len(reduced_indices))]
                    non_matches = [_j for _j, d in enumerate(dur_matches) if d[1] != el_sym[1]]
                    reduced_indices = [reduced_indices[nm] for nm in non_matches]
                ii += 1
                if ii >= len(reduced_indices):
                    not_exhausted = False

        #if debug:
            #print("JTKAEJFKA")
            #print("bdepth")
            #print(depth)
        #    if depth >= 5:
        #        tt = [infill_corpus.dictionary.idx2word[el] for el in reduced_indices]
        #        print("debug in prune")
        #        from IPython import embed; embed(); raise ValueError()
        print("depth", depth, "max_depth", max_depth)

        # use reduced indices instead of originals
        sumlogprob_ordered_indices = np.array(reduced_indices)
        extra_sz = sumlogprob_ordered_indices.shape[-1] % current_log_probs.shape[1]

        if extra_sz != 0:
            pad_sz = current_log_probs.shape[1] - extra_sz
            # set pad to -1
            sumlogprob_ordered_indices = np.concatenate((sumlogprob_ordered_indices, sumlogprob_ordered_indices[:pad_sz]))

        # pad so we can evaluate probs in batches
        batched_indices = sumlogprob_ordered_indices.reshape((sumlogprob_ordered_indices.shape[0] // current_log_probs.shape[1], current_log_probs.shape[1]))

        results_data = {}
        for _bi in batched_indices:

            """
            # skip this...
            min_check = sumlogprob[0, el] < -1E9
            if min_check:
                print("min prob occured at depth {}".format(depth))
                return None
            """

            # set uniform probs for now?
            sub_batch = torch.cat((current_batch, current_batch[:1] * 0 + _bi[None, :, None]), axis=0)
            sub_batch_masks = torch.cat((current_batch_masks, current_batch_masks[:1] * 0), axis=0)
            out, out_mems = model(sub_batch, sub_batch_masks, list_of_mems=None)

            log_p = out.cpu().data.numpy()

            sub_log_probs = copy.deepcopy(log_p)

            # apply temperature here to avoid numerical trouble dividing neginf
            log_p_t = log_p

            # create set of all possible durations left in possibles...
            # don't do this
            #duration_constraint_set = functools.reduce(lambda a,b: a | b, [set([eli for el in current_voice_possibility_map[v_i] for eli in el]) for v_i in current_voice_possibility_map.keys()])

            #invalid_indices = [_s for _s in range(len(infill_corpus.dictionary.idx2word))
                               #if infill_corpus.dictionary.idx2word[_s][1] not in duration_constraint_set]
            # also block answer token?
            #log_p_t[:, :, invalid_indices] = np.min(log_p_t)
            #
            # top_p reduction as normal, but we cannot do top_p if we catch a loop where answer is way more probable
            # than continuation
            # long term mebbe switch from top_p to top_k for this case
            # top p is eliminating everything but 1 element
            #reduced, reduced_mask = top_p_from_logits_np(log_p_t, 0.00001, return_indices=True)

            reduced = log_p_t * (1. / temperature)
            reduced_probs = softmax_np(reduced)

            # create new voice possibility map...
            sub_voice_possibility_map = copy.deepcopy(current_voice_possibility_map)

            for el_index, el in enumerate(_bi):
                if el in results_data:
                    # if already seen, was a pad element
                    continue
                symbol = infill_corpus.dictionary.idx2word[el]
                #scuffed, but these should be references NOT copies. so we can keep them around
                results_data[el] = [sub_log_probs, el_index, el, sub_voice_possibility_map, sub_batch]#, sub_batch_masks]#, out, out_mems]
            del out
            del out_mems

        for _bi in batched_indices:
            for el_index, el in enumerate(_bi):
                if el not in results_data:
                    # if it is in the batched_indices but not the results (because we deleted it!), was a pad element
                    continue
                sub_log_probs_out, el_index_a, el_a, sub_voice_possibility_map, sub_batch_out = results_data[el]
                #reduced_probs, el_index, el, sub_voice_possibility_map, sub_batch, sub_batch_masks, out, out_mems = results_data[el]
                # rebroadcast it for each element
                sub_batch_fixed = 0. * sub_batch_out + sub_batch_out[:, el_index, :][:, None]
                sub_log_probs_fixed = 0. * sub_log_probs_out + sub_log_probs_out[:, el_index, :][:, None]
                sub_batch_masks_fixed = 0. * sub_batch_fixed[..., 0]

                if _i > 0:
                    tmp_answer_so_far_tokens = current_batch[context_boundary:, 0]
                    tmp_answer_so_far_tokens = [a for a in tmp_answer_so_far_tokens.cpu().data.numpy().astype("int32").flat]
                    tmp_answer_so_far_symbols = [infill_corpus.dictionary.idx2word[a] for a in tmp_answer_so_far_tokens]

                sub_res = sub_search_pruned_breadth(infill_corpus,
                                                    sub_batch_fixed, sub_batch_masks_fixed,
                                                    sub_log_probs_fixed,
                                                    sub_voice_possibility_map,
                                                    total_duration_constraint_per_voice,
                                                    context_boundary,
                                                    gt_offsets,
                                                    gt_answers,
                                                    list_pos,
                                                    mask_pos,
                                                    temperature,
                                                    p_cutoff,
                                                    depth=depth + 1,
                                                    max_depth=max_depth,
                                                    debug=debug)

                if sub_res is not None:
                    #if debug:
                    #    print("debug solve")
                    #    from IPython import embed; embed(); raise ValueError()
                    return sub_res
                del sub_log_probs_out
                del sub_batch_fixed
                del sub_log_probs_fixed
                del sub_batch_out
                del results_data[el]

        #print("failed subset search")
        #from IPython import embed; embed(); raise ValueError()
        #raise ValueError("No solutions found in recusive sub_search...")


    # need to start full search here...
    def full_search(infill_corpus, input_batch, batch_masks,
                    batch_index, context_boundary,
                    mask_positions, gt_offsets, gt_answers):
        current_boundary = context_boundary
        current_batch = input_batch
        current_batch_masks = batch_masks


        all_answers = []
        all_answers_durations = []
        all_answers_voices = []
        all_answers_voices = []
        all_res = []
        for list_pos, mask_pos in enumerate(mask_positions):
            # set of voices used in this subproblem
            active_voice_set = [int(v) for v in sorted(list(set(gt_offsets[list_pos][:, 0])))]

            # total durations per voice
            total_duration_constraint_per_voice = [0, 0, 0, 0]
            for _t in range(4):
                voice_match = np.where(gt_offsets[list_pos][:, 0] == _t)[0]
                if len(voice_match) == 0:
                    this_voice_dur = 0
                else:
                    this_voice_dur = gt_offsets[list_pos][voice_match].sum(axis=0)[-1]
                total_duration_constraint_per_voice[_t] = this_voice_dur

            if list_pos > 0:
                _i = list_pos
                #print(_i)

                for vi in sorted(set(gt_offsets[_i][:, 0])):
                    this_gt_offset = gt_offsets[_i][np.where(gt_offsets[_i][:, 0] == vi)[0]]
                    start_offset_t = this_gt_offset[0, 1]
                    last_offset_t = this_gt_offset[-1, 1]
                    last_dur_t = this_gt_offset[-1, -1]
                    #print("gt", vi, start_offset_t, last_offset_t, last_dur_t, last_offset_t + last_dur_t)
                #print(total_duration_constraint_per_voice)
                #from IPython import embed; embed(); raise ValueError()

            # set of all durations
            durations = sorted(set([k[1] for k in infill_corpus.dictionary.word2idx.keys() if k[1] > 0]))

            # need at least active_voice_set values
            # cannot use a single note longer than the whole duration constraint
            # when creating multiple, use existing duration to further eliminate
            # we also know the total maximum infilling size is 8
            all_valid = []
            # can only have at most 8 - (n_voices - 1) notes in a voice (+1 due to range not including last entry)
            # due to pigeonhole principle
            for l in range(1, max([9, 9 - (len(active_voice_set) - 1)])):
                sub_durations = [d for d in durations if d <= max(total_duration_constraint_per_voice)]
                # is this generating all permutations as well?
                all_reduced = [ar for ar in itertools.combinations_with_replacement(sub_durations, l)]

                # reduce per voice based on constraint
                valid_per_voice = []
                for v_i, _ in enumerate(total_duration_constraint_per_voice):
                    if total_duration_constraint_per_voice[v_i] == 0.:
                        valid_per_voice.append(None)
                    else:
                        new_valids = [ar_i for ar_i in all_reduced if sum(ar_i) == total_duration_constraint_per_voice[v_i]]
                        valid_per_voice.append(new_valids)
                all_valid.append(valid_per_voice)

            # all valid is list of list, first list is length indexed (1 through 8) second list is voice indexed (0-4)
            # loop through and be sure at least 1 option works for each voice?
            voice_possibility_map = {}
            for _v in all_valid:
                for  _v_i, _vv in enumerate(_v):
                    if _vv is not None:
                        if _v_i not in voice_possibility_map:
                            voice_possibility_map[_v_i] = []
                        if len(_vv) > 0:
                            voice_possibility_map[_v_i].extend(_vv)

            assert all([len(v) > 0 for k, v in voice_possibility_map.items()])

            this_answer = []
            this_answer_durations = []
            this_answer_voices = []

            reset_current_boundary = current_boundary
            n_resets = 0
            reset_limit = 100
            reset_voice_possibility_map = copy.deepcopy(voice_possibility_map)

            duration_constraint = max(total_duration_constraint_per_voice)
            # we just ignore the answer token altogether here
            # not ideal but we know enough to start/stop 

            out, out_mems = model(current_batch[:current_boundary], batch_masks[:current_boundary], list_of_mems=None)
            temperature = args.temperature
            p_cutoff = args.p_cutoff
            # *must* duration constrain and voice assign these
            # can either force same amount of notes, and thus identical durations for each
            # or just force the overall answer to be the same total duration
            # would do greedy same durations for now, but need to move to constrained beam search

            # first, set probability of any predictions > total_duration_constraint to -inf
            log_p = out.cpu().data.numpy()

            # apply temperature here to avoid numerical trouble dividing neginf
            log_p_t = log_p

            current_log_probs = copy.deepcopy(log_p)

            # create set of all possible durations left in possibles...
            #duration_constraint_set = functools.reduce(lambda a,b: a | b, [set([eli for el in voice_possibility_map[v_i] for eli in el]) for v_i in voice_possibility_map.keys()])

            #invalid_indices = [_s for _s in range(len(infill_corpus.dictionary.idx2word))
                               #if infill_corpus.dictionary.idx2word[_s][1] not in duration_constraint_set]
            # also block answer token?
            #log_p_t[:, :, invalid_indices] = np.min(log_p_t)
            #
            # top_p reduction as normal, but we cannot do top_p if we catch a loop where answer is way more probable
            # than continuation
            # long term mebbe switch from top_p to top_k for this case
            # top p is eliminating everything but 1 element
            #reduced, reduced_mask = top_p_from_logits_np(log_p_t, 0.00001, return_indices=True)

            reduced = log_p_t * (1. / temperature)
            reduced_probs = softmax_np(reduced)
            reduced_probs = reduced_probs[:, batch_index:batch_index+1]

            # take in 
            current_corpus = infill_corpus
            current_batch_feed = copy.deepcopy(current_batch)
            current_batch_masks_feed = copy.deepcopy(current_batch_masks)

            debug = True
            #if len(all_res) >= 2:
                #print("here we are?")
                #from IPython import embed; embed(); raise ValueError()
            #    debug = False
            max_depth = min(len(active_voice_set) + 3, 9)

            # the possible voice mappings
            current_voice_possibility_map = copy.deepcopy(voice_possibility_map)
            res = sub_search_pruned_breadth(infill_corpus,
                                            current_batch[:current_boundary], current_batch_masks[:current_boundary],
                                            current_log_probs,
                                            current_voice_possibility_map,
                                            total_duration_constraint_per_voice,
                                            context_boundary,
                                            gt_offsets,
                                            gt_answers,
                                            list_pos,
                                            mask_pos,
                                            temperature,
                                            p_cutoff,
                                            depth=0,
                                            max_depth=max_depth,
                                            debug=debug)

            if res is None:
                print("got no result for sub_search! debug this")
                from IPython import embed; embed(); raise ValueError()

            answer_sz = sum([len(_v) for _v in res[-2].values()])
            this_answer = [int(ta) for ta in res[0][-answer_sz:, 0, 0].cpu().data.numpy()]
            this_answer_tokens = [(_n, infill_corpus.dictionary.idx2word[ta]) for _n, ta in enumerate(this_answer)]
            this_answer_durations = [tat[1][1] for tat in this_answer_tokens]
            
            print("broke the loop")
            this_answer_voices = []
            for tat in this_answer_tokens:
                for _k, _assigned in res[1].items():
                    if tat in _assigned:
                        this_answer_voices.append(_k)
            assert len(this_answer_voices)

            all_answers.append(this_answer)
            all_answers_durations.append(this_answer_durations)
            all_answers_voices.append(this_answer_voices)

            current_batch = res[0]
            # last token
            # infill_corpus.dictionary.idx2word[int(current_batch[:, 0, 0][-1].cpu().data.numpy().astype("int32"))]
            ans = infill_corpus.dictionary.word2idx[infill_corpus.answer_symbol]
            # add on the answer token manually
            current_batch = torch.cat((current_batch, current_batch[:1] * 0 + ans), axis=0)
            current_batch_masks = torch.cat((current_batch_masks, current_batch_masks[:1] * 0), axis=0)
            current_boundary = len(current_batch)
            print("partial answer {} of {} completed".format(len(all_answers), len(mask_positions)))

            all_res.append((current_batch, res[1], res[2]))
        print("end")
        return all_answers, all_answers_durations, all_answers_voices, all_res

    # only send the minibatch element in question to the search procedure
    batch_index = t
    input_batch = input_batch[:1] * 0 + input_batch[:, t:t+1]
    batch_masks = batch_masks[:1] * 0 + batch_masks[:, t:t+1]

    try:
        import cPickle as pickle
    except ModuleNotFoundError:
        import pickle

    fname, step = infill_corpus.valid_files_attribution[batch_file_indices[batch_index]]
    cache_path = '{}_tmp_{}.pkl'.format("-".join(fname.split(os.sep)[1:]), step)
    if not os.path.exists(cache_path):
        import time
        start_time = time.time()
        returned = full_search(infill_corpus, input_batch, batch_masks,
                          batch_index, context_boundary,
                          mask_positions, gt_offsets, gt_answers)
        end_time = time.time()
        print("returned")
        print("elapsed", end_time - start_time)
        print("save")
        start_time = time.time()
        with open(cache_path, 'wb') as f:
            pickle.dump(returned, f, pickle.HIGHEST_PROTOCOL)
        end_time = time.time()
        print("cached", end_time - start_time)
    else:
        import time
        start_time = time.time()
        with open(cache_path, 'rb') as f:
            returned = pickle.load(f)
        end_time = time.time()
        print("loaded", end_time - start_time)

    all_answers, all_answers_durations, all_answers_voices, all_res = returned
    pred_durs = [sum(aad) for aad in all_answers_durations]
    gt_durs = [gto.sum(axis=0)[-1] for gto in gt_offsets]
    # loose check that exact total duration sum matches
    # per voice is already checked
    assert len(pred_durs) == len(gt_durs)
    try:
        assert all([pd == gtd for pd, gtd in zip(pred_durs, gt_durs)])
    except:
        print("duration err")
        from IPython import embed; embed(); raise ValueError()

    modified_batch = copy.deepcopy(input_batch[:context_boundary].cpu().data.numpy())
    modified_offsets = copy.deepcopy(pad_batch_offsets_np[:context_boundary])
    modified_durations = modified_offsets[:, :, -1]
    modified_voices = modified_offsets[:, :, 0]
    modified_offsets = modified_offsets[:, :, 1]
    modified_durations = modified_durations[:, t][:, None] * (0 * modified_durations[:1, :] + 1)
    modified_voices = modified_voices[:, t][:, None] * (0 * modified_voices[:1, :] + 1)
    modified_offsets = modified_offsets[:, t][:, None] * (0 * modified_offsets[:1, :] + 1)
    for _i in range(len(all_answers)):
        for vi in sorted(set(gt_offsets[_i][:, 0])):
            this_gt_offset = gt_offsets[_i][np.where(gt_offsets[_i][:, 0] == vi)[0]]
            start_offset_t = this_gt_offset[0, 1]
            last_offset_t = this_gt_offset[-1, 1]
            last_dur_t = this_gt_offset[-1, -1]
            #print("gt", vi, start_offset_t, last_offset_t, last_dur_t, last_offset_t + last_dur_t)

        answer = all_answers[_i]
        answer_durations = all_answers_durations[_i]
        answer_voices = all_answers_voices[_i]

        mask_positions = np.where(modified_batch[:, t, 0] == mask_token)[0]
        # use 0 always because we are filling in! so the past mask tokens are gone now
        current_mask_position = mask_positions[0]

        pre = modified_batch[:current_mask_position]
        # broadcast hacks
        mid = np.array(answer)[:, None, None] * (0. * modified_batch[:1] + 1.)
        mid = mid.astype(pre.dtype)
        post = modified_batch[current_mask_position+1:]

        pre_d = modified_durations[:current_mask_position]
        # broadcast hacks
        mid_d = np.array(answer_durations)[:, None] * (0. * modified_batch[:1] + 1.)[:, :, 0]
        post_d = modified_durations[current_mask_position+1:]

        pre_v = modified_voices[:current_mask_position]
        mid_v = np.array(answer_voices)[:, None] * (0. * modified_batch[:1] + 1.)[:, :, 0]
        post_v = modified_voices[current_mask_position+1:]

        # recalculate offsets based on left and right hand span
        pre_o = modified_offsets[:current_mask_position]
        post_o = modified_offsets[current_mask_position+1:]
        mid_o = np.zeros_like(mid_v)
        for vi in sorted(list(set(mid_v[:, 0]))):
            last_up = [p for p in pre_o[:, 0][np.where(pre_v[:, 0]  == vi)[0]] if p > 0][-1]
            last_d = [p for p in pre_d[:, 0][np.where(pre_v[:, 0]  == vi)[0]] if p > 0][-1]
            fresh_o = last_up + last_d
            mid_o_list = [fresh_o]
            for _d in mid_d[:, 0][np.where(mid_v[:, 0]  == vi)[0]]:
                mid_o_list.append(mid_o_list[-1] + _d)
            # truncate...?
            mid_o_list = mid_o_list[:-1]
            # be sure that the assigned duration + our calculated offset matches the right hand side
            mid_d_last = [p for p in mid_d[:, 0][np.where(mid_v[:, 0]  == vi)[0]] if p > 0][-1]

            # find the next fill spot...
            next_voice_offset = current_mask_position + np.where(post_v[:, 0] == vi)[0][0]
            if _i < (len(all_answers) - 1):
                # always use 1 because we are "filling in"
                next_fill = mask_positions[1]
            else:
                # very last available in this voice, just bypass
                next_fill = -1 #current_mask_position + np.where(post_v[:, 0] == vi)[0][-1]

            if next_voice_offset < next_fill:
                try:
                    assert mid_o_list[-1] + mid_d_last == [p for p in post_o[:, 0][np.where(post_v[:, 0]  == vi)[0]] if p > 0][0]
                    #print("ok")
                    #from IPython import embed; embed(); raise ValueError()
                except:
                    print("??? shouldnt happn")
                    from IPython import embed; embed(); raise ValueError()
            else:
                # can't check this because the next thing to check "on the right" is AFTER the next fill... oyoyoy
                # have to trust our constraint setup
                # this same issue shouldn't come up 'on the left' because we fill left to right
                pass

            #print("p", vi, mid_o_list[0], mid_o_list[-1], mid_d_last, mid_o_list[-1] + mid_d_last)
            #if _i > 1:
            #    from IPython import embed; embed(); raise ValueError()

            mid_o[:, 0][np.where(mid_v[:, 0]  == vi)[0]] = np.array(mid_o_list)

        #if _i > 0:
        #    from IPython import embed; embed(); raise ValueError()

        # we only assigned into mb 0, lets re-broadcast to a full array even though it will all be the same
        mid_o = mid_o[:, 0][:, None] * (0. * mid_o[:1] + 1.)
        #if _i > 0:
            # check for blanks leaking
        #   from IPython import embed; embed(); raise ValueError()
        modified_batch = np.concatenate((pre, mid, post), axis=0)
        modified_durations = np.concatenate((pre_d, mid_d, post_d), axis=0)
        modified_voices = np.concatenate((pre_v, mid_v, post_v), axis=0)
        modified_offsets = np.concatenate((pre_o, mid_o, post_o), axis=0)

    gt_syms = [infill_corpus.dictionary.idx2word[ib] for ib in input_batch[:, 0].cpu().data.numpy().ravel()]
    if infill_corpus.file_separator_symbol in gt_syms:
        print("WARNING: file change symbol detected in groundtruth! output may be timeshifted during file change")

    # start with 0th entry
    token_sequence = modified_batch[:, t, 0]
    symbol_sequence = [infill_corpus.dictionary.idx2word[ts] for ts in token_sequence]
    duration_sequence = modified_durations[:, t]
    voice_sequence = modified_voices[:, t]
    offset_sequence = modified_offsets[:, t]

    # check again that everything is aligned
    assert len(symbol_sequence) == len(duration_sequence)
    for _n in range(len(symbol_sequence)):
        if symbol_sequence[_n][0] < 0:
            assert duration_sequence[_n] == 0.
            assert voice_sequence[_n] == -1
        elif symbol_sequence[_n][0] >= 0:
            assert symbol_sequence[_n][1] == duration_sequence[_n]

    special_symbols = [infill_corpus.mask_symbol,
                       infill_corpus.answer_symbol,
                       infill_corpus.end_context_symbol,
                       #infill_corpus.file_separator_symbol,
                       infill_corpus.fill_symbol]

    # KEEP FILE SEPARATOR SO WE CAN HANDLE IT NEXT
    reduced_indices = [n for n in range(len(symbol_sequence)) if symbol_sequence[n] not in special_symbols]

    symbol_sequence = np.array([symbol_sequence[n] for n in reduced_indices])
    duration_sequence = np.array([duration_sequence[n] for n in reduced_indices])
    voice_sequence = np.array([voice_sequence[n] for n in reduced_indices])
    offset_sequence = np.array([offset_sequence[n] for n in reduced_indices])

    # need to find this min! is srs
    # do starting chunk if > 0 offset?
    # check if the piece starts with an offset
    # if so, we account for that
    # take care because a later offset could be 0 we cross a file boundary...
    offsets_addition_step_mapping = {}
    if infill_corpus.file_separator_symbol in symbol_sequence:
        # get all file separators
        all_file_sep_pos = [0] + [_n for _n, el in enumerate(symbol_sequence) if tuple(el) == infill_corpus.file_separator_symbol]
    else:
        all_file_sep_pos = [0]

    if len(all_file_sep_pos) > 2:
        print("multiple file separators, handle it...")
        from IPython import embed; embed(); raise ValueError()

    # for each voice, get the offset we have to add to the front to make it sound right
    previous_gap = 0
    for _n, afsp in enumerate(all_file_sep_pos):
        per_voice_offset_start = []
        if _n < len(all_file_sep_pos) - 1:
            nxt = all_file_sep_pos[_n + 1]
        else:
            nxt = len(voice_sequence)
        all_voice_indices = [previous_gap + np.where(voice_sequence[previous_gap:nxt] == vi)[0] for vi in range(4)]
        for vi in range(4):
            _voice_set = np.array(offset_sequence)[all_voice_indices[vi]]
            first_val = _voice_set[0]
            min_val = np.min(_voice_set)
            if first_val != min_val:
                print("WARNING: offset calculation min val != first val!. sample likely corrupt, dropping to debug!")
                from IPython import embed; embed(); raise ValueError()
            per_voice_offset_start.append(min_val)
        calculated_offsets = [el - min(per_voice_offset_start) for el in per_voice_offset_start]
        if previous_gap != 0:
            # had an offset previously applied. we need to add *additional* time to "fix" the previous offset...
            # get the last key
            last_added = max([k for k in offsets_addition_step_mapping.keys()])
            prv = offsets_addition_step_mapping[last_added]
            # add 4.0 as a natural break, alongside the "corrective factor" to put the previous interval offsets back to even
            prv_correction = [4.0 + (max(prv) - el) for el in prv]
            # combine with new offset
            for _m in range(len(calculated_offsets)):
                calculated_offsets[_m] += prv_correction[_m]
        offsets_addition_step_mapping[previous_gap] = calculated_offsets
        # +1 to skip the separator itself
        previous_gap = nxt + 1

    splits = sorted(offsets_addition_step_mapping.keys())
    if len(splits) > 1:
        # do correction
        symbol_seps = []
        duration_seps = []
        voice_seps = []
        for a, b in zip(splits[:-1], splits[1:]):
            symbol_seps.append(symbol_sequence[a:b])
            duration_seps.append(duration_sequence[a:b])
            voice_seps.append(voice_sequence[a:b])
        symbol_seps.append(symbol_sequence[b:])
        duration_seps.append(duration_sequence[b:])
        voice_seps.append(voice_sequence[b:])
        assert len(offsets_addition_step_mapping) == len(symbol_seps)
        assert len(offsets_addition_step_mapping) == len(duration_seps)
        assert len(offsets_addition_step_mapping) == len(voice_seps)

        joined_symbols = []
        joined_voices = []

        ordered_k = sorted(offsets_addition_step_mapping.keys())

        for a, b in zip(range(len(offsets_addition_step_mapping)), range(len(symbol_seps))):
            # construct the offset chunks and append to respective list
            this_sym = symbol_seps[b]
            this_dur = duration_seps[b]
            this_voice = voice_seps[b]
            k = ordered_k[a]
            this_offset = offsets_addition_step_mapping[k]
            this_offset_voice = [_v for _v in range(len(this_offset))]
            this_offset_dur = [t_o for t_o in this_offset]
            this_offset_pitch = [0. for _v in range(len(this_offset))]
            # to ensure it is in the dictionary, make it into multiples of 1, .5, .25 etc
            rems = [t_o_d - int(t_o_d) for t_o_d in this_offset_dur]

            this_offset_spread_voice = []
            this_offset_spread_dur = []
            this_offset_spread_pitch = []

            for _n, rem in enumerate(rems):
                if rem == 0:
                    continue
                else:
                   # this assumes rem is in the dataset!
                   assert rem in [.125, .25, .5, 1.]
                   this_offset_spread_dur.append(rem)
                   this_offset_spread_voice.append(_n)
                   this_offset_spread_pitch.append(0)

            # now that we handled remainders, stick in the rests in chunks of 1.
            cores = [int(t_o_d) for t_o_d in this_offset_dur]
            for _n, core in enumerate(cores):
                if core == 0:
                    continue
                else:
                    # core should be an integer >= 1
                    for _j in range(core):
                        this_offset_spread_voice.append(_n)
                        this_offset_spread_dur.append(1.0)
                        this_offset_spread_pitch.append(0)

            #this_offset_sym = [infill_corpus.dictionary.idx2word[
            assert len(this_offset_spread_dur) == len(this_offset_spread_pitch)
            this_offset_symbols = np.array([(el1, el2) for el1, el2 in zip(this_offset_spread_pitch, this_offset_spread_dur)])
            joined_voices.append(np.array(this_offset_spread_voice))
            joined_symbols.append(this_offset_symbols)
            joined_voices.append(this_voice)
            joined_symbols.append(this_sym)
        joined_voices = np.concatenate(joined_voices)
        joined_symbols = np.concatenate(joined_symbols)
    else:
        # TODO: unify top and bottom case, lots of repeat logic here
        # only one element, just add offset to start...
        this_sym = symbol_sequence
        this_dur = duration_sequence
        this_voice = voice_sequence
        k = list(offsets_addition_step_mapping.keys())[0]
        assert len(offsets_addition_step_mapping.keys()) == 1
        this_offset = offsets_addition_step_mapping[k]
        this_offset_voice = [_v for _v in range(len(this_offset))]
        this_offset_dur = [t_o for t_o in this_offset]
        this_offset_pitch = [0. for _v in range(len(this_offset))]
        # to ensure it is in the dictionary, make it into multiples of 1, .5, .25 etc
        rems = [t_o_d - int(t_o_d) for t_o_d in this_offset_dur]

        this_offset_spread_voice = []
        this_offset_spread_dur = []
        this_offset_spread_pitch = []

        for _n, rem in enumerate(rems):
            if rem == 0:
                continue
            else:
               # this assumes rem is in the dataset!
               assert rem in [.125, .25, .5, 1.]
               this_offset_spread_dur.append(rem)
               this_offset_spread_voice.append(_n)
               this_offset_spread_pitch.append(0)
        # now that we handled remainders, stick in the rests in chunks of 1.
        cores = [int(t_o_d) for t_o_d in this_offset_dur]
        for _n, core in enumerate(cores):
            if core == 0:
                continue
            else:
                # core should be an integer >= 1
                for _j in range(core):
                    this_offset_spread_voice.append(_n)
                    this_offset_spread_dur.append(1.0)
                    this_offset_spread_pitch.append(0)
        assert len(this_offset_spread_dur) == len(this_offset_spread_pitch)
        this_offset_symbols = np.array([(el1, el2) for el1, el2 in zip(this_offset_spread_pitch, this_offset_spread_dur)])
        joined_voices = []
        joined_symbols = []
        joined_voices.append(np.array(this_offset_spread_voice))
        joined_symbols.append(this_offset_symbols)
        joined_voices.append(this_voice)
        joined_symbols.append(this_sym)
        joined_voices = np.concatenate(joined_voices)
        joined_symbols = np.concatenate(joined_symbols)

    assert len(joined_voices) == len(joined_symbols)
    # remove file separator symbol now
    joined_symbols = [tuple(j) for j in joined_symbols]
    joined_symbols = [(int(j[0]), j[1]) for j in joined_symbols]
    joined_voices = [int(v) for v in joined_voices]

    reduced_indices = [n for n in range(len(joined_symbols)) if joined_symbols[n] != infill_corpus.file_separator_symbol]
    joined_symbols = [joined_symbols[n] for n in reduced_indices]
    joined_voices = [joined_voices[n] for n in reduced_indices]
    symbol_sequence = joined_symbols
    voice_sequence = joined_voices

    pitches_list = [s[0] for s in symbol_sequence]
    durations_list = [s[1] for s in symbol_sequence]
    voices_list = [vi for vi in voice_sequence]
    try:
        assert all([v != -1 for v in voices_list])
    except:
        print(t)
        print("error?")
        from IPython import embed; embed()#; raise ValueError()

    data = convert_voice_lists_to_music_json(pitch_lists=pitches_list, duration_lists=durations_list, voices_list=voices_list)

    json_fpath = midi_sample_dir + os.sep + "sampled{}_{}.json".format(t, tstamp)
    write_music_json(data, json_fpath)

    fpath = midi_sample_dir + os.sep + "sampled{}_{}.midi".format(t, tstamp)
    #a = "harpsichord_preset"
    #b = "woodwind_preset"
    #m = {0: [(a, 0), (b, marked_quarters_context_boundary[0])],
    #     1: [(a, 0), (b, marked_quarters_context_boundary[1])],
    #     2: [(a, 0), (b, marked_quarters_context_boundary[2])],
    #     3: [(a, 0), (b, marked_quarters_context_boundary[3])]}

    #a = "harpsichord_preset"
    a = "grand_piano_preset"
    m = {0: [(a, 0),],
         1: [(a, 0),],
         2: [(a, 0),],
         3: [(a, 0),]}
    music_json_to_midi(data, fpath, voice_program_map=m)
    print("Sampled {}".format(fpath))
