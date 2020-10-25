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

batch_np, batch_masks_np, batch_offsets_np = next(itr)

# sanity check one at a time
sampling_random_state = np.random.RandomState(111111)
for t in range(batch_np.shape[1]):
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

    current_boundary = context_boundary
    mask_positions = np.where(input_batch[:context_boundary, t, 0] == mask_token)[0]

    t1 = input_batch[context_boundary:]
    t2 = pad_batch_offsets_np[context_boundary:]
    a1 = [infill_corpus.dictionary.idx2word[ts] for ts in t1[:, t, 0]]
    assert len(a1) == (len(t2) - 1)
    # t2 and a1 are off by one because t2 has 1 extra "closure" element, slice it out
    assert np.all(t2[-1, t] == np.array([-1, 0, 0]))
    _a1_durations = [aa1[1] for aa1 in a1]
    _t2_durations = t2[:-1, t, -1]
    # ensure the sequences line up
    assert all([_a == _t for _a, _t in zip(_a1_durations, _t2_durations) if _a > 0])
    assert all([_t == 0 for _a, _t in zip(_a1_durations, _t2_durations) if _a <= 0])

    # 2 separate checks because < 0 is "special" symbols in the iterator, should have gt duration of 0

    # get expected duration total for each answer, for each voice!
    gt_answers = []
    gt_offsets = []
    _pre = [_aa1 if _aa1 > 0 else None for _aa1 in _a1_durations]
    split_idx = [_n for _n in range(len(_pre)) if _pre[_n] == None]
    if split_idx[0] != 0:
        split_idx.insert(0, 0)
    # if the last split idx isn't at the last element, add a dummy splitter
    if split_idx[-1] != (len(_pre) - 1):
         # list grabs aren't inclusive, so we add 1 so that l[a:b] grabs elem b-1 (the last value in our data)
         split_idx.append(len(_pre))
    boundaries = list(zip(split_idx[:-1], split_idx[1:]))
    for b in boundaries:
        assert b[0] + 1 < b[1]
        if b[0] > 0:
            # +1 to skip the boundary value of None
            gt_answers.append(a1[b[0] + 1:b[1]])
            gt_offsets.append(t2[b[0] + 1:b[1], t])
        else:
            gt_answers.append(a1[b[0]:b[1]])
            gt_offsets.append(t2[b[0]:b[1], t])

    # now that we have gathered the answer groups, make sure there are the same number as mask positions
    assert len(gt_answers) == len(mask_positions)
    assert len(gt_offsets) == len(mask_positions)

    all_answers = []
    all_answers_durations = []
    all_answers_voices = []
    for list_pos, mask_pos in enumerate(mask_positions):
        class ExactDurationConstraint(Constraint):
            def __init__(self, durations, total_duration_per_voice) -> None:
                super().__init__(durations)
                self.durations = durations
                self.total_duration_per_voice = total_duration_per_voice
                self.total_duration_per_voice_active = [tdpv for tdpv in self.total_duration_per_voice if tdpv > 0]

            def satisfied(self, assignment: Dict[tuple, int]) -> bool:
                # if all haven't been assigned, wait til they are
                if len(assignment) < len(self.durations):
                    return True

                # if there aren't notes in every voice, can't be a valid solution
                if len(set(assignment.values())) < len(self.total_duration_per_voice_active):
                    return False

                for v in sorted(list(set(assignment.values()))):
                    # for every voice which has been assigned notes, see if the voice is exactly the right total length
                    subset = {k: v_i for k, v_i in assignment.items() if v_i == v}
                    voice_sum = 0
                    for k, v_i in subset.items():
                        # add duration to voice sum
                        voice_sum += k[1]
                    # if any voice violates, solution is invalid
                    if voice_sum != self.total_duration_per_voice[v]:
                        return False
                # if everything passed
                return True
        # set of voices used in this subproblem
        active_voice_set = [int(v) for v in list(set(gt_offsets[list_pos][:, 0]))]

        # total durations per voice
        total_duration_constraint = gt_offsets[list_pos].sum(axis=0)[-1]
        total_duration_constraint_per_voice = [0, 0, 0, 0]
        for _t in range(4):
            voice_match = np.where(gt_offsets[list_pos][:, 0] == _t)[0]
            if len(voice_match) == 0:
                this_voice_dur = 0
            else:
                this_voice_dur = gt_offsets[list_pos][voice_match].sum(axis=0)[-1]
            total_duration_constraint_per_voice[_t] = this_voice_dur

        # set of all durations
        durations = sorted(set([k[1] for k in infill_corpus.dictionary.word2idx.keys() if k[1] > 0]))

        # need at least active_voice_set values
        # cannot use a single note longer than the whole duration constraint
        # when creating multiple, use existing duration to further eliminate
        # we also know the total maximum infilling size is 8

        #HEURISTIC: minimalism principle - least notes to fill the space?

        # max ngram is 8
        # could precalculate this maybe
        all_valid = []
        # should be 9, kept at 6 for debug
        for l in range(len(active_voice_set), 6):
            print("{}".format(l))
            sub_durations = [d for d in durations if d <= max(total_duration_constraint_per_voice)]
            all_combined = itertools.combinations_with_replacement(sub_durations, l)
            min_active = [t for v_i, t in enumerate(total_duration_constraint_per_voice) if v_i in active_voice_set]
            smol = min(min_active)
            all_reduced = [a for a in filter(lambda x: sum(x) >= smol, all_combined)]

            if l > len(active_voice_set):
                # reduce further by pigeonhole principle
                # if there are more durations than voices, and there is no combination of two that is smaller than the max, also violates
                all_reduced = [ar for ar in filter(lambda x: min(x) + min(x) <= max(total_duration_constraint_per_voice), all_reduced)]
                # if there are more durations than voices, and the sum of all largest n-1 values cannot make up the minimum, also violates
                all_reduced = [ar for ar in filter(lambda x: sum(x) - min(x) >= smol, all_reduced)]

            valid = []
            for ar in all_reduced:
                # note index as a unique entity, note tuple of pitch, duration
                variables = [(_n, _d) for _n, _d in enumerate(ar)]

                domains = {}
                for variable in variables:
                    domains[variable] = active_voice_set

                exact_csp = CSP(variables, domains)
                exact_csp.add_constraint(ExactDurationConstraint(variables, total_duration_constraint_per_voice))
                solution = exact_csp.backtracking_search()
                if solution is not None:
                    valid.append(solution)
            if len(valid) > 0:
                all_valid.extend(valid)

        from IPython import embed; embed(); raise ValueError()
        # note index as a unique entity, note tuple of pitch, duration
        variables = [(_n, infill_corpus.dictionary.idx2word[t_a]) for _n, t_a in enumerate(this_answer)]

        print(variables)
        domains = {}
        for variable in variables:
            domains[variable] = copy.deepcopy(active_voice_set)

        exact_csp = CSP(variables, domains)
        exact_csp.add_constraint(ExactVoiceConstraint(variables, total_duration_constraint_per_voice))

        this_answer = []
        this_answer_durations = []

        n_failed_answer_tries = 0
        n_resets = 0
        reset_current_boundary = current_boundary

        reset_limit = 1000

        duration_constraint = max(total_duration_constraint_per_voice)
        for _ in range(100):
            out, out_mems = model(input_batch[:current_boundary], batch_masks[:current_boundary], list_of_mems=None)
            temperature = args.temperature
            p_cutoff = args.p_cutoff
            # *must* duration constrain and voice assign these
            # can either force same amount of notes, and thus identical durations for each
            # or just force the overall answer to be the same total duration
            # would do greedy same durations for now, but need to move to constrained beam search

            # first, set probability of any predictions > total_duration_constraint to -inf
            log_p = out.cpu().data.numpy()

            # apply temperature here to avoid numerical trouble dividing neginf
            log_p_t = log_p / temperature

            """
            don't duration constrain it for now
            if n_failed_answer_tries < 5:
                invalid_indices = [_s for _s in range(len(infill_corpus.dictionary.idx2word))
                                   if infill_corpus.dictionary.idx2word[_s][1] > duration_constraint]
                # also block answer token?
                log_p_t[:, :, invalid_indices] = -np.inf

                # use top k instead of top p? so we don't loop on "answer" being huge prob

                # top_p reduction as normal, but we cannot do top_p if we catch a loop where answer is way more probable
                # than continuation
                # long term mebbe switch from top_p to top_k for this case
                reduced = top_p_from_logits_np(log_p_t, p_cutoff)
            else:
                reduced = log_p_t
            """
            reduced = log_p_t
            reduced_probs = softmax_np(reduced)

            sampled = sampling_random_state.choice(np.arange(reduced_probs.shape[-1]), p=reduced_probs[-1, t])
            sampled_duration = infill_corpus.dictionary.idx2word[sampled][1]

            if sampled == answer_token:
                # don't let answers just be predicted
                pass
            else:
                # a real beam search  or smarter thing would resample the next token til at least one solution exists?
                # resetting if sampled many times with no luck
                this_answer.append(sampled)
                this_answer_durations.append(infill_corpus.dictionary.idx2word[sampled][1])
                input_batch[current_boundary, t, 0] = sampled
                current_boundary += 1

            # flip this on its head - solution reweighting after constrained search? based on transformer logprob

            # after proposal, look at tentative greedy voice assignment
            if len(this_answer) > 1:
                # look at tentative voice assignment to get duration constraint
                active_voice_set = [int(v) for v in list(set(gt_offsets[list_pos][:, 0]))]

                # note index as a unique entity, note tuple of pitch, duration
                variables = [(_n, infill_corpus.dictionary.idx2word[t_a]) for _n, t_a in enumerate(this_answer)]
                print(variables)
                domains = {}
                for variable in variables:
                    domains[variable] = copy.deepcopy(active_voice_set)

                class VoiceConstraint(Constraint):
                    def __init__(self, notes, total_duration_per_voice) -> None:
                        super().__init__(notes)
                        self.notes = notes
                        self.total_duration_per_voice = total_duration_per_voice

                    def satisfied(self, assignment: Dict[tuple, int]) -> bool:
                        for v in sorted(list(set(assignment.values()))):
                            # for every voice which has been assigned notes, be sure the duration constraint per voice 
                            # has not yet been violated
                            subset = {k: v_i for k, v_i in assignment.items() if v_i == v}
                            voice_sum = 0
                            for k, v_i in subset.items():
                                voice_sum += k[1][1]
                            if voice_sum > self.total_duration_per_voice[v]:
                                return False
                        return True

                class ExactVoiceConstraint(Constraint):
                    def __init__(self, notes, total_duration_per_voice) -> None:
                        super().__init__(notes)
                        self.notes = notes
                        self.total_duration_per_voice = total_duration_per_voice

                    def satisfied(self, assignment: Dict[tuple, int]) -> bool:
                        # if there aren't notes in every voice, can't be a valid solution
                        if len(set(assignment.values())) != len(self.total_duration_per_voice):
                            return False

                        for v in sorted(list(set(assignment.values()))):
                            # for every voice which has been assigned notes, see if the voice is exactly the right total length
                            subset = {k: v_i for k, v_i in assignment.items() if v_i == v}
                            voice_sum = 0
                            for k, v_i in subset.items():
                                voice_sum += k[1][1]
                            if voice_sum != self.total_duration_per_voice[v]:
                                return False
                        return True

                partial_csp = CSP(variables, domains)
                partial_csp.add_constraint(VoiceConstraint(variables, total_duration_constraint_per_voice))

                exact_csp = CSP(variables, domains)
                exact_csp.add_constraint(ExactVoiceConstraint(variables, total_duration_constraint_per_voice))

                partial_solution = partial_csp.backtracking_search()
                if partial_solution is None:
                    print("No partial solutions, resetting!")
                    this_answer = []
                    this_answer_durations = []
                    current_boundary = reset_current_boundary
                    if n_resets > reset_limit:
                        print("too many resets need to think on this")
                        from IPython import embed; embed(); raise ValueError()
                    n_resets += 1
                    continue

                duration_constraint = 0
                remaining_durations = [tdc for tdc in total_duration_constraint_per_voice]
                for v_s in sorted(list(set(partial_solution.values()))):
                    subset = {k: v_i for k, v_i in partial_solution.items() if v_i == v_s}
                    voice_sum = 0
                    for k, v_i in subset.items():
                        voice_sum += k[1][1]
                    remaining_durations[v_s] = total_duration_constraint_per_voice[v_s] - voice_sum

                # need to check exact solutions and see if any exist
                exact_solution = exact_csp.backtracking_search()
                if exact_solution is not None:
                    print("exact soln found")
                    # add answer symbol to the main batch and bail
                    from IPython import embed; embed(); raise ValueError()
                else:
                    duration_constraint = max(remaining_durations)
            else:
                # must have at least 1 event per active voice!
                duration_constraint = max(total_duration_constraint_per_voice)

            if current_boundary >= (len(input_batch) - 5):
                # extend input batch with some zeros if we are close to the end
                input_batch = torch.cat((input_batch, 0 * input_batch[:10]), axis=0)

        # need to reconstruct the batch per answer autoregressively? or do all at once...
        all_answers.append(this_answer)
        all_answers_durations.append(this_answer_durations)
        #all_answers_voices.append(this_answer_voices)

    pred_durs = [sum(aad) for aad in all_answers_durations]
    gt_durs = [gto.sum(axis=0)[-1] for gto in gt_offsets]
    assert len(pred_durs) == len(gt_durs)
    assert all([pd == gtd for pd, gtd in zip(pred_durs, gt_durs)])

    modified_batch = copy.deepcopy(input_batch[:context_boundary].cpu().data.numpy())
    modified_offsets = copy.deepcopy(pad_batch_offsets_np[:context_boundary])
    modified_durations = modified_offsets[:, :, -1]
    for _i in range(len(all_answers)):
        answer = all_answers[_i]
        answer_durations = all_answers_durations[_i]

        mask_positions = np.where(modified_batch[:, t, 0] == mask_token)[0]
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

        modified_batch = np.concatenate((pre, mid, post), axis=0)
        modified_durations = np.concatenate((pre_d, mid_d, post_d), axis=0)

    # start with 0th entry
    token_sequence = modified_batch[:, t, 0]
    symbol_sequence = [infill_corpus.dictionary.idx2word[ts] for ts in token_sequence]
    duration_sequence = modified_durations[:, t]

    # check again that everything is aligned
    assert len(symbol_sequence) == len(duration_sequence)
    for _n in range(len(symbol_sequence)):
        if symbol_sequence[_n][0] < 0:
            assert duration_sequence[_n] == 0.
        elif symbol_sequence[_n][0] >= 0:
            assert symbol_sequence[_n][1] == duration_sequence[_n]

    special_symbols = [infill_corpus.mask_symbol,
                       infill_corpus.answer_symbol,
                       infill_corpus.end_context_symbol,
                       infill_corpus.file_separator_symbol,
                       infill_corpus.fill_symbol]

    reduced_indices = [n for n in range(len(symbol_sequence)) if symbol_sequence[n] not in special_symbols]

    symbol_sequence = [symbol_sequence[n] for n in reduced_indices]
    #offset_sequence = offsets[reduced_indices]

    offset_sequence = offsets

    # assert on gt test
    # assert sum(np.abs(np.array([s[1] for s in symbol_sequence]) - offset_sequence[:, 0, 2])) == 0
    # renormalize to start from 0....

    # add rests to compensate? or take in initial offsets to function
    # need to track offsets throughout in case it bridges a file boundary?
    # offsets are voice, offset timestep, duration
    pitches_list = []
    durations_list = []
    voices_list = []

    min_offset = min(offset_sequence[:, t, 1])
    per_voice_min_offset = [min(offset_sequence[np.where(offset_sequence[:, t, 0] == v)[0], t, 1]) for v in range(4)]

    base_pitches_list = [s[0] for s in symbol_sequence]
    base_durations_list = [s[1] for s in symbol_sequence]
    base_voices_list = [int(os) for os in offset_sequence[:, t, 0]]
    try:
        assert all([bv != -1 for bv in base_voices_list])
    except:
        print(t)
        print("error?")
        from IPython import embed; embed()#; raise ValueError()

    # add "blanks" to make the first notes start the correct place in time
    for v in range(4):
        scoot = per_voice_min_offset - min_offset
        if scoot[v] > 0:
            base_pitches_list.insert(0, 0)
            base_durations_list.insert(0, scoot[v])
            base_voices_list.insert(0, v)

    pitches_list.extend(base_pitches_list)
    durations_list.extend(base_durations_list)
    voices_list.extend(base_voices_list)

    """
    pitches_list.extend(base_pitches_list)
    durations_list.extend(base_durations_list)
    voices_list.extend([1 for s in symbol_sequence])

    pitches_list.extend(base_pitches_list)
    durations_list.extend(base_durations_list)
    voices_list.extend([2 for s in symbol_sequence])

    pitches_list.extend(base_pitches_list)
    durations_list.extend(base_durations_list)
    voices_list.extend([3 for s in symbol_sequence])
    """

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

    a = "harpsichord_preset"
    m = {0: [(a, 0),],
         1: [(a, 0),],
         2: [(a, 0),],
         3: [(a, 0),]}
    music_json_to_midi(data, fpath, voice_program_map=m)
    print("Sampled {}".format(fpath))
