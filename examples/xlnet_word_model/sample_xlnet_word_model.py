from __future__ import print_function
import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F
import copy
import sys

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


from importlib import import_module
model_import_path, fname = saved_model_definition_path.rsplit(os.sep, 1)
sys.path.insert(0, model_import_path)
p, _ = fname.split(".", 1)
mod = import_module(p)
get_hparams = getattr(mod, "get_hparams")
build_model = getattr(mod, "build_model")
sys.path.pop(0)

hp = get_hparams()
model = build_model(hp)
model_dict = torch.load(saved_model_path, map_location=hp.use_device)
model.load_state_dict(model_dict)
model.eval()

from kkpthlib import softmax_np
from kkpthlib import top_k_from_logits_np
from kkpthlib import top_p_from_logits_np
from kkpthlib import WordCorpus
from kkpthlib import StepIterator
from kkpthlib import make_batches_from_list

train_data_file_path = hp.data_storage_dir + os.sep + "train.txt"
valid_data_file_path = hp.data_storage_dir + os.sep + "valid.txt"
test_data_file_path = hp.data_storage_dir + os.sep + "test.txt"
corpus = WordCorpus(train_data_file_path=train_data_file_path,
                    valid_data_file_path=valid_data_file_path,
                    test_data_file_path=test_data_file_path,
                    cleaner_fn="lower_ascii_keep_standard_punctuation",
                    max_vocabulary_size=hp.max_vocabulary_size,
                    use_eos=False)
train_batches = make_batches_from_list(corpus.train, batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len)
valid_batches = make_batches_from_list(corpus.valid, batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len)
test_batches = make_batches_from_list(corpus.test, batch_size=hp.batch_size, sequence_length=hp.max_sequence_length, overlap=hp.context_len)

train_random_state = np.random.RandomState(hp.random_seed)
valid_random_state = np.random.RandomState(hp.random_seed + 1)
test_random_state = np.random.RandomState(hp.random_seed + 2)

sampling_random_state = np.random.RandomState(2123)

train_itr = StepIterator([train_batches], circular_rotation=True, random_state=train_random_state)
valid_itr = StepIterator([valid_batches], random_state=valid_random_state)
test_itr = StepIterator([test_batches], random_state=test_random_state)

np_data = next(valid_itr)
true_data = np_data.copy()

gen_random_state = np.random.RandomState(hp.random_seed + 3)

np_perm_masks, np_target_mappings, np_target_masks, np_input_ks, np_input_qs, np_targets, np_perm_orders = model.transformer.make_inputs_targets_masks_and_mappings(np_data, K=6, context_cut=hp.context_len, random_state=gen_random_state, sequential_order=True)

out_mems = None

num_blanks = np.sum(np_target_masks, axis=0)
num_filled = 0. * num_blanks
orig_np_target_masks = np.copy(np_target_masks)

in_mems = None
#in_mems = out_mems

if hp.use_target_mappings:
    old = np.copy(np_targets)
    old_subs = [old[np.where(np_target_masks[:, i])[0], i][None] for i in range(hp.batch_size)]
    np_targets = np.concatenate(old_subs).transpose(1, 0)
    # all 1s mask
    np_target_masks = 0. * np_targets + 1.

    pad_l = np.zeros((hp.context_len, hp.batch_size))
    np_targets = np.concatenate((pad_l, np_targets))
    np_target_masks = np.concatenate((pad_l, np_target_masks))
    # assume len(np_input_ks) == orig len(targets)
    pad_r = np.zeros((len(np_input_ks) - len(np_targets), hp.batch_size))
    np_targets = np.concatenate((np_targets, pad_r))
    np_target_masks = np.concatenate((np_target_masks, pad_r))
else:
    np_target_mappings = None


input_ks = torch.tensor(np_input_ks).to(hp.use_device)
input_qs = torch.tensor(np_input_qs).to(hp.use_device)
targets = torch.tensor(np_targets).long().to(hp.use_device)

input_ks = input_ks[..., None]
input_qs = input_qs[..., None]
targets = targets[..., None]

perm_masks = torch.tensor(np_perm_masks).to(hp.use_device)
target_masks = torch.tensor(np_target_masks).to(hp.use_device)

if np_target_mappings is not None:
    target_mappings = torch.tensor(np_target_mappings).to(hp.use_device)
else:
    target_mappings = None


# loop through and UNK out to be assured of no leaks
for k in range(np_data.shape[1]):
    blank_spots = np.where(orig_np_target_masks[:, k])[0]
    np_data[blank_spots, k] = corpus.dictionary.word2idx["<unk>"]

filled_sentences = [list() for i in range(np_data.shape[1])]
finished_sampling = [False for i in range(np_data.shape[1])]
while not all(finished_sampling):
    print("step {} done".format(max(num_filled)))
    np_input_ks = np_data[:-1].astype("float32")
    input_ks = torch.tensor(np_input_ks).to(hp.use_device)
    input_ks = input_ks[..., None]

    #input_qs = None

    # modify perm masks and target masks?
    # once we sample, should we recreate the perm masks and target masks with one less

    out_h, out_g, out_mems = model(input_ks, input_qs, perm_masks, target_mappings, target_masks, list_of_mems=in_mems)
    temp = .9

    #unk_index = corpus.dictionary.word2idx["<unk>"]
    # don't let it predict UNK
    #out_g[:, :, unk_index] = -1E30

    reduced = top_p_from_logits_np(out_g.cpu().data.numpy() / temp, .95)
    # don't let it predict unk
    reduced_probs = softmax_np(reduced)

    for k in range(np_data.shape[1]):
        blank_spots = np.where(orig_np_target_masks[:, k])[0]
        si = int(num_filled[k])

        if si >= len(blank_spots):
            finished_sampling[k] = True
            continue

        if np_target_mappings is None:
            print("ARE WE HANDLING NP_TARGET_MAPPINGS CORRECTLY AND INDEXING APPROPRIATELY")
            from IPython import embed; embed(); raise ValueError()

        blank_spot = blank_spots[si]
        #perm_masks[:, blank_spot, k] = 0.
        #target_masks[blank_spot, k] = 0.
        #input_qs[blank_spot, k, 0] = 0.

        context_sentence = " ".join([corpus.dictionary.idx2word[c] for c in np_data[:, k]])
        if len(filled_sentences[k]) == 0:
            filled_sentences[k].append(context_sentence)

        sampled_i = sampling_random_state.choice(np.arange(reduced_probs.shape[-1]), p=reduced_probs[si, k])

        np_data[blank_spot, k] = sampled_i

        new_context_sentence = " ".join([corpus.dictionary.idx2word[c] for c in np_data[:, k]])
        filled_sentences[k].append(new_context_sentence)

        num_filled[k] += 1

from IPython import embed; embed(); raise ValueError()
np_data = np_data[:hp.context_len]

# do a true sampling?
if False:
    context_sentence = " ".join([corpus.dictionary.idx2word[c] for c in np_data[:hp.context_len + 1, 0]])
    sampled_sentence = " ".join([corpus.dictionary.idx2word[c] for c in np_data[hp.context_len + 1:, 0]])
    print("==================")
    print("step {}".format(i))
    print("context: {}".format(context_sentence))
    print("sampled: {}".format(sampled_sentence))
    print("==================")
    input_data = torch.tensor(np_data).to(hp.use_device)
    input_data = input_data[..., None]

    in_mems = out_mems
    linear_out, out_mems = model(input_data, list_of_mems=in_mems)

    temp = .9
    reduced = top_p_from_logits_np(linear_out.cpu().data.numpy() / temp, .95)
    reduced_probs = softmax_np(reduced)
    sample_last = [sampling_random_state.choice(np.arange(reduced_probs.shape[-1]), p=reduced_probs[-1, j]) for j in range(reduced_probs.shape[1])]
    np_data = np.concatenate((np_data, np.array(sample_last)[None]), axis=0)
true_sentence = " ".join([corpus.dictionary.idx2word[c] for c in true_data[:, 0]])
sampled_sentence = " ".join([corpus.dictionary.idx2word[c] for c in np_data[:, 0]])
print("=================")
print("true sentence: {}".format(true_sentence))
print("=================")
print("sampled sentence: {}".format(sampled_sentence))
print("=================")
from IPython import embed; embed(); raise ValueError()
