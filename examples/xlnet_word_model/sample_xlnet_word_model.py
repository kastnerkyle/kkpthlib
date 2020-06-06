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
# +1 to account for autoregressive targets
# context_len because we have a reduction mapping - targets are a subset of the "target sequence", effectively
np_perm_masks, np_target_mappings, np_target_masks, np_input_qs, np_perm_orders = model.transformer.make_masks_and_mappings(np_data, context_cut=hp.context_len + 1, random_state=gen_random_state, sequential_order=True)

from IPython import embed; embed(); raise ValueError()


np_data = np_data[:hp.context_len + 1]
out_mems = None

for i in range(210):
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
