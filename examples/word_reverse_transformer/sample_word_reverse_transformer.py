from __future__ import print_function
import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F
import copy

from kkpthlib import fetch_jsb_chorales
from kkpthlib import MusicJSONRasterIterator
from kkpthlib import softmax_np
from kkpthlib import get_logger
from kkpthlib import CategoricalCrossEntropy
from kkpthlib import top_p_from_logits_np

from kkpthlib.datasets import fetch_norvig_words

from word_reverse_transformer import get_hparams
from word_reverse_transformer import build_model

import json
import os

logger = get_logger()
hp = get_hparams()

vocab = "_0123456789abcdefghijklmnopqrstuvwxyz"
norvig = fetch_norvig_words()
words = norvig["data"]
maxlen = max([len(words_i) for words_i in words])

word_length_limit = hp.word_length_limit

words = [words_i for words_i in words if len(words_i) <= word_length_limit]
v2i = {v: k for k, v in enumerate(vocab)}
i2v = {v: k for k, v in v2i.items()}
word_inds = [np.array([v2i[wi] for wi in word_i] + [0] * (word_length_limit - len(word_i)))[..., None] for word_i in words]
rev_word_inds = [np.array([v2i[wi] for wi in word_i][::-1] + [0] * (word_length_limit - len(word_i)))[..., None] for word_i in words]

model = build_model(hp)
m_dict = torch.load("model_checkpoint.pth", map_location=hp.use_device)
model.load_state_dict(m_dict)
model = model.eval()

test_word = "abcdefghij"
test_word = "gelato____"
test_inds = [v2i[c] for c in test_word]
test_inds = np.array(test_inds)[:, None, None].astype("float32")
bcast = np.ones((1, hp.batch_size, 1))
test_inds = test_inds * bcast
test_mask = np.ones_like(test_inds[..., 0])

sampled = 0. * test_inds[:1]
sampled_mask = 0. * sampled + 1.
sampled_mask = sampled_mask[..., 0]

for i in range(hp.word_length_limit):
    print("step {}".format(i))
    x = test_inds
    x_mask = test_mask

    y = sampled
    y_mask = sampled_mask

    t_x = torch.Tensor(x).to(hp.use_device)
    mask_t_x = torch.Tensor(x_mask).to(hp.use_device)
    t_y = torch.Tensor(y).to(hp.use_device)
    mask_t_y = torch.Tensor(y_mask).to(hp.use_device)
    pred_logit = model(t_x, t_y, mask_t_x, mask_t_y)
    new_sampled = pred_logit[-1].argmax(dim=-1).cpu().data.numpy()
    sampled = np.concatenate((sampled, new_sampled[None, :, None]))
    sampled_mask = 0. * sampled + 1.
    sampled_mask = sampled_mask[..., 0]
    current_str = [i2v[c] for c in sampled[1:, 0].ravel()]
    print("current string: {}".format(current_str))
from IPython import embed; embed(); raise ValueError()
