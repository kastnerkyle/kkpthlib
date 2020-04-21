from __future__ import print_function
import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F

from kkpthlib.datasets import fetch_jsb_chorales
from kkpthlib import AWDTransformerXLDecoderBlock

random_state = np.random.RandomState(11)
model = AWDTransformerXLDecoderBlock([380],
                                     name="transformer_block",
                                     random_state=random_state,
                                     mem_len=0,
                                     context_len=5,
                                     device="cpu")

fake = np.zeros((15, 7, 380))
fake_tensor = torch.Tensor(fake)

m = None
for i in range(5):
    o, m = model(fake_tensor, list_of_mems=m)

print("made it")
from IPython import embed; embed(); raise ValueError()

