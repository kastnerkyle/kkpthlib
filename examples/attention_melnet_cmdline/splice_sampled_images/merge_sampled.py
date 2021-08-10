import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t00 = np.load("tier0_0_sampled/unnormalized_samples.npy")
t01_c00 = np.load("tier0_1_cond0_0_sampled/unnormalized_samples.npy")

comb = np.concatenate((0. * t00, 0. * t01_c00), axis=2)
comb[:, :, ::2, :] = t00
comb[:, :, 1::2, :] = t01_c00

plt.imshow(comb[0, :, :, 0])
plt.savefig("tmp.png")
from IPython import embed; embed(); raise ValueError()

