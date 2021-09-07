from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gc
import math

import os
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F

from scipy.io import wavfile

from kkpthlib.datasets import EnglishSpeechCorpus
from kkpthlib.datasets.speech.audio_processing.audio_tools import stft, istft

from kkpthlib import get_logger

from kkpthlib import Linear
from kkpthlib import Dropout
from kkpthlib import BernoulliCrossEntropyFromLogits
from kkpthlib import DiscretizedMixtureOfLogisticsCrossEntropyFromLogits
from kkpthlib import MixtureOfGaussiansNegativeLogLikelihood
from kkpthlib import relu
from kkpthlib import softmax
from kkpthlib import log_softmax
#from kkpthlib import clipping_grad_norm_
from kkpthlib import clipping_grad_value_
from kkpthlib import ListIterator
from kkpthlib import run_loop
from kkpthlib import HParams
from kkpthlib import Conv2d
from kkpthlib import Conv2dTranspose
from kkpthlib import SequenceConv1dStack
from kkpthlib import BiLSTMLayer

from kkpthlib import AttentionMelNetTier
from kkpthlib import MelNetTier
from kkpthlib import MelNetFullContextLayer
from kkpthlib import CategoricalCrossEntropyFromLogits
from kkpthlib import relu
from kkpthlib import Embedding

from kkpthlib import space2batch
from kkpthlib import batch2space
from kkpthlib import split
from kkpthlib import split_np
from kkpthlib import batch_mean_variance_update
from kkpthlib import interleave
from kkpthlib import scan
from kkpthlib import LSTMCell
from kkpthlib import BiLSTMLayer
from kkpthlib import GaussianAttentionCell

random_seed = 2122
data_random_state = np.random.RandomState(random_seed)
folder_base = "/usr/local/data/kkastner/robovoice/robovoice_d_25k"
fixed_minibatch_time_secs = 4
fraction_train_split = .9
speech = EnglishSpeechCorpus(metadata_csv=folder_base + "/metadata.csv",
                             wav_folder=folder_base + "/wavs/",
                             alignment_folder=folder_base + "/alignment_json/",
                             fixed_minibatch_time_secs=fixed_minibatch_time_secs,
                             train_split=fraction_train_split,
                             random_state=data_random_state)
# https://raw.githubusercontent.com/jfsantos/pytorch_enhancement/5384387457818f621a3248f9eaf5dc94433a046f/test_model_fast.py
import sys
import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from autoptim import minimize
from numpy.lib.stride_tricks import as_strided

'''
self.n_mels = 256
self.mel_freq_min = 125
self.mel_freq_max = 7600

self.stft_size = 6 * 256
self.stft_step = 256

# preemphasis filter
self.preemphasis_coef = 0.97
self.ref_level_db = 20
self.min_level_db = -90
'''
# same function as speech corpus code
def np_melspec(data, sample_rate):
    # takes in a raw sequence scaled between -1 and 1 (such as loaded from a wav file)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    x = data
    sr = sample_rate

    n_mels = 256

    fmin = 125
    fmax = 7600

    n_fft = 6 * 256
    n_step = 256

    # preemphasis filter
    coef = 0.97
    b = np.array([1.0, -coef], x.dtype)
    a = np.array([1.0], x.dtype)
    preemphasis_filtered = signal.lfilter(b, a, x)

    # mel weights
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype="float32")

    fftfreqs = np.linspace(0, float(sr) / 2., int(1 + n_fft // 2), endpoint=True)

    min_mel = herz_to_mel(fmin)
    max_mel = herz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    mel_f = mel_to_herz(mels)[:, 0]

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / float(fdiff[i])
        upper = ramps[i + 2] / float(fdiff[i + 1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0., np.minimum(lower, upper))
    # slaney style norm
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    mel_weights = weights

    # do stft
    ref_level_db = self.ref_level_db
    min_level_db = self.min_level_db
    def _amp_to_db(a):
        min_level = np.exp(min_level_db / 20. * np.log(10))
        return 20 * np.log10(np.maximum(min_level, a))

    abs_stft = np.abs(stft(preemphasis_filtered, fftsize=n_fft, step=n_step, real=True))
    melspec_ref = _amp_to_db(np.dot(mel_weights, abs_stft.T)) - ref_level_db
    melspec_clip = np.clip((melspec_ref - min_level_db) / -min_level_db, 0, 1)
    return melspec_clip.T

def lcl_overlap(X, window_size, window_step):
    """
    Create an overlapped version of X

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap

    window_size : int
        Size of windows to take

    window_step : int
        Step size between windows

    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    overlap_sz = window_size - window_step
    new_shape = X.shape[:-1] + ((X.shape[-1] - overlap_sz) // window_step, window_size)
    new_strides = X.strides[:-1] + (window_step * X.strides[-1],) + X.strides[-1:]
    X_strided = as_strided(X, shape=new_shape, strides=new_strides)
    return X_strided


def np_stft(waveform, window_size, window_step):
    # this matches the torch one...
    w = waveform - waveform.mean()
    X = lcl_overlap(w, window_size, window_step)
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))
    X = X * win[None]
    dft_matrix = np.zeros((n_fft, n_fft)).astype("complex64")
    #n_fft = window_size
    for k in range(0, n_fft):
        for n in range(0, n_fft):
            # -1 convention in np for forward transform
            dft_matrix[k, n] = np.exp(-1.j * 2 * np.pi * n * k / float(n_fft))
    # transpose it back after
    X_stft = np.dot(dft_matrix, X.T).T
    cut = window_size // 2 + 1
    X_stft = X_stft[:, :cut]
    return X_stft


def th_stft(waveform, window_size, window_step):
    w = waveform - waveform.mean()
    X = lcl_overlap(w, window_size, window_step)
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))
    X = X * win[None]
    X = torch.rfft(torch.Tensor(X), 1, onesided=True)
    cut = window_size // 2 + 1
    X = X[:, :cut]
    magnitudes = torch.sqrt(X[:, :, 0] ** 2 + X[:, :, 1] ** 2)
    """
    #freq_r_i = torch.stft(waveform, window_size, hop_length=window_step,
    #                      window=torch.hann_window(window_size).type(waveform.dtype))
    w = waveform - waveform.mean()
    # multiply by 2 / divide by 2 due to internal impl
    freq_r_i = torch.stft(w, window_size, hop_length=window_step,
                          window=win.type(waveform.dtype), onesided=False,
                          )
    magnitudes = torch.sqrt(freq_r_i[:, :, 0] ** 2 + freq_r_i[:, :, 1] ** 2)
    magnitudes = magnitudes.transpose(1, 0)
    magnitudes = magnitudes[:, :cut]
    """
    return magnitudes


def np_stft2(waveform, window_size, window_step):
    w = waveform - waveform.mean()
    X = lcl_overlap(w, window_size, window_step)
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))
    X = X * win[None]
    # need to pad in order to match fftpack.rfft
    # which of course is no longer fftpack, but was changed to use pocketfft?
    X = np.hstack((X, 0. * X[:, :1], X[:, ::-1]))
    #X = np.hstack((X, X[:, :-1][:, ::-1]))
    #X = np.hstack((X, X[:, ::-1]))
    n_fft = X.shape[1]
    dft_matrix = np.zeros((n_fft, n_fft)).astype("complex64")
    # fftpack (really pocketfft) appears to implement a 1/2 bin shift
    # ala https://dsp.stackexchange.com/a/54193
    # this is probably not what torch fft does by default
    for k in range(0, n_fft):
        for n in range(0, n_fft):
            # -1 convention in np for forward transform
            dft_matrix[k, n] = np.exp(-1.j * 2 * np.pi * n * (k + .5) / float(n_fft))
            #dft_matrix[k, n] = np.exp(-1.j * 2 * np.pi * n * k / float(n_fft))
    # transpose it back after
    X_stft = np.dot(dft_matrix, X.T).T
    cut = window_size // 2 + 1
    #X_stft = X_stft[:, :cut]
    X_stft = X_stft[:, -cut:][:, ::-1]
    return X_stft

def th_stft2(waveform, window_size, window_step):
    w = waveform - waveform.mean()
    X = lcl_overlap(w, window_size, window_step)
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))
    X = X * win[None]
    X = np.hstack((X, 0. * X[:, :1], X[:, ::-1]))
    n_fft = X.shape[1]
    dft_matrix = np.zeros((n_fft, n_fft)).astype("complex64")
    # fftpack (really pocketfft) appears to implement a 1/2 bin shift
    # ala https://dsp.stackexchange.com/a/54193
    # this is probably not what torch fft does by default
    for k in range(0, n_fft):
        for n in range(0, n_fft):
            # -1 convention in np for forward transform
            dft_matrix[k, n] = np.exp(-1.j * 2 * np.pi * n * (k + .5) / float(n_fft))
            #dft_matrix[k, n] = np.exp(-1.j * 2 * np.pi * n * k / float(n_fft))
    th_dft_matrix = torch.zeros(n_fft, n_fft, dtype=torch.cfloat)
    th_dft_matrix += dft_matrix.real
    th_dft_matrix += 1.0j * dft_matrix.imag
    th_dft_matrix = th_dft_matrix.to("cuda")

    th_X = torch.Tensor(X)
    th_X = th_X.to("cuda")

    X_stft = torch.matmul(th_dft_matrix, th_X.T).T
    cut = window_size // 2 + 1
    #X_stft = X_stft[:, :cut]
    #X_stft = X_stft[:, -cut:][:, ::-1]
    X_stft = torch.flip(X_stft[:, -cut:], [1])
    return X_stft

def np_melspec2(x, sr):
    # 'Center freqs' of mel bands - uniformly spaced between limits
    n_mels = 256

    fmin = 125
    fmax = 7600

    n_fft = 6 * 256
    n_step = 256

    # preemphasis filter
    coef = 0.97
    b = np.array([1.0, -coef], x.dtype)
    a = np.array([1.0], x.dtype)
    preemphasis_filtered = signal.lfilter(b, a, x)

    # mel weights
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype="float32")

    fftfreqs = np.linspace(0, float(sr) / 2., int(1 + n_fft // 2), endpoint=True)

    min_mel = herz_to_mel(fmin)
    max_mel = herz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    mel_f = mel_to_herz(mels)[:, 0]

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / float(fdiff[i])
        upper = ramps[i + 2] / float(fdiff[i + 1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0., np.minimum(lower, upper))
    # slaney style norm
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    mel_weights = weights

    # do stft
    ref_level_db = self.ref_level_db
    min_level_db = self.min_level_db
    def _amp_to_db(a):
        min_level = np.exp(min_level_db / 20. * np.log(10))
        return 20 * np.log10(np.maximum(min_level, a))

    abs_stft = np.abs(np_stft2(preemphasis_filtered, fftsize=n_fft, step=n_step, real=True))
    melspec_ref = _amp_to_db(np.dot(mel_weights, abs_stft.T)) - ref_level_db
    melspec_clip = np.clip((melspec_ref - min_level_db) / -min_level_db, 0, 1)
    return melspec_clip.T

el = speech.get_valid_utterances(1)
cond_seq_data_batch, cond_seq_mask, data_batch, data_mask = speech.format_minibatch(el)

filename = list(el[0][3].keys())[0] + ".wav"
wav_path = folder_base + "/wavs/" + filename
fs, d = wavfile.read(wav_path)
x = d.astype('float32') / (2 ** 15)
n_fft = 6 * 256
n_step = 256
np_stft_data = np.abs(stft(x, fftsize=n_fft, step=n_step, real=True))
alt_np_stft_data = np.abs(np_stft2(x, n_fft, n_step))
th_stft_data = torch.abs(th_stft(x, n_fft, n_step)).cpu().data.numpy()
alt_th_stft_data = torch.abs(th_stft2(x, n_fft, n_step)).cpu().data.numpy()

plt.imshow(th_stft_data)
plt.savefig("resample_spec_tmp_th.png")
plt.close()

plt.imshow(np_stft_data)
plt.savefig("resample_spec_tmp_np.png")
plt.close()

plt.imshow(alt_np_stft_data)
plt.savefig("resample_spec_tmp_np_alt.png")
plt.close()

plt.imshow(alt_th_stft_data)
plt.savefig("resample_spec_tmp_th_alt.png")
plt.close()
from IPython import embed; embed(); raise ValueError()



def sonify(magnitude, x0,
        fft_length=512,
        sample_rate=16000,
        hop_length=128,
        max_iter=100,
        sonify_tol=1e-9):
    # Define cost function
    def cost(theta):
        x = th_stft(theta, fft_length, hop_length)
        y = torch.FloatTensor(magnitude).type(x.dtype)

        def normalize(a):
            return a / torch.sqrt(torch.sum(a ** 2, dim=0) + 1E-12)

        x = normalize(x)
        y = normalize(y)
        min_shp = min(x.shape[0], y.shape[0])
        return torch.mean((x[:min_shp] - y[:min_shp]) ** 2)
    res, _ = minimize(cost, x0,
                      method='L-BFGS-B',
                      tol=sonify_tol,
                      options={"maxiter": max_iter,
                               "disp": True})
    return res


def test_fn(model, criterion, batch):
    x, y, lengths = batch

    x = Variable(x.cuda(), volatile=True)
    y = Variable(y.cuda(), requires_grad=False)

    mask = Variable(torch.ByteTensor(y.size()).fill_(1).cuda(),
            requires_grad=False)
    for k, l in enumerate(lengths):
        mask[:l, k, :] = 0

    hidden = model.init_hidden(x.size(0))
    y_hat = model.forward(x, hidden)

    # Apply mask
    y_hat.masked_fill_(mask, 0.0)
    y.masked_fill_(mask, 0.0)

    test_loss = criterion(y_hat, y).data.item()
    return y_hat.data.cpu().numpy(), test_loss

Y_hat, test_loss = test_fn(model, train_loop_best.criterion, batch)
X = np.exp(Y_hat.squeeze())
x0 = inv_spectrogram(X, phase,
		fft_length=int(window*1e-3*16000), sample_rate=16000,
		hop_length=int(step*1e-3*16000))
y_hat = sonify(X, x0,
		fft_length=int(window*1e-3*16000),
		hop_length=int(step*1e-3*16000))

y_hat = postprocess(y_hat, fs, step)
wavwrite(os.path.join(conddir, os.path.basename(f)),
                      y_hat, 16000)
