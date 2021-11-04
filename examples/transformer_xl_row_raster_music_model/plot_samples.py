import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kkpthlib import plot_piano_roll

#fpath = "310k_midi_samples_1024_5x_longer_context/sampled0.midi"
true_fpath = "310k_midi_samples_1024_5x_longer_context/true0.json"
# too many, comment a few out for now...
sample_fpaths = [
                 #"310k_midi_samples_1024_5x_longer_context/sampled0.json",
                 #"310k_midi_samples_1024_5x_longer_context/sampled1.json",
                 #"310k_midi_samples_1024_5x_longer_context/sampled2.json",
                 #"310k_midi_samples_1024_5x_longer_context/sampled3.json",
                 #"310k_midi_samples_1024_5x_longer_context/sampled4.json",
                 #"310k_midi_samples_1024_5x_longer_context/sampled5.json",
                 "310k_midi_samples_1024_5x_longer_context/sampled6.json",
                 "310k_midi_samples_1024_5x_longer_context/sampled7.json",
                 #"310k_midi_samples_1024_5x_longer_context/sampled8.json",
                 #"310k_midi_samples_1024_5x_longer_context/sampled9.json",]
                 ]

f, axarr = plt.subplots(len(sample_fpaths) + 1, 1, sharex=True)

force_length = 275
pitch_bot = 40
quantization = .25
plot_piano_roll(true_fpath, quantization, axis_handle=axarr[0], autorange=False, pitch_bot=pitch_bot, force_length=force_length)

axarr[0].axvline(32, color="black")
for n, fp in enumerate(sample_fpaths):
    plot_piano_roll(fp, quantization, axis_handle=axarr[n + 1], autorange=False, pitch_bot=pitch_bot, force_length=force_length)
    # line for seed
    axarr[n + 1].axvline(32, color="black")

plt.savefig("tmp.png")
