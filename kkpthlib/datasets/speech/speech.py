import os
import numpy as np
import json
import time
import copy
from scipy import signal
from scipy.io import wavfile
from .frontends import EnglishPhonemeLookup
from .audio_processing.audio_tools import herz_to_mel, mel_to_herz
from .audio_processing.audio_tools import stft, istft

class EnglishSpeechCorpus(object):
    """
    Frontend processing inspired by r9y9 (Ryuichi Yamamoto) DeepVoice3 implementation
    https://github.com/r9y9/deepvoice3_pytorch

    which in turn variously is inspired by librosa filter implementations, Tacotron implementation by Keith Ito
    some modifications from more recent TTS work
    """
    def __init__(self, metadata_csv, wav_folder, alignment_folder=None, remove_misaligned=True, cut_on_alignment=True,
                       train_split=0.9,
                       min_length_words=3, max_length_words=100,
                       min_length_symbols=7, max_length_symbols=200,
                       min_length_time_secs=2, max_length_time_secs=None,
                       fixed_minibatch_time_secs=6,
                       random_state=None):
        self.metadata_csv = metadata_csv
        self.wav_folder = wav_folder
        self.alignment_folder = alignment_folder
        self.random_state = random_state
        self.train_split = train_split

        self.min_length_words = min_length_words
        self.max_length_words = max_length_words
        self.min_length_symbols = min_length_symbols
        self.max_length_symbols = max_length_symbols
        self.min_length_time_secs = min_length_time_secs
        if max_length_time_secs is None:
            max_length_time_secs = fixed_minibatch_time_secs
        self.max_length_time_secs = max_length_time_secs

        self.n_mels = 256

        self.cut_on_alignment = cut_on_alignment
        self.remove_misaligned = remove_misaligned

        self.mel_freq_min = 125
        self.mel_freq_max = 7600

        self.stft_size = 6 * 256
        self.stft_step = 256

        # preemphasis filter
        self.preemphasis_coef = 0.97
        self.ref_level_db = 20
        self.min_level_db = -90

        info = {}
        with open(metadata_csv, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                info[parts[0]] = {}
                info[parts[0]]["metadata_transcript"] = parts[2]

        self.misaligned_keys = []
        self.aligned_keys = []
        start_time = time.time()
        if alignment_folder is not None:
            for k in sorted(info.keys()):
                alignment_info_json = alignment_folder + k + ".json"
                with open(alignment_info_json, "r") as read_file:
                    alignment = json.load(read_file)
                skip_example = False
                for el in alignment["words"]:
                    if el["case"] != "success":
                        #print("skipping {} due to unaligned word".format(k))
                        skip_example = True
                if skip_example == True:
                    self.misaligned_keys.append(k)
                else:
                    self.aligned_keys.append(k)
        shuf_keys = copy.deepcopy(self.aligned_keys)
        random_state.shuffle(shuf_keys)
        splt = int(self.train_split * len(shuf_keys))
        self.train_keys = shuf_keys[:splt]
        self.valid_keys = shuf_keys[splt:]

        """
        # code used to pre-calculate information about pauses and gaps
        start_time = time.time()
        all_gaps = []
        for _n, k in enumerate(sorted(info.keys())):
            print("{} of {}".format(_n, len(info.keys())))
            if k in self.misaligned_keys:
                continue
            fs, d, this_melspec, this_info = self._fetch_utterance(k, skip_mel=True)
            # start stop boundaries
            start_stop = [(el["start"], el["end"]) for el in this_info[k]["full_alignment"]["words"]]
            gaps = []
            last_end = 0
            for _s in start_stop:
                gaps.append(_s[0] - last_end)
                last_end = _s[1]
            final_gap = (len(d) / float(fs)) - last_end
            gaps.append(final_gap)
            all_gaps.extend(gaps)
        stop_time = time.time()
        print("total iteration time")
        print(stop_time - start_time)
        gap_arr = np.array(all_gaps)
        # assume any negatives are failed alignment - just give min pause duration aka basically 0
        gap_arr[gap_arr < 0] = 0.
        # look at histograms to pick values...
        # A Large Scale Study of Multi-lingual Pause Duration
        # longest category only occuring in spontaneous speech
        # we split the smaller chunk into 0.01, 0.05, 0.125, .5 as approximate "halving" of remaining values. With the .5 value adding a separate
        # category for "long tail" silence that may be difficult to model with attention
        # https://web.archive.org/web/20131114035947/http://aune.lpl.univ-aix.fr/projects/aix02/sp2002/pdf/campione-veronis.pdf
        import matplotlib.pyplot as plt
        n, bins, patches = plt.hist(gap_arr, 100, density=True)
        plt.savefig("tmp1.png")
        plt.close()

        n, bins, patches = plt.hist(gap_arr[gap_arr > 0.01], 100, density=True)
        plt.savefig("tmp2.png")
        plt.close()

        n, bins, patches = plt.hist(gap_arr[gap_arr > 0.0625], 100, density=True)
        plt.savefig("tmp3.png")
        plt.close()

        n, bins, patches = plt.hist(gap_arr[gap_arr > 0.1325], 100, density=True)
        plt.savefig("tmp4.png")
        plt.close()

        n, bins, patches = plt.hist(gap_arr[gap_arr > 0.25], 100, density=True)
        plt.savefig("tmp5.png")
        plt.close()
        """
        self.pause_duration_breakpoints = [0.01, 0.0625, 0.1325, 0.25]
        self.phone_lookup = EnglishPhonemeLookup()
        # each sil starts with # , so #0, #1, #2, #3, #4

        sil_val = len(self.phone_lookup.keys())
        # +1 because 0:0.01, 0.01:0.0625, 0.0625:0.1325, 0.1325:0.25, 0.25:inf
        for _i in range(len(self.pause_duration_breakpoints) + 1):
            self.phone_lookup["!{}".format(_i)] = sil_val
            sil_val += 1

        # add eos symbol
        self.phone_lookup["~"] = sil_val
        sil_val += 1
        # add pad
        self.phone_lookup["_"] = sil_val
        assert len(self.phone_lookup.keys()) == len(np.unique(list(self.phone_lookup.keys())))

    def get_utterances(self, size, all_keys,
                       min_length_words=None, max_length_words=None,
                       min_length_symbols=None, max_length_symbols=None,
                       min_length_time_secs=None, max_length_time_secs=None):
        if min_length_words is None:
            min_length_words = self.min_length_words
        if max_length_words is None:
            max_length_words = self.max_length_words
        if min_length_symbols is None:
            min_length_symbols = self.min_length_symbols
        if max_length_symbols is None:
            max_length_symbols = self.max_length_symbols
        if min_length_time_secs is None:
            min_length_time_secs = self.min_length_time_secs
        if max_length_time_secs is None:
            max_length_time_secs = self.max_length_time_secs

        utts = []
        # get a bigger extent, so if some don't match out filters we can keep going
        idx = self.random_state.choice(len(all_keys), 100 * size)
        for ii in idx:
            utt = self._fetch_utterance(all_keys[ii])
            # fs, d, melspec, info
            core_key = list(utt[-1].keys())[0]
            word_length = len(utt[-1][core_key]["transcript"].split(" "))
            time_length = len(utt[1]) / float(utt[0])
            #print("{}".format(utt[-1][core_key]["transcript"]))
            #print("time_length {}".format(time_length))
            phoneme_parts = [len(el["phones"]) for el in utt[-1][core_key]["full_alignment"]["words"]]
            # add on len(phoneme_parts) - 1 to account for added spaces
            phoneme_length = sum(phoneme_parts) + len(phoneme_parts) - 1
            char_length = len(utt[-1][core_key]["transcript"])
            # just use the min for filtering, should be close in length for most cases
            symbol_length = min(char_length, phoneme_length)
            if word_length > max_length_words or word_length < min_length_words:
                continue
            if symbol_length > max_length_symbols or symbol_length < min_length_symbols:
                continue
            if time_length > max_length_time_secs or time_length < min_length_time_secs:
                continue

            utts.append(utt)
            if len(utts) >= size:
                break
        if len(utts) < size:
            raise ValueError("Unable to build correct length in get_utterances! Something has gone very wrong, debug this!")
        return utts

    def get_train_utterances(self, size):
        return self.get_utterances(size, self.train_keys)

    def get_valid_utterances(self, size):
        return self.get_utterances(size, self.valid_keys)

    def format_minibatch(self, utterances,
                               symbol_type="phoneme",
                               pause_duration_breakpoints=None,
                               quantize_to_n_bins=256):
        if pause_duration_breakpoints is None:
            pause_duration_breakpoints = self.pause_duration_breakpoints
        phoneme_sequences = []
        melspec_sequences = []

        if symbol_type != "phoneme":
            raise ValueError("Only supporting symbol_type phoneme for now")

        if symbol_type == "phoneme":
            if self.alignment_folder is None:
                raise ValueError("symbol_type phoneme minibatch formatting not supported without 'aligment_folder' argument to speech corpus init!")
            for utt in utterances:
                fs, d, melspec, al = utt
                melspec_sequences.append(melspec)
                k = list(al.keys())[0]
                phone_groups = [el["phones"] for el in al[k]["full_alignment"]["words"]]
                start_stop = [(el["start"], el["end"]) for el in al[k]["full_alignment"]["words"]]
                gaps = []
                last_end = 0
                for _s in start_stop:
                    gaps.append(_s[0] - last_end)
                    last_end = _s[1]
                final_gap = (len(d) / float(fs)) - last_end
                gaps.append(final_gap)
                gaps_arr = np.array(gaps)
                v = len(pause_duration_breakpoints) - 1
                gap_idx_groups = []
                prev_pd = -np.inf
                for pd in pause_duration_breakpoints:
                    gap_idx_groups.append((gaps_arr >= prev_pd) & (gaps_arr < pd))
                    prev_pd = pd
                gap_idx_groups.append((gaps_arr >= prev_pd))
                for _n, gig in enumerate(gap_idx_groups):
                    gaps_arr[gig] = _n
                # reverse iterate pause duration breakpoints to quantized gap values
                gaps_arr = gaps_arr.astype("int32")
                phone_group_syms = [[pgi["phone"] for pgi in pg] for pg in phone_groups]
                flat_phones_and_gaps = []
                for _n in range(len(phone_group_syms)):
                    flat_phones_and_gaps.append("!{}".format(gaps_arr[_n]))
                    flat_phones_and_gaps.extend(phone_group_syms[_n])
                flat_phones_and_gaps.append("~")
                flat_phones_and_gaps.append("!{}".format(gaps_arr[-1]))
                seq_as_ints = [self.phone_lookup[s] for s in flat_phones_and_gaps]
                phoneme_sequences.append(seq_as_ints)
        # pad it out so all are same length
        max_seq_len = max([len(ps) for ps in phoneme_sequences])
        # mask and padded sequence
        input_seq_mask = [[1.] * len(ps) + [0.] * (max_seq_len - len(ps)) for ps in phoneme_sequences]
        input_seq_mask = np.array(input_seq_mask).T
        phoneme_sequences = [ps + (max_seq_len - len(ps)) * [self.phone_lookup["_"]] for ps in phoneme_sequences]
        phoneme_sequences = np.array(phoneme_sequences).astype("float32").T
        overlap_len = ((self.max_length_time_secs * fs) + self.stft_size) % self.stft_size
        max_frame_count = (((self.max_length_time_secs * fs) + self.stft_size) - overlap_len) / self.stft_step
        divisors = [2, 4, 8]
        for di in divisors:
            # nearest divisble number above, works because largest divisor divides by smaller
            q = int(max_frame_count / di)
            if float(max_frame_count / di) == int(max_frame_count / di):
                max_frame_count = di * q
            else:
                max_frame_count = di * (q + 1)
        assert max_frame_count == int(max_frame_count)
        melspec_seq_mask = [[1.] * ms.shape[0] + [0.] * int(max_frame_count - ms.shape[0]) for ms in melspec_sequences]
        melspec_seq_mask = np.array(melspec_seq_mask)
        padded_melspec_sequences = []
        for ms in melspec_sequences:
            melspec_padded = 0. * melspec[:1, :] + np.zeros((int(max_frame_count), 1)).astype("float32")
            melspec_padded[:len(ms)] = ms
            padded_melspec_sequences.append(melspec_padded)
        padded_melspec_sequences = np.array(padded_melspec_sequences)
        if quantize_to_n_bins is not None:
            n_bins = quantize_to_n_bins
            bins = np.linspace(0., 1., num=n_bins, endpoint=True)
            quantized_melspec_sequences = np.digitize(padded_melspec_sequences, bins)
        else:
            quantized_melspec_sequences = padded_melspec_sequences
        return phoneme_sequences, input_seq_mask.astype("float32"), quantized_melspec_sequences.astype("float32"), melspec_seq_mask.astype("float32")

    def _fetch_utterance(self, basename, skip_mel=False):
        # fs, d, melspec, info
        this_info = {}
        this_info[basename] = {}
        wav_path = self.wav_folder + "/" + basename + ".wav"
        fs, d = wavfile.read(wav_path)
        d = d.astype('float32') / (2 ** 15)
        if self.alignment_folder is not None:
            # alignment from gentle
            alignment_info_json = self.alignment_folder + "/" + basename + ".json"
            with open(alignment_info_json, "r") as read_file:
                alignment = json.load(read_file)
            end_in_samples = fs * alignment["words"][-1]["end"]
            end_in_samples += 2 * self.stft_size
            end_in_samples = int(end_in_samples)
            # add a little bit of extra, if the cut is a ways before the end
            start_in_samples = fs * alignment["words"][0]["start"]
            start_in_samples -= 2 * self.stft_size
            start_in_samples = int(max(start_in_samples, 0))
            # cut a little bit before the start
            if self.cut_on_alignment:
                d = d[start_in_samples:end_in_samples]
            this_info[basename]["full_alignment"] = alignment
            this_info[basename]["transcript"] = alignment["transcript"]
        if skip_mel:
            return fs, d, None, this_info
        # T, F melspec
        melspec = self._melspectrogram_preprocess(d, fs)
        # check full validity outside the core fetch
        return fs, d, melspec, this_info

    def _melspectrogram_preprocess(self, data, sample_rate):
        # takes in a raw sequence scaled between -1 and 1 (such as loaded from a wav file)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        x = data
        sr = sample_rate

        n_mels = self.n_mels

        fmin = self.mel_freq_min
        fmax = self.mel_freq_max

        n_fft = self.stft_size
        n_step = self.stft_step

        # preemphasis filter
        coef = self.preemphasis_coef
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
