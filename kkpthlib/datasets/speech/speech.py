import os
import numpy as np
import json
import time
import copy
from scipy import signal
from scipy.io import wavfile
from collections import OrderedDict
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
                       build_skiplist=True,
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

        self.cached_mean_vec_ = None
        self.cached_std_vec_ = None
        self.cached_count_ = None

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
        # each sil starts with ! , so !0, !1, !2, !3, !4

        sil_val = len(self.phone_lookup.keys())
        # +1 because 0:0.01, 0.01:0.0625, 0.0625:0.1325, 0.1325:0.25, 0.25:inf
        for _i in range(len(self.pause_duration_breakpoints) + 1):
            self.phone_lookup["!{}".format(_i)] = sil_val
            sil_val += 1

        # add start symbol
        self.phone_lookup["$"] = sil_val
        sil_val += 1
        # add started on a continue symbol
        self.phone_lookup["&"] = sil_val
        sil_val += 1
        # add eos symbol
        self.phone_lookup["~"] = sil_val
        sil_val += 1
        # add pad symbol
        self.phone_lookup["_"] = sil_val
        assert len(self.phone_lookup.keys()) == len(np.unique(list(self.phone_lookup.keys())))

        self.build_skiplist = build_skiplist
        if self.build_skiplist:
            # should be deterministic if we run the same script 2x
            random_state_val = random_state.randint(10000)
            skiplist_base_dir = os.getcwd() + "/skiplist_cache/"
            skiplist_base_path = skiplist_base_dir + "{}".format(metadata_csv.split(".csv")[0].replace("/", "-"))
            skiplist_train_path = skiplist_base_path + "_keyval{}_train_skip.txt".format(random_state_val)
            keeplist_train_path = skiplist_base_path + "_keyval{}_train_keep.txt".format(random_state_val)

            skiplist_valid_path = skiplist_base_path + "_keyval{}_valid_skip.txt".format(random_state_val)
            keeplist_valid_path = skiplist_base_path + "_keyval{}_valid_keep.txt".format(random_state_val)

            if not os.path.exists(skiplist_base_dir):
                os.mkdir(skiplist_base_dir)

            if not all([os.path.exists(p) for p in [skiplist_train_path, keeplist_train_path, skiplist_valid_path, keeplist_valid_path]]):
                # info for skip / keep lists is missing, must create it
                checks = ["noise", "sil", "oov", "#", "laughter", "<eps>"]

                train_failed_checks = OrderedDict()
                train_passed_checks = OrderedDict()
                for n, k in enumerate(self.train_keys):
                    utt = self.get_utterances(1, [self.train_keys[n]], do_not_filter=True)
                    utt_key = list(utt[0][3].keys())[0]
                    words = utt[0][3][utt_key]["full_alignment"]["words"]
                    for i in range(len(words)):
                        for j in range(len(words[i]["phones"])):
                            p = words[i]["phones"][j]["phone"]
                            for c in checks:
                                if c in p:
                                    if k not in train_failed_checks:
                                        train_failed_checks[k] = [(n, k, c)]
                                    else:
                                        train_failed_checks[k].append((n, k, c))
                    if k not in train_failed_checks:
                        train_passed_checks[k] = True
                    print("Building skiplist for train" + "," + str(n) + "," + k + " :::  " + utt[0][3][utt_key]["transcript"])
                # be sure there havent somehow been elements put into both lists
                for tk in train_passed_checks.keys():
                    assert tk not in train_failed_checks

                valid_failed_checks = OrderedDict()
                valid_passed_checks = OrderedDict()
                for n, k in enumerate(self.valid_keys):
                    utt = self.get_utterances(1, [self.valid_keys[n]], do_not_filter=True)
                    utt_key = list(utt[0][3].keys())[0]
                    words = utt[0][3][utt_key]["full_alignment"]["words"]
                    for i in range(len(words)):
                        for j in range(len(words[i]["phones"])):
                            p = words[i]["phones"][j]["phone"]
                            for c in checks:
                                if c in p:
                                    if k not in valid_failed_checks:
                                        valid_failed_checks[k] = [(n, k, c)]
                                    else:
                                        valid_failed_checks[k].append((n, k, c))
                    if k not in valid_failed_checks:
                        valid_passed_checks[k] = True
                    print("Building skiplist for valid" + "," + str(n) + "," + k + " :::  " + utt[0][3][utt_key]["transcript"])

                for vk in valid_passed_checks.keys():
                    assert vk not in valid_failed_checks
                    assert vk not in train_passed_checks
                    assert vk not in train_failed_checks
                for vk in valid_failed_checks.keys():
                    assert vk not in train_passed_checks
                    assert vk not in train_failed_checks
                # be sure there havent somehow been elements put into both lists, and no train/valid cross-pollination

                # we are to the critical point, writing the info out!
                with open(skiplist_train_path, "w") as f:
                    f.write("\n".join(list(train_failed_checks)))
                    print("Wrote skiplist for {}".format(skiplist_train_path))

                with open(keeplist_train_path, "w") as f:
                    f.write("\n".join(list(train_passed_checks)))
                    print("Wrote keeplist for {}".format(keeplist_train_path))

                with open(skiplist_valid_path, "w") as f:
                    f.write("\n".join(list(valid_failed_checks)))
                    print("Wrote skiplist for {}".format(skiplist_valid_path))

                with open(keeplist_valid_path, "w") as f:
                    f.write("\n".join(list(valid_passed_checks)))
                    print("Wrote keeplist for {}".format(keeplist_valid_path))

            # now we can assume the skip/keep lists exist, lets double check the current train/val keys against the one written out
            # then prune down
            with open(keeplist_train_path, "r") as f:
                loaded_train_keep_keys = [el.strip() for el in f.readlines()]
            with open(keeplist_valid_path, "r") as f:
                loaded_valid_keep_keys = [el.strip() for el in f.readlines()]
            self.train_keep_keys = [k for k in self.train_keys if k in loaded_train_keep_keys]
            self.valid_keep_keys = [k for k in self.valid_keys if k in loaded_valid_keep_keys]

            self._batch_utts_queue = []
            self._batch_used_keys_queue = []
            # sanity check we didnt delete the whole dataset
            assert len(self.train_keep_keys) > (len(self.train_keys) // 10)
            assert len(self.valid_keep_keys) > (len(self.valid_keys) // 10)
            # sanity check train keys wasnt so short we got 0
            assert len(self.train_keep_keys) > 0
            assert len(self.valid_keep_keys) > 0

    def get_utterances(self, size, all_keys, skip_mel=False,
                       min_length_words=None, max_length_words=None,
                       min_length_symbols=None, max_length_symbols=None,
                       min_length_time_secs=None, max_length_time_secs=None,
                       do_not_filter=False):
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
        used_keys = []
        # get a bigger extent, so if some don't match out filters we can keep going
        idx = self.random_state.choice(len(all_keys), 100 * size)
        for ii in idx:
            utt = self._fetch_utterance(all_keys[ii], skip_mel=skip_mel)
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
            if do_not_filter:
                pass
            else:
                if word_length > max_length_words or word_length < min_length_words:
                    continue
                if symbol_length > max_length_symbols or symbol_length < min_length_symbols:
                    continue
                if time_length > max_length_time_secs or time_length < min_length_time_secs:
                    # we could split this into subparts? potentially to make use of more data...
                    continue

            utts.append(utt)
            used_keys.append(all_keys[ii])
            if len(utts) >= size:
                break
        if len(utts) < size:
            raise ValueError("Unable to build correct length in get_utterances! Something has gone very wrong, debug this!")

        self._batch_used_keys_queue.append(used_keys)
        self._batch_used_keys_queue = self._batch_used_keys_queue[-5:]

        self._batch_utts_queue.append(utts)
        self._batch_utts_queue = self._batch_utts_queue[-5:]
        return utts

    def get_train_utterances(self, size, skip_mel=False):
        if self.build_skiplist:
            # we skip elements which had poor recognition
            return self.get_utterances(size, self.train_keep_keys, skip_mel=skip_mel)
        else:
            return self.get_utterances(size, self.train_keys, skip_mel=skip_mel)

    def get_valid_utterances(self, size, skip_mel=False):
        if self.build_skiplist:
            # we skip elements which had poor recognition
            return self.get_utterances(size, self.valid_keep_keys, skip_mel=skip_mel)
        else:
            return self.get_utterances(size, self.valid_keys, skip_mel=skip_mel)

    def load_mean_std_from_filepath(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError("Unable to find mean/std file at {}".format(filepath))
        d = np.load(filepath)
        self.cached_mean_vec_ = d["mean"].copy()
        self.cached_std_vec_ = d["std"].copy()
        self.cached_count_ = d["frame_count"].copy()

    def format_minibatch(self, utterances,
                         symbol_type="phoneme",
                         pause_duration_breakpoints=None,
                         quantize_to_n_bins=None):
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
                flat_phones_and_gaps = ["$"]
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
            assert mean_std_per_bin_normalization is False
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

    def melspectrogram_denormalize(self, ms):
        from IPython import embed; embed(); raise ValueError()

    def _old_melspectrogram_preprocess(self, data, sample_rate):
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
        # nfft - 1 because onesided=False cuts off last bin
        weights = np.zeros((n_mels, n_fft - 1), dtype="float32")

        fftfreqs = np.linspace(0, float(sr) / 2., n_fft - 1, endpoint=True)

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

        # ONE SIDED MUST BE FALSE!!!!!!!!
        abs_stft = np.abs(stft(preemphasis_filtered, fftsize=n_fft, step=n_step, real=True, compute_onesided=False))
        melspec_ref = _amp_to_db(np.dot(mel_weights, abs_stft.T)) - ref_level_db
        melspec_clip = np.clip((melspec_ref - min_level_db) / -min_level_db, 0, 1)
        return melspec_clip.T
