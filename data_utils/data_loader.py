# conding : utf-8
#
#
#
#
import time
import json
import codecs
import librosa
import numpy as np
from joblib import Parallel, delayed
from data_utils.audio_feature import AudioFeature
from multiprocessing import Process, Queue, Pool, Manager


class AudioParser(object):
    def __init__(self,
                 sample_rate=8000,
                 window_ms=32,
                 stride_ms=16,
                 snr=0,
                 windows_name=None,
                 use_complex=False):
        self.snr = snr
        self.sample_rate = sample_rate
        self.window_s = window_ms / 1000
        self.stride_s = stride_ms / 1000
        self.extractor = AudioFeature(windows_name)
        self.complex = use_complex

    def load_audio(self, audio_filepath):
        signal, sample_rate = librosa.load(audio_filepath, sr=self.sample_rate)
        return signal, sample_rate

    def add_noise(self, speech, noise):
        if len(speech) >= len(noise):
            diff_size = len(speech) - len(noise)
            for i in range(int(np.ceil(diff_size / len(noise)))):
                noise = np.concatenate((noise, noise*np.random.uniform(0, 2)))
            noise = noise[:len(speech)]
        else:
            diff_length = len(noise) - len(speech)
            start_point = np.random.randint(0, diff_length)
            end_point = start_point + len(speech)
            noise = noise[start_point:end_point]

        p_sig = np.sum(abs(speech) ** 2)
        background_volume = p_sig / (10 ** (self.snr / 10))
        p_back = np.sum(abs(noise) ** 2)
        new_noise = np.sqrt(background_volume / p_back) * noise
        mix_sig = speech + new_noise
        return mix_sig

    def parse_audio(self, sig):
        spectrogram = self.extractor.compute_spectrogram(sig,
                                                         self.sample_rate,
                                                         window_s=self.window_s,
                                                         stride_s=self.stride_s,
                                                         nfft=256,
                                                         use_complex=self.complex)
        return spectrogram


class DataSet(AudioParser):
    def __init__(self,
                 manifest_filepath,
                 noise_manifest,
                 sample_rate=16000,
                 window_ms=32,
                 stride_ms=16,
                 snr=0,
                 min_duration=0.4,
                 max_duration=float('inf'),
                 windows_name=None,
                 use_complex=False):
        super(DataSet, self).__init__(sample_rate=sample_rate,
                                      window_ms=window_ms,
                                      stride_ms=stride_ms,
                                      snr=snr,
                                      windows_name=windows_name,
                                      use_complex=use_complex)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.noise_manifest = noise_manifest

        self.item_list = self.read_manifest(manifest_filepath)
        if noise_manifest is not None:
            self.noise_list = self.read_manifest(noise_manifest)
            if len(self.noise_list) < len(self.item_list):
                self.noise_list = self.noise_list * int(np.ceil(len(self.item_list) / len(self.noise_list)))
            assert len(self.noise_list) >= len(self.item_list)

    def read_manifest(self, manifest_path):
        """
        Load data from manifest file.
        :param manifest_path:  str, path of manifest file
        :return manifest:      list, json list
        """
        manifest = []
        for json_line in codecs.open(manifest_path, 'r', 'utf-8'):
            try:
                json_data = json.loads(json_line)
            except Exception as e:
                raise IOError("Error reading manifest: %s" % str(e))
            if self.max_duration >= json_data["duration"] >= self.min_duration:
                manifest.append(json_data)
        return manifest

    def __getitem__(self, index):
        if self.noise_manifest is not None:
            clean_audio = self.item_list[index]["audio_filepath"]
            noise_audio = self.noise_list[index]["audio_filepath"]
            speech, _ = self.load_audio(clean_audio)
            noise, _ = self.load_audio(noise_audio)
            mix_sig = self.add_noise(speech, noise)
            speech_spec = self.parse_audio(speech)
            mix_spec = self.parse_audio(mix_sig)
        else:
            clean_audio = self.item_list[index]["clean_audio_filepath"]
            mix_audio = self.item_list[index]["mix_audio_filepath"]
            speech, _ = self.load_audio(clean_audio)
            mix_sig, _ = self.load_audio(mix_audio)
            speech_spec = self.parse_audio(speech)
            mix_spec = self.parse_audio(mix_sig)
        return (mix_sig, speech), (mix_spec, speech_spec)

    def __len__(self):
        return len(self.item_list)

    def __call__(self, *args, **kwargs):
        return self

    def shuffle(self):
        np.random.shuffle(self.item_list)


class Sampler(object):
    def __init__(self, dataset, batch_size, start_index=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.start_index = start_index
        if drop_last:
            last_size = len(self.dataset) % batch_size
            self.dataset.item_list = self.dataset.item_list[:-last_size]
        else:
            last_size = (int(len(self.dataset) / batch_size) + 1) * batch_size - len(self.dataset)
            self.dataset.item_list.extend(self.dataset.item_list[-last_size:])
        ids = list(range(len(self.dataset)))
        self.bins = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]
        self.indices = (np.random.permutation(len(self.bins) - self.start_index) + self.start_index).tolist()

    def __iter__(self):
        for x in self.indices:
            batch_ids = self.bins[x]
            np.random.shuffle(batch_ids)
            yield batch_ids

    def __len__(self):
        return len(self.bins) - self.start_index

    def reset_start_index(self, start_index):
        self.start_index = start_index

    def __call__(self, *args, **kwargs):
        return self

    def iter_num(self):
        return len(self.indices)


class DataLoader(object):
    def __init__(self, dataset, batch_size, sampler=None, num_works=2):
        super(DataLoader, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_works = num_works
        self.q = []
        if self.sampler is None:
            self.bins = []
            for i in range(0, len(self.dataset), self.batch_size):
                if i + self.batch_size < len(self.dataset):
                    index_list = list(range(i, i + self.batch_size))
                else:
                    index_list = list(range(i, len(self.dataset)))
                self.bins.append(index_list)

    def one_point(self, x):
        a, b = self.dataset[x]
        return a, b

    def pool_process(self, index_list):
        results = Parallel(n_jobs=self.num_works)(
            delayed(self.one_point)(index) for index in index_list
        )
        self.q.extend(results)

    def padding_batch(self, batch_list):
        max_array = max(batch_list, key=lambda x: x.shape[1])
        batch_data = []
        for arr in batch_list:
            batch_data_sample = np.zeros_like(max_array)
            hang, lie = arr.shape
            batch_data_sample[:hang, :lie] = arr
            batch_data.append(batch_data_sample)
        batch_data = np.array(batch_data)
        batch_data = np.expand_dims(batch_data, axis=1)
        batch_data = np.transpose(batch_data, (0, 3, 2, 1))
        return batch_data

    def collect_fn(self, batch_data):
        size = len(batch_data)
        input_mix_sig = []
        input_mix_spec = []
        target_clean_sig = []
        target_clean_spec = []
        for i in range(size):
            input_mix_sig.append(batch_data[i][0][0])
            input_mix_spec.append(batch_data[i][1][0])
            target_clean_sig.append(batch_data[i][0][1])
            target_clean_spec.append(batch_data[i][1][1])
        input_mix_array = self.padding_batch(input_mix_spec)
        target_clean_array = self.padding_batch(target_clean_spec)
        assert input_mix_array.shape == target_clean_array.shape
        return input_mix_array, target_clean_array, input_mix_sig, target_clean_sig

    def __iter__(self):
        if self.sampler is not None:
            if self.sampler.batch_size != self.batch_size:
                self.batch_size = self.sampler.batch_size
                print("Warrning: sampler.batch_size != batch_size. batch_size changed!")
            for index_list in self.sampler:
                # print("index_list", index_list)
                start_time = time.time()
                self.pool_process(index_list)
                end_time = time.time()
                # print("pool_process time:{}".format(end_time - start_time))
                results = self.q
                start_time = time.time()
                batch_mix, batch_clean, mix_sig, clean_sig = self.collect_fn(results)
                end_time = time.time()
                # print("collect_fn time:{}".format(end_time - start_time))
                self.q = []
                yield batch_mix, batch_clean, mix_sig, clean_sig
        else:
            for index_list in self.bins:
                # print("index_list", index_list)
                # print("index file", self.dataset.item_list[index_list[0]])
                self.pool_process(index_list)
                results = self.q
                batch_mix, batch_clean, mix_sig, clean_sig = self.collect_fn(results)
                self.q = []
                yield batch_mix, batch_clean, mix_sig, clean_sig

    def __len__(self):
        if self.sampler is not None:
            return len(self.sampler)
        else:
            return len(self.bins)

    def shuffle(self):
        self.dataset.shuffle()
