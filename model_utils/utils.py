# coding: utf-8
#
#
#
#
import os
import librosa
import numpy as np
from pypesq import pesq
from pystoi import stoi


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class PESQ(object):
    def __init__(self, sr=16000):
        self.sr = sr

    def __call__(self, a, b):
        """
        :param a: 时域信号
        :param b: 时域信号
        :return:
        """
        assert len(a.shape) == 1
        assert len(a) == len(b)
        score = pesq(a, b, self.sr)
        return score


class STOI(object):
    def __init__(self, sr=16000):
        self.sr = sr

    def __call__(self, a, b):
        """
        :param a: 时域信号
        :param b: 时域信号
        :return:
        """
        assert len(a.shape) == 1
        assert len(a) == len(b)
        score = stoi(a, b, self.sr)
        return score


class SDR(object):
    def __init__(self):
        pass

    def sdr(self, y, y_pred):
        """
        :param y:  时域信号
        :param y_pred: 时域信号
        :return:
        """
        assert len(y.shape) == 1
        assert len(y) == len(y_pred)
        # y_en = np.power(y, 2).sum()
        # y_pred_en = np.power(y_pred-y, 2).sum()
        # sdr_score = 10 * np.log10(y_en / (y_pred_en + np.finfo(np.float32).eps))

        y_en = np.sum(y ** 2, axis=-1, keepdims=True)
        optimal_scaling = np.sum(y * y_pred, axis=-1, keepdims=True) / y_en
        y = optimal_scaling * y
        noise = y_pred-y
        ratio = np.sum(y ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
        sdr_score = 10 * np.log10(ratio)
        return sdr_score

    def __call__(self, x, y):
        score = self.sdr(x, y)
        return score


class AudioReBuild(object):
    def __init__(self, windows_name=None):
        windows = {
            'hamming': np.hamming,
            'hanning': np.hanning,
            'blackman': np.blackman,
            'bartlett': np.bartlett
        }
        self.window = windows.get(windows_name, windows['hamming'])

    def de_emphasis(self, signal):
        pre_emphasis = 0.97
        de_emphasized_signal = [signal[0]]
        for i in range(1, len(signal)):
            sample_value = signal[i] + de_emphasized_signal[i - 1] * pre_emphasis
            de_emphasized_signal.append(sample_value)
        return de_emphasized_signal

    def ifft(self, x, window_size):
        x = np.fft.irfft(x, window_size)
        return x

    def merge_magphase(self, magnitude, phase):
        """merge magnitude (S) and phase (P) to a complex-valued stft D so that `D = S * P`."""
        complex_sig = magnitude * phase
        return complex_sig

    def de_window(self, frame_sig, frame_length):
        with_window = self.window(frame_length)
        frame_sig_new = []
        for frame in frame_sig:
            frame_sig_new.append(frame / with_window)
        return np.array(frame_sig_new)

    def de_frame(self, frame_sig, frame_stride):
        sig = []
        sig.extend(frame_sig[0])
        for index in range(1, len(frame_sig)):
            stride_cache = frame_sig[index][frame_stride:]
            sig.extend(stride_cache)
        return np.array(sig)

    @staticmethod
    def en_frame(frame_size, frame_stride, sample_rate, signal):
        """
        # TODO 分帧
        :param frame_size:        帧大小,单位s,float
        :param frame_stride:      步大小,单位s,float
        :param sample_rate:       采样率,int
        :param signal:            音频数据,list
        :return:                  分帧后的音频数据,array
        """
        frame_length = int(round(frame_size * sample_rate))
        frame_step = int(round(frame_stride * sample_rate))
        signal_length = len(signal)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step + 1))
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(signal, z)
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
            np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]
        return frame_length, frames

    def rebuild_audio(self, sig_length, spec, phase, sample_rate, windows_ms, stride_ms):
        n_window = int((windows_ms * sample_rate) / 1000)
        n_overlap = int((stride_ms * sample_rate) / 1000)
        hop_size = n_window - n_overlap
        stft_reconstructed_clean = self.merge_magphase(spec, phase)
        signal_reconstructed_frame = self.ifft(stft_reconstructed_clean, n_window)
        signal_reconstructed_frame = self.de_window(signal_reconstructed_frame, n_window)
        signal_reconstructed_emphasized = self.de_frame(signal_reconstructed_frame, hop_size)
        signal_reconstructed_clean = self.de_emphasis(signal_reconstructed_emphasized)
        return np.array(signal_reconstructed_clean[:sig_length])
