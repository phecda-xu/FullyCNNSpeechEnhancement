# coding:utf-8
#
#
#
#
# DEC:
#     audio feature extract function

import numpy as np


class AudioFeature(object):
    def __init__(self, windows_name=None):
        windows = {
            'hamming': np.hamming,
            'hanning': np.hanning,
            'blackman': np.blackman,
            'bartlett': np.bartlett
        }
        self.window = windows.get(windows_name, windows['hamming'])

    def compute_spectrogram(self,
                            signal,
                            sample_rate,
                            window_s=0.02,
                            stride_s=0.01,
                            nfft=512,
                            use_complex=False):
        if stride_s > window_s:
            raise ValueError("Stride size must not be greater than window size.")
        # 预加重
        emphasized_signal = self.pre_emphasis(signal)
        # 分帧
        frame_length, frames = self.en_frame(window_s, stride_s, sample_rate, emphasized_signal)
        # 加上汉明窗
        frames = self.add_windows(frame_length, frames)
        # 傅立叶变换
        fft_frames = self.fft(frames, nfft=nfft)  # nfft 通常为256 或者 512
        # 功率谱
        pow_frames = self.power_spectrum(fft_frames)
        if use_complex:
            return np.transpose(fft_frames)
        else:
            return np.transpose(pow_frames).astype(np.float32)

    @staticmethod
    def pre_emphasis(signal):
        """
        # TODO 预加重
        :param signal:         原始音频数据，array
        :return:               预加重后的音频数据，array
        """
        pre_emphasis = 0.97
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        return emphasized_signal

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

    def add_windows(self, frame_length, frames):
        """
        TODO
        :param frame_length:     每一帧音频采样点数,int
        :param frames:           帧音频数据,array.shape:(len(frames), frame_length)
        :return:                 shape:(len(frames), frame_length)
        """
        with_window = self.window(frame_length)  # 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))
        frames *= with_window  # (len(frames), frame_length) * (frame_length,) = (len(frames), frame_length)
        return frames

    @staticmethod
    def fft(frames, nfft=512):
        """
        TODO 傅里叶变换
        :param frames:            帧音频数据,array.shape:(len(frames), frame_length)
        :param nfft:              傅里叶变换,nfft通常取256或者512
        :return:                  shape:(numframes,257)或者(numframes,128)
        """
        fft_frames = np.fft.rfft(frames, nfft)
        return fft_frames

    @staticmethod
    def power_spectrum(frames):
        """
        TODO 功率谱
        :param frames:             shape:(numframes,257)或者(numframes,128)
        :return:                   shape:(numframes,257)或者(numframes,128)
        """
        mag_frames = np.absolute(frames)
        # pow_frames = ((1.0 / nfft) * (mag_frames ** 2))  # Power Spectrum
        return mag_frames

    @staticmethod
    def divide_phase(fft_frames):
        phase = np.exp(1.j * np.angle(fft_frames))
        return phase
