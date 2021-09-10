# coding:utf-8
#
#
#
#

import os
import argparse
import numpy as np
import soundfile as sf
import tensorflow as tf
from config import load_conf_info
from data_utils.data_loader import AudioParser
from model_utils.tester import BaseTester
from model_utils.utils import AudioReBuild
from model_utils.model import FullyCNNSEModel, FullyCNNSEModelV2, FullyCNNSEModelV3


class InferenceEngine(BaseTester):
    def __init__(self, infer_config):
        super(InferenceEngine, self).__init__(infer_config)
        self.sample_rate = int(infer_config.get("data", 'sample_rate'))
        self.feature_dim = int(infer_config.get('data', 'feature_dim'))
        self.audio_save_path = infer_config.get('data', 'audio_save_path')
        self.window_ms = int(infer_config.get("data", "window_ms"))
        self.stride_ms = int(infer_config.get("data", "stride_ms"))
        self.net_arch = infer_config.get('model', 'net_arch')
        self.net_work = infer_config.get('model', 'net_work')
        self.creat_graph()
        self._init_session()
        self._load_checkpoint()
        self.param_count()
        self.audio_parser = AudioParser(self.sample_rate, self.window_ms, self.stride_ms, use_complex=True)
        self.audio_rebuilder = AudioReBuild()

    def creat_graph(self):
        #
        self.input_x = tf.placeholder(shape=[None, None, self.feature_dim, 1],
                                      dtype=tf.float32,
                                      name="input")
        self.target_y = tf.placeholder(shape=[None, None, self.feature_dim, 1],
                                       dtype=tf.float32,
                                       name="target")
        #
        if self.net_work == "FullyCNNV2":
            self.model = FullyCNNSEModelV2(is_training=False)
        elif self.net_work == "FullyCNNV3":
            self.model = FullyCNNSEModelV3(is_training=False)
        else:
            print("net_work set default or not wright. Use FullyCNN")
            self.model = FullyCNNSEModel(is_training=False)
        self.pred = self.model(self.input_x)

    def denoise(self, audio_file):
        sig, sr = self.audio_parser.load_audio(audio_file)
        sig_length = len(sig)
        complex_spectrogram = self.audio_parser.parse_audio(sig)
        mag = self.audio_parser.extractor.power_spectrum(complex_spectrogram)
        mag = np.reshape(mag, (1, mag.shape[1], mag.shape[0], 1))
        phase = self.audio_parser.extractor.divide_phase(complex_spectrogram)
        phase = np.transpose(phase)
        feed_dict = {
            self.input_x: mag
        }
        pred = self.sess.run(self.pred, feed_dict=feed_dict)
        denoise = self.audio_rebuilder.rebuild_audio(sig_length,
                                                     pred.squeeze(),
                                                     phase,
                                                     self.sample_rate,
                                                     self.window_ms,
                                                     self.stride_ms)
        if not os.path.exists(self.audio_save_path):
            os.makedirs(self.audio_save_path)
        denoise_audio_path = os.path.join(self.audio_save_path,
                                          os.path.basename(audio_file).replace('.wav', '_de.wav'))
        sf.write(denoise_audio_path, denoise, samplerate=self.sample_rate)
        print("Saving denoise file to {}.".format(denoise_audio_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--cfg', default='', type=str, help='cfg file for infer')
    parser.add_argument('--audio-file', default='', type=str, help='audio to denoise')
    args = parser.parse_args()
    config = load_conf_info(args.cfg)
    model = InferenceEngine(config)

    audio_filepath = args.audio_file
    model.denoise(audio_filepath)
