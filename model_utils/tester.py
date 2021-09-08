# coding: utf-8
#
#
#
#
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import soundfile as sf
from model_utils.model import FullyCNNSEModel, FullyCNNSEModelV2, FullyCNNSEModelV3
from model_utils.utils import AudioReBuild, AverageMeter
from model_utils.utils import PESQ, STOI, SDR


class BaseTester(object):
    def __init__(self, test_config):
        self.checkpoint_file = test_config.get('testing', "checkpoint_filepath")
        self.net_arch = test_config.get('model', 'net_arch')
        self.net_work = test_config.get('model', 'net_work')

    def _init_session(self):
        """
        configure session
        """
        gpu_options = tf.GPUOptions(allow_growth=True)
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=gpu_options)
        self.sess = tf.Session(config=session_conf)
        self.sess.as_default()

    def _load_checkpoint(self):
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.restore(self.sess, self.checkpoint_file)
        print('recover from checkpoint_file: {}'.format(self.checkpoint_file))

    def param_count(self):
        total_params = tf.trainable_variables()
        for i in total_params:
            print('{} layer parameter numbers | {}'.format(i.name,
                                                           np.prod(tf.shape(i.value()).eval(session=self.sess))))
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval(session=self.sess)), total_params))
        print('\nTotal number of Parameters: {}\n'.format(num_params))


class FullyCNNTester(BaseTester):
    def __init__(self, test_config):
        super(FullyCNNTester, self).__init__(test_config)
        self.sample_rate = int(test_config.get("data", 'sample_rate'))
        self.feature_dim = int(test_config.get('data', 'feature_dim'))
        self.audio_save_path = test_config.get('data', 'audio_save_path')
        self.batch_size = int(test_config.get('testing', 'batch_size'))
        self.creat_graph()
        self._init_session()
        self._load_checkpoint()
        self.param_count()
        self.pesq_score = AverageMeter()
        self.stoi_score = AverageMeter()
        self.sdr_score = AverageMeter()
        if not os.path.exists(self.audio_save_path):
            os.makedirs(self.audio_save_path)

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

    def test_step(self, input_x):
        feed_dict = {
            self.input_x: input_x
        }
        output = self.sess.run(self.pred, feed_dict=feed_dict)
        return output

    def test(self, valid_loader):
        rebuilder = AudioReBuild()
        Pesq = PESQ(sr=self.sample_rate)
        Stoi = STOI(sr=self.sample_rate)
        Sdr = SDR()
        window_ms = valid_loader.dataset.window_s * 1000
        stride_ms = valid_loader.dataset.stride_s * 1000
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for index, (batch_mix, batch_clean, mix_sig, clean_sig) in pbar:
            audio_bins = valid_loader.bins[index]
            # print("audio_bins", audio_bins)
            # print("audio_bins", valid_loader.dataset.item_list[audio_bins[0]])
            pbar.set_description("Testing %s" % (index))
            batch_mag = valid_loader.dataset.extractor.power_spectrum(batch_mix)
            batch_phase = valid_loader.dataset.extractor.divide_phase(batch_mix)
            pred_mag = self.test_step(batch_mag)
            for i in range(len(audio_bins)):
                clean = clean_sig[i]
                mix = mix_sig[i]
                sig_length = len(clean)
                mag = pred_mag[i].squeeze()
                phase = batch_phase[i].squeeze()
                denoise = rebuilder.rebuild_audio(sig_length,
                                                  mag,
                                                  phase,
                                                  self.sample_rate,
                                                  window_ms,
                                                  stride_ms)

                p_score = Pesq(clean, denoise)
                st_score = Stoi(clean, denoise)
                sd_score = Sdr(clean, denoise)

                self.pesq_score.update(p_score)
                self.stoi_score.update(st_score)
                self.sdr_score.update(sd_score)

                clean_audio_path = valid_loader.dataset.item_list[audio_bins[i]]["audio_filepath"]
                # print("clean_audio_path ", clean_audio_path)
                new_save_clean_path = os.path.join(self.audio_save_path,
                                                   os.path.basename(clean_audio_path))
                mix_audio_path = os.path.join(self.audio_save_path,
                                              os.path.basename(clean_audio_path).replace('.wav', '_mix.wav'))
                denoise_audio_path = os.path.join(self.audio_save_path,
                                                  os.path.basename(clean_audio_path).replace('.wav', '_de.wav'))
                sf.write(new_save_clean_path, clean, samplerate=self.sample_rate)
                sf.write(mix_audio_path, mix, samplerate=self.sample_rate)
                sf.write(denoise_audio_path, denoise, samplerate=self.sample_rate)
            pbar.set_postfix(PESQ=self.pesq_score.avg, STOI=self.stoi_score.avg, SDR=self.sdr_score.avg)
        print("Average p_score: {:.4f}; "
              "Average st_score: {:.4f}; "
              "Average sd_score: {:.4f}.\n".format(self.pesq_score.avg,
                                                   self.stoi_score.avg,
                                                   self.sdr_score.avg))
