# coding: utf-8
#
#
#
#
import os
import time
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.contrib import slim as slim
from tqdm import tqdm
from model_utils.model import FullyCNNSEModel, FullyCNNSEModelV2
from model_utils.utils import AudioReBuild, AverageMeter
from model_utils.utils import PESQ, STOI, SDR


class BaseTrainer(object):
    def __init__(self, train_config):
        self.base_checkpoint = train_config.get('training', "base_checkpoint_file")
        self.checkpoints_path = train_config.get('training', 'checkpoints_path')
        self.continue_train = train_config.getboolean('training', "continue_train")
        self.net_arch = train_config.get('model', 'net_arch')
        self.net_work = train_config.get('model', 'net_work')
        self.init_lr = float(train_config.get('training', "lr"))
        self.lr = self.init_lr
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate_input = tf.placeholder(
            tf.float32, [], name='learning_rate_input')

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

    def _init_saver(self, max_to_keep=10):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        if self.base_checkpoint != "":
            self.continue_from = self.base_checkpoint
        elif self.continue_train:
            self.continue_from = tf.train.latest_checkpoint(self.checkpoints_path
                                                            + '/{}_{}'.format(self.net_arch, self.net_work))
        else:
            self.continue_from = None

        if self.continue_from is not None and os.path.exists(self.continue_from):
            self.saver.restore(self.sess, self.continue_from)
            print('recover from checkpoint_file: {}'.format(self.continue_from))
        else:
            self.sess.run(tf.global_variables_initializer())
            print('variables_initial finished!')


    def noam_scheme(self, global_step, warmup_steps=4000.):
        '''Noam scheme learning rate decay
        init_lr: initial learning rate. scalar.
        global_step: scalar.
        warmup_steps: scalar. During warmup_steps, learning rate increases
            until it reaches init_lr.
        '''
        step = global_step + 1
        return self.init_lr * warmup_steps ** 0.5 * np.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    def param_count(self):
        total_params = tf.trainable_variables()
        for i in total_params:
            print('{} layer parameter numbers | {}'.format(i.name,
                                                           np.prod(tf.shape(i.value()).eval(session=self.sess))))
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval(session=self.sess)), total_params))
        print('\nTotal number of Parameters: {}\n'.format(num_params))

    def save_param(self, checkpoint_path):
        """
        :param checkpoint_path:
        :return:
        """
        print('Saving model to "{}".\n'.format(checkpoint_path))
        self.saver.save(self.sess, checkpoint_path)

    def save_summary(self, checkpoint_path, summary_set):
        """
        Args:
            checkpoint_path:
            summary_set:
        Returns:
        """
        summary_writer = tf.summary.FileWriter(checkpoint_path + '/{}'.format(summary_set),
                                               self.sess.graph)
        return summary_writer

    def save_graph_txt(self, checkpoint_path):
        tf.train.write_graph(self.sess.graph_def, checkpoint_path, 'fully_cnn.pbtxt')

    def train_step(self, *args):
        return NotImplementedError

    def train(self, *args):
        return NotImplementedError

    def valid_step(self, *args):
        return NotImplementedError

    def valid(self, *args):
        return NotImplementedError


class FullyCNNTrainer(BaseTrainer):
    def __init__(self, train_config):
        super(FullyCNNTrainer, self).__init__(train_config)
        self.sample_rate = int(train_config.get("data", 'sample_rate'))
        self.feature_dim = int(train_config.get('data', 'feature_dim'))
        self.batch_size = int(train_config.get('training', 'batch_size'))
        self.num_iter_print = int(train_config.get('training', 'num_iter_print'))
        self.audio_save_path = train_config.get('data', 'audio_save_path')
        self._init_session()
        self.creat_graph()
        self._init_summary()
        self._init_saver()
        self.param_count()
        self.train_loss = AverageMeter()
        self.pesq_score = AverageMeter()
        self.stoi_score = AverageMeter()
        self.sdr_score = AverageMeter()
        self.data_time = AverageMeter()
        self.batch_time = AverageMeter()
        if not os.path.exists(self.audio_save_path):
            os.makedirs(self.audio_save_path)

    def _init_summary(self):
        tf.summary.scalar('learning_rate', self.learning_rate_input)
        tf.summary.scalar('loss_value', self.loss_value)
        self.merged_summaries = tf.summary.merge_all()

    def l1_loss(self, target, current):
        return tf.reduce_sum(tf.abs(target - current)) / self.batch_size

    def l2_loss(self, target, current):
        return tf.reduce_sum(tf.square(target - current)) / self.batch_size

    def loss_fun(self, y, pred):
        # loss_value = tf.losses.absolute_difference(x, y)
        # loss_value = tf.losses.mean_squared_error(x, y) * 10
        # loss_value = tf.reduce_sum(loss_value)
        loss_value = self.l2_loss(y, pred)
        return loss_value

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
            self.model = FullyCNNSEModelV2(is_training=True)
        else:
            self.model = FullyCNNSEModel(is_training=True)
        self.pred = self.model(self.input_x)
        self.loss_value = self.loss_fun(self.input_x, self.pred)
        # batch_norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate_input)
            self.train_op = slim.learning.create_train_op(total_loss=self.loss_value,
                                                          optimizer=self.optimizer)

    def train_step(self, input_x, target_y):
        feed_dict = {
            self.input_x: input_x,
            self.target_y: target_y,
            self.learning_rate_input: self.lr,
        }
        batch_loss, train_summary, global_step, _ = self.sess.run([self.loss_value,
                                                                   self.merged_summaries,
                                                                   self.global_step,
                                                                   self.train_op],
                                                                  feed_dict=feed_dict)
        return batch_loss, train_summary, global_step

    def train(self, train_loader, valid_loader, epochs, logger):
        global_step = 0
        train_summary_writter = self.save_summary(self.checkpoints_path,
                                                  "train_summary_{}_{}".format(self.net_arch, self.net_work))
        if self.continue_from is not None:
            start_epoch = int(self.continue_from.split("_")[-2]) + 1
        else:
            start_epoch = 0
        for epoch in range(start_epoch, epochs):
            train_batch_id = 0
            total_train_aduios = 0
            total_train_loss = 0
            train_loader.shuffle()
            start_time = time.time()
            for index, (batch_mix, batch_clean, _, _) in enumerate(train_loader):
                train_batch_id += 1
                total_train_aduios += train_loader.batch_size

                self.data_time.update(time.time() - start_time)
                start_time = time.time()
                batch_loss, train_summary, global_step = self.train_step(batch_mix, batch_clean)
                self.lr = self.noam_scheme(global_step=global_step, warmup_steps=1000)
                total_train_loss += batch_loss
                self.train_loss.update(batch_loss, n=1)
                train_summary_writter.add_summary(train_summary, global_step)
                end_time = time.time()
                self.batch_time.update(end_time - start_time)
                if train_batch_id % self.num_iter_print == 0:
                    print("epoch: {}, batch: {}/{}, "
                          "TrainLoss: {train_loss.val:.4f}({train_loss.avg:.4f}), "
                          "DataTime: {data_time.val:.3f}({data_time.avg:.3f}), "
                          "BatchTime: {batch_time.val:.3f}({batch_time.avg:.3f})".format(epoch,
                                                                                         train_batch_id,
                                                                                         len(train_loader),
                                                                                         train_loss=self.train_loss,
                                                                                         data_time=self.data_time,
                                                                                         batch_time=self.batch_time))
                start_time = time.time()
            if not os.path.exists(self.checkpoints_path + '/{}_{}'.format(self.net_arch, self.net_work)):
                os.makedirs(self.checkpoints_path + '/{}_{}'.format(self.net_arch, self.net_work))
            checkpoint_path = os.path.join(self.checkpoints_path + '/{}_{}'.format(self.net_arch, self.net_work),
                                           "{}_{}_{}_{}.ckpt".format(self.net_arch,
                                                                     self.net_work,
                                                                     epoch,
                                                                     global_step-1))
            self.save_param(checkpoint_path)
            if epoch == 0:
                self.save_graph_txt(checkpoint_path)
            if (epoch + 1) % 5 == 0:
                self.valid(valid_loader, epoch, logger)

    def valid_step(self, input_x):
        feed_dict = {
            self.input_x: input_x
        }
        output = self.sess.run(self.pred, feed_dict=feed_dict)
        return output

    def valid(self, valid_loader, epoch, logger):
        rebuilder = AudioReBuild()
        Pesq = PESQ(sr=self.sample_rate)
        Stoi = STOI(sr=self.sample_rate)
        Sdr = SDR()
        window_ms = valid_loader.dataset.window_s * 1000
        stride_ms = valid_loader.dataset.stride_s * 1000
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for index, (batch_mix, batch_clean, mix_sig, clean_sig) in pbar:
            audio_bins = valid_loader.bins[index]
            pbar.set_description("Epoch %s Validating %s"%(epoch, index))
            batch_mag = valid_loader.dataset.extractor.power_spectrum(batch_mix)
            batch_phase = valid_loader.dataset.extractor.divide_phase(batch_mix)
            pred_mag = self.valid_step(batch_mag)
            for i in range(self.batch_size):
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
                epoch_save_path = os.path.join(self.audio_save_path, str(epoch))
                if not os.path.exists(epoch_save_path):
                    os.makedirs(epoch_save_path)
                new_save_clean_path = os.path.join(epoch_save_path,
                                                   os.path.basename(clean_audio_path))
                mix_audio_path = os.path.join(epoch_save_path,
                                              os.path.basename(clean_audio_path).replace('.wav', '_mix.wav'))
                denoise_audio_path = os.path.join(epoch_save_path,
                                                  os.path.basename(clean_audio_path).replace('.wav', '_de.wav'))
                sf.write(new_save_clean_path, clean, samplerate=self.sample_rate)
                sf.write(mix_audio_path, mix, samplerate=self.sample_rate)
                sf.write(denoise_audio_path, denoise, samplerate=self.sample_rate)
            pbar.set_postfix(PESQ=self.pesq_score.avg, STOI=self.stoi_score.avg, SDR=self.sdr_score.avg)
        print("Epoch: {}, Average p_score: {:.4f}; "
              "Average st_score: {:.4f}; "
              "Average sd_score: {:.4f}.\n".format(epoch,
                                                   self.pesq_score.avg,
                                                   self.stoi_score.avg,
                                                   self.sdr_score.avg))
        logger.info("Epoch: {}, Average p_score: {:.4f}; "
                    "Average st_score: {:.4f}; "
                    "Average sd_score: {:.4f}.\n".format(epoch,
                                                         self.pesq_score.avg,
                                                         self.stoi_score.avg,
                                                         self.sdr_score.avg))
