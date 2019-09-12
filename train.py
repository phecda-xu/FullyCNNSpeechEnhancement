import os
import argparse
import time
import prepare_data as pp_data
from data_generator import DataGenerator
from keras.models import Model
from keras.layers import Add, Activation, Reshape, Conv2D, Input, BatchNormalization
import tensorflow as tf
from keras import optimizers
import numpy as np


def train(args):
    """Train the neural network. Write out model every several iterations.

    Args:
      workspace: str, path of workspace.
      tr_snr: float, training SNR.
      te_snr: float, testing SNR.
      lr: float, learning rate.
    """
    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    lr = args.lr
    # Load data.
    t1 = time.time()
    tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "data.h5")
    te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "%ddb" % int(te_snr), "data.h5")
    (tr_x, tr_y) = pp_data.load_hdf5(tr_hdf5_path)
    (te_x, te_y) = pp_data.load_hdf5(te_hdf5_path)
    print(tr_x.shape, tr_y.shape)
    print(te_x.shape, te_y.shape)
    print("Load data time: %s s" % (time.time() - t1,))
    batch_size = 128
    print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))
    # Build model
    _, n_freq = tr_x.shape
    # encode
    T = 1
    data = Input(shape=[n_freq])
    x = Reshape([1, T, n_freq])(data)
    x1 = Conv2D(12, (T, 13), strides=(10, 1), data_format='channels_first', padding='same')(x)
    x1 = BatchNormalization(axis=-1)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(16, (T, 11), strides=(10, 1), data_format='channels_first', padding='same')(x1)
    x2 = BatchNormalization(axis=-1)(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(20, (1, 9), strides=(10, 1), data_format='channels_first', padding='same')(x2)
    x3 = BatchNormalization(axis=-1)(x3)
    x3 = Activation('relu')(x3)

    x4 = Conv2D(24, (1, 7), strides=(10, 1), data_format='channels_first', padding='same')(x3)
    x4 = BatchNormalization(axis=-1)(x4)
    x4 = Activation('relu')(x4)

    x5 = Conv2D(32, (1, 7), strides=(10, 1), data_format='channels_first', padding='same')(x4)
    x5 = BatchNormalization(axis=-1)(x5)
    x5 = Activation('relu')(x5)
    # decode
    y1 = Conv2D(24, (1, 7), strides=(10, 1), data_format='channels_first', padding='same')(x5)
    y1 = Add()([y1, x4])
    y1 = BatchNormalization(axis=-1)(y1)
    y1 = Activation('relu')(y1)

    y2 = Conv2D(20, (1, 9), strides=(10, 1), data_format='channels_first', padding='same')(y1)
    y2 = Add()([y2, x3])
    y2 = BatchNormalization(axis=-1)(y2)
    y2 = Activation('relu')(y2)

    y3 = Conv2D(16, (1, 11), strides=(10, 1), data_format='channels_first', padding='same')(y2)
    y3 = Add()([y3, x2])
    y3 = BatchNormalization(axis=-1)(y3)
    y3 = Activation('relu')(y3)

    y4 = Conv2D(12, (1, 13), strides=(10, 1), data_format='channels_first', padding='same')(y3)
    y4 = Add()([y4, x1])
    y4 = BatchNormalization(axis=-1)(y4)
    y4 = Activation('relu')(y4)

    y5 = Conv2D(1, (1, n_freq), strides=(10, 1), data_format='channels_first', padding='same')(y4)
    # y5 = BatchNormalization(axis=-1)(y5)
    y5 = Activation('relu')(y5)

    out = Reshape([n_freq])(y5)
    model = Model(inputs=data, outputs=out)
    adam = optimizers.Adam(lr=lr)
    model.compile(loss='mean_absolute_error', optimizer=adam)
    model.summary()

    # Data generator.
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=200)
    # Directories for saving models and training stats
    model_dir = os.path.join(workspace, "models", "%ddb" % int(tr_snr))
    pp_data.create_folder(model_dir)
    # Train.
    t1 = time.time()
    model.fit_generator(tr_gen.generate(xs=[tr_x], ys=[tr_y]), validation_data=te_gen.generate(xs=[te_x], ys=[te_y]),
                        validation_steps=100, steps_per_epoch=200, epochs=100)
    print("Training complete.")
    model_name = 'FullyCNN.h5'
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    print("Training time: %s s" % (time.time() - t1,))


# 定义权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 定义偏置
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


# 定义卷积
def conv2d(x, W):
    # srtide[1,x_movement,y_,movement,1]步长参数说明
    return tf.nn.conv2d(x, W, strides=[1, 1, 2, 1], padding='VALID')


def MSE_cost(out, Y):
    cost = tf.reduce_mean(tf.square(out-Y))
    return cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str,  default='workspace')
    parser_train.add_argument('--tr_snr', type=float,  default=0)
    parser_train.add_argument('--te_snr', type=float,  default=0)
    parser_train.add_argument('--lr', type=float,  default=0.0001)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        raise Exception("Error!")
