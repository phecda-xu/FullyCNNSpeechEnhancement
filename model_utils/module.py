# coding: utf-8
#
#
#
#
#
import tensorflow as tf
import tensorflow.contrib.slim as slim


def conv_bn_relu(inputs, out_channels, kernel_size, stride, is_training, padding='SAME',
                 use_norm=True, use_act=True, scope='conv', skip_input=None):
    """
    Conv layer with batch_norm and relu.
    :param inputs:  input tensor with shape as [batch, height, width, in_channels]
    :param out_channels: int
    :param kernel_size: tuple as (3,3)
    :param stride:  tuple as (1,1)
    :param is_training: True or False
    :param padding: "SAME" or "VALID"
    :param use_norm: True or False
    :param use_act: True or False
    :param scope: layer name
    :param skip_input: skip connection or resdual connection
    :return: tensor with shape as [batch, height, width, in_channels]
    """
    conv = tf.layers.conv2d(inputs, out_channels, kernel_size, stride, padding, name=scope)
    if use_norm:
        conv = tf.layers.batch_normalization(conv, training=is_training, name=scope + '/batch_norm')
    if skip_input is not None:
        conv = conv + skip_input
    if use_act:
        conv = tf.nn.relu(conv)
    return conv


def separable_conv(inputs, out_channels, kernel_size, stride, is_training, scope='SC', skip_input=None):
    """
    Separable Conv with batch_norm and relu.
    :param inputs:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param is_training:
    :param scope:
    :param skip_input:
    :return:
    """
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        activation_fn=None,
                        weights_initializer=slim.initializers.xavier_initializer(),
                        biases_initializer=slim.init_ops.zeros_initializer()):
        with slim.arg_scope([slim.batch_norm],
                            is_training=is_training,
                            decay=0.96,
                            updates_collections=None,
                            activation_fn=tf.nn.relu):
            depthwise_conv = slim.separable_convolution2d(inputs,
                                                          num_outputs=None,
                                                          stride=stride,
                                                          depth_multiplier=1,
                                                          kernel_size=kernel_size,
                                                          padding='VALID',
                                                          scope=scope + '/dw_conv')

            bn = slim.batch_norm(depthwise_conv, scope=scope + '/dw_conv/batch_norm')
            pointwise_conv = slim.convolution2d(bn,
                                                out_channels,
                                                kernel_size=[1, 1],
                                                scope=scope + '/pw_conv')
            if skip_input is not None:
                pointwise_conv = pointwise_conv + skip_input
            bn = slim.batch_norm(pointwise_conv, scope=scope + '/pw_conv/batch_norm')
    return bn
