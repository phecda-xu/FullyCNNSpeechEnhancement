# coding: utf-8
#
#
#
#
#
import tensorflow as tf
import tensorflow.contrib.slim as slim


def conv_bn_relu(inputs, out_channels, kernel_size, stride, is_training, padding='SAME',
                 activate=tf.nn.relu, use_norm=True, scope='conv', skip_input=None):
    with slim.arg_scope([slim.convolution2d],
                        activation_fn=None,
                        weights_initializer=slim.initializers.xavier_initializer(),
                        biases_initializer=slim.init_ops.zeros_initializer()):
        with slim.arg_scope([slim.batch_norm],
                            is_training=is_training,
                            decay=0.96,
                            updates_collections=None,
                            activation_fn=tf.nn.relu):
            conv = slim.conv2d(inputs, out_channels, kernel_size, stride, padding,
                               activation_fn=activate, scope=scope)
            if use_norm:
                conv = slim.batch_norm(conv, scope=scope + '/batch_norm')
            if skip_input is not None:
                conv = conv + skip_input
    return conv


def separable_conv(inputs, out_channels, kernel_size, stride, is_training, scope='SC'):
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
                                                          scope=scope + '/dw_conv')

            bn = slim.batch_norm(depthwise_conv, scope=scope + '/dw_conv/batch_norm')
            pointwise_conv = slim.convolution2d(bn,
                                                out_channels,
                                                kernel_size=[1, 1],
                                                scope=scope + '/pw_conv')
            bn = slim.batch_norm(pointwise_conv, scope=scope + '/pw_conv/batch_norm')
    return bn



