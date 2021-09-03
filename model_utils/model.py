# coding: utf-8

from model_utils.module import *


class FullyCNNSEModel(object):
    def __init__(self, is_training):
        self.is_training = is_training

    def encode(self, x):
        self.encode_1 = conv_bn_relu(x, 10, kernel_size=(8, 11), stride=(1, 1), is_training=self.is_training, scope="encode_1")
        self.encode_2 = conv_bn_relu(self.encode_1, 12, kernel_size=(1, 7), stride=(1, 1), is_training=self.is_training, scope="encode_2")
        self.encode_3 = conv_bn_relu(self.encode_2, 14, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="encode_3")
        self.encode_4 = conv_bn_relu(self.encode_3, 15, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="encode_4")
        self.encode_5 = conv_bn_relu(self.encode_4, 19, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="encode_5")
        self.encode_6 = conv_bn_relu(self.encode_5, 21, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="encode_6")
        self.encode_7 = conv_bn_relu(self.encode_6, 23, kernel_size=(1, 7), stride=(1, 1), is_training=self.is_training, scope="encode_7")
        encode_8 = conv_bn_relu(self.encode_7, 25, kernel_size=(1, 11), stride=(1, 1), is_training=self.is_training, scope="encode_8")
        return encode_8

    def decode(self, x):
        x = conv_bn_relu(x, 23, kernel_size=(1, 7), stride=(1, 1), is_training=self.is_training, scope="decode_1", skip_input=self.encode_7)
        x = conv_bn_relu(x, 21, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="decode_2", skip_input=self.encode_6)
        x = conv_bn_relu(x, 19, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="decode_3", skip_input=self.encode_5)
        x = conv_bn_relu(x, 15, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="decode_4", skip_input=self.encode_4)
        x = conv_bn_relu(x, 14, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="decode_5", skip_input=self.encode_3)
        x = conv_bn_relu(x, 12, kernel_size=(1, 7), stride=(1, 1), is_training=self.is_training, scope="decode_6", skip_input=self.encode_2)
        x = conv_bn_relu(x, 10, kernel_size=(1, 11), stride=(1, 1), is_training=self.is_training, scope="decode_7", skip_input=self.encode_1)
        x = conv_bn_relu(x, 1, kernel_size=(1, 129), stride=(1, 1), is_training=self.is_training, scope="decode_8", use_norm=False, use_act=False)
        return x

    def __call__(self, x):
        encode_out = self.encode(x)
        decode_out = self.decode(encode_out)
        return decode_out


class FullyCNNSEModelV2(object):
    def __init__(self, is_training):
        self.is_training = is_training

    def encode(self, x):
        self.encode_1 = conv_bn_relu(x, 10, kernel_size=(1, 1), stride=(1, 1), is_training=self.is_training, padding="Valid", scope="encode_1")
        self.encode_2 = conv_bn_relu(self.encode_1, 12, kernel_size=(8, 7), stride=(1, 1), is_training=self.is_training, scope="encode_2")
        self.encode_3 = conv_bn_relu(self.encode_2, 14, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="encode_3")
        self.encode_4 = conv_bn_relu(self.encode_3, 15, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="encode_4")
        self.encode_5 = conv_bn_relu(self.encode_4, 19, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="encode_5")
        self.encode_6 = conv_bn_relu(self.encode_5, 21, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="encode_6")
        self.encode_7 = conv_bn_relu(self.encode_6, 23, kernel_size=(1, 7), stride=(1, 1), is_training=self.is_training, scope="encode_7")
        encode_8 = conv_bn_relu(self.encode_7, 25, kernel_size=(1, 11), stride=(1, 1), is_training=self.is_training, scope="encode_8")
        return encode_8

    def decode(self, x):
        x = conv_bn_relu(x, 23, kernel_size=(1, 7), stride=(1, 1), is_training=self.is_training, scope="decode_1", skip_input=self.encode_7)
        x = conv_bn_relu(x, 21, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="decode_2", skip_input=self.encode_6)
        x = conv_bn_relu(x, 19, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="decode_3", skip_input=self.encode_5)
        x = conv_bn_relu(x, 15, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="decode_4", skip_input=self.encode_4)
        x = conv_bn_relu(x, 14, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="decode_5", skip_input=self.encode_3)
        x = conv_bn_relu(x, 12, kernel_size=(8, 7), stride=(1, 1), is_training=self.is_training, scope="decode_6", skip_input=self.encode_2)
        x = conv_bn_relu(x, 10, kernel_size=(1, 11), stride=(1, 1), is_training=self.is_training, scope="decode_7", skip_input=self.encode_1)
        x = conv_bn_relu(x, 1, kernel_size=(1, 15), stride=(1, 1), is_training=self.is_training, scope="decode_8", use_norm=False, use_act=False)
        return x

    def __call__(self, x):
        encode_out = self.encode(x)
        decode_out = self.decode(encode_out)
        return decode_out


class FullyCNNSEModelV3(object):
    def __init__(self, is_training):
        self.is_training = is_training

    def encode(self, x):
        self.encode_1 = separable_conv(x, 10, kernel_size=(1, 1), stride=(1, 1), is_training=self.is_training, scope="encode_1")
        self.encode_2 = conv_bn_relu(self.encode_1, 12, kernel_size=(4, 4), stride=(1, 1), is_training=self.is_training, scope="encode_2")
        self.encode_3 = conv_bn_relu(self.encode_2, 14, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="encode_3")
        self.encode_4 = conv_bn_relu(self.encode_3, 16, kernel_size=(1, 7), stride=(1, 1), is_training=self.is_training, scope="encode_7")
        encode_5 = conv_bn_relu(self.encode_4, 18, kernel_size=(1, 11), stride=(1, 1), is_training=self.is_training, scope="encode_8")
        return encode_5

    def decode(self, x):
        x = conv_bn_relu(x, 16, kernel_size=(1, 7), stride=(1, 1), is_training=self.is_training, scope="decode_1", skip_input=self.encode_4)
        x = conv_bn_relu(x, 14, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training, scope="decode_2", skip_input=self.encode_3)
        x = conv_bn_relu(x, 12, kernel_size=(1, 4), stride=(1, 1), is_training=self.is_training, scope="decode_3", skip_input=self.encode_2)
        x = conv_bn_relu(x, 10, kernel_size=(1, 11), stride=(1, 1), is_training=self.is_training, scope="decode_4", skip_input=self.encode_1)
        x = conv_bn_relu(x, 1, kernel_size=(1, 15), stride=(1, 1), is_training=self.is_training, scope="decode_5", use_norm=False, use_act=False)
        return x

    def __call__(self, x):
        encode_out = self.encode(x)
        decode_out = self.decode(encode_out)
        return decode_out