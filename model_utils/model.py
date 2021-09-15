# coding: utf-8

from model_utils.module import *


class FullyCNNSEModel(object):
    def __init__(self, is_training):
        self.is_training = is_training

    def encode(self, x):
        self.encode_1 = conv_bn_relu(x, 12, kernel_size=(8, 13), stride=(1, 1), is_training=self.is_training, scope="encode_1")
        self.encode_2 = conv_bn_relu(self.encode_1, 16, kernel_size=(1, 11), stride=(1, 1), is_training=self.is_training, scope="encode_2")
        self.encode_3 = conv_bn_relu(self.encode_2, 20, kernel_size=(1, 9), stride=(1, 1), is_training=self.is_training, scope="encode_3")
        self.encode_4 = conv_bn_relu(self.encode_3, 24, kernel_size=(1, 7), stride=(1, 1), is_training=self.is_training, scope="encode_4")
        encode_5 = conv_bn_relu(self.encode_4, 32, kernel_size=(1, 7), stride=(1, 1), is_training=self.is_training, scope="encode_8")
        return encode_5

    def decode(self, x):
        x = conv_bn_relu(x, 24, kernel_size=(1, 7), stride=(1, 1), is_training=self.is_training, scope="decode_1", skip_input=self.encode_4)
        x = conv_bn_relu(x, 20, kernel_size=(1, 9), stride=(1, 1), is_training=self.is_training, scope="decode_2", skip_input=self.encode_3)
        x = conv_bn_relu(x, 16, kernel_size=(1, 11), stride=(1, 1), is_training=self.is_training, scope="decode_3", skip_input=self.encode_2)
        x = conv_bn_relu(x, 12, kernel_size=(1, 13), stride=(1, 1), is_training=self.is_training, scope="decode_4", skip_input=self.encode_1)
        x = conv_bn_relu(x, 1, kernel_size=(1, 129), stride=(1, 1), is_training=self.is_training, scope="decode_5", use_norm=False, use_act=False)
        return x

    def __call__(self, x):
        encode_out = self.encode(x)
        decode_out = self.decode(encode_out)
        return decode_out


class FullyCNNSEModelV2(object):
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


class FullyCNNSEModelV3(object):
    def __init__(self, is_training):
        self.is_training = is_training

    def simple_RCED(self, x, first_kernel, name, skip_input=None):
        encode_1 = conv_bn_relu(x, 18, kernel_size=first_kernel, stride=(1, 1), is_training=self.is_training,
                                scope="{}_encode_1".format(name))
        encode_2 = conv_bn_relu(encode_1, 30, kernel_size=(1, 5), stride=(1, 1), is_training=self.is_training,
                                scope="{}_encode_2".format(name))
        encode_3 = conv_bn_relu(encode_2, 8, kernel_size=(1, 9), stride=(1, 1), is_training=self.is_training,
                                scope="{}_decode".format(name))
        if skip_input is not None:
            encode_3 = encode_3 + skip_input

        return encode_3

    def cascaded_encoder(self, x):
        self.c_encode_1 = self.simple_RCED(x, first_kernel=(8, 9), name="CE1")
        self.c_encode_2 = self.simple_RCED(self.c_encode_1, first_kernel=(1, 9), name="CE2")
        c_encode_3 = self.simple_RCED(self.c_encode_2, first_kernel=(1, 9), name="CE3")
        return c_encode_3

    def cascaded_decoder(self, x):
        x = self.simple_RCED(x, first_kernel=(1, 9), name="CD1", skip_input=self.c_encode_2)
        x = self.simple_RCED(x, first_kernel=(1, 9), name="CD2", skip_input=self.c_encode_1)
        x = conv_bn_relu(x, 1, kernel_size=(1, 129), stride=(1, 1), is_training=self.is_training, scope="decode_final",
                         use_norm=False, use_act=False)
        return x

    def __call__(self, x):
        encode_out = self.cascaded_encoder(x)
        decode_out = self.cascaded_decoder(encode_out)
        return decode_out
