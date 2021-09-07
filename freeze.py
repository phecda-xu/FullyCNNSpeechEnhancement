# coding:utf-8
#
#
#
#

import tensorflow as tf
from tensorflow.python.framework import graph_util
from model_utils.model import FullyCNNSEModel, FullyCNNSEModelV2, FullyCNNSEModelV3


class FreezeEngine(object):
    def __init__(self, net_work, feature_dim=129):
        self.net_work = net_work
        self.feature_dim = feature_dim

    def creat_graph(self):
        input_x = tf.placeholder(shape=[None, None, self.feature_dim, 1],
                                 dtype=tf.float32,
                                 name="input")
        #
        if self.net_work == "FullyCNNV2":
            model = FullyCNNSEModelV2(is_training=False)
        elif self.net_work == "FullyCNNV3":
            model = FullyCNNSEModelV3(is_training=False)
        else:
            model = FullyCNNSEModel(is_training=False)
        pred = model(input_x)
        return pred

    def freeze_graph(self, checkpoint_file, pb_file):
        if self.net_work == "FullyCNNV2":
            output_node_names = "decode_8/BiasAdd"
        elif self.net_work == "FullyCNNV3":
            output_node_names = "decode_8/BiasAdd"
        else:
            output_node_names = "decode_5/BiasAdd"
        pred = self.creat_graph()
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, checkpoint_file)
            output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
                sess=sess,
                input_graph_def=sess.graph_def,  # 等于:sess.graph_def
                output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开
            with tf.gfile.GFile(pb_file, "wb") as f:  # 保存模型
                f.write(output_graph_def.SerializeToString())  # 序列化输出
            print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


if __name__ == "__main__":
    freeze_tool = FreezeEngine(net_work="FullyCNNV2")
    checkpoint_file = "checkpoints/aishell_1/RCED_FullyCNNV2/RCED_FullyCNNV2_0_9.ckpt"
    pb_file = "checkpoints/aishell_1/RCED_FullyCNNV2/RCED_FullyCNNV2_0_9.pb"
    freeze_tool.freeze_graph(checkpoint_file, pb_file)
