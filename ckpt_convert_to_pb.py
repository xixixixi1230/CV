from tensorflow.python.framework import graph_util
import tensorflow._api.v2.compat.v1 as tf
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()
# #model_path='model/cnn_enu/'
# #checkpoint = tf.train.get_checkpoint_state(model_path) #检查目录下ckpt文件状态是否可用
# #input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径:model/cnn_enu/enu.ckpt-30

def freeze_graph(input_checkpoint, output_graph):
    output_node_names = "save/restore_all"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  #获得默认的图
    input_graph_def = graph.as_graph_def()  #返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  #等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  #保存模型
            f.write(output_graph_def.SerializeToString())  #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

input_checkpoint = 'model_chs/cnn_chs/chs.ckpt-32'
out_pb_path = 'pb_models/frozen_model_chs.pb'
freeze_graph(input_checkpoint, out_pb_path)

# ckpt = r'model_chs/cnn_chs/chs.ckpt-32' 这一段用来找最后的节点
# # read node name way 1
# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
#     graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
#     node_list = [n.name for n in graph_def.node]
#     for node in node_list:
#         print("node_name", node)
