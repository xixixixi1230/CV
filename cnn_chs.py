#采用cnn模型 有两个py文件来训练模型：cnn_chs.py(汉字)和cnn_eng.py(英文和字母) 路径以这个文件里的为准
#有尝试过：把汉字和字母数字放在一起训练 准确度也还行 但是会发生把数字识别成汉字的错误 就干脆分开训练了
import os
import numpy as np
import cv2 as cv
import joblib
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

#模型保存路径
MODEL_PATH = 'model_chs/cnn_chs/chs.ckpt'
TRAIN_DIR = 'data_chs/'
TEST_DIR = 'data_chs/'

#重置图片大小 值不能乱改
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
CLASSIFICATION_COUNT = 31 #共31类
LABEL_DICT = { #标签 必须从0开始 奇怪 好像之前从34开始的时候也对了 不记得改哪了
    'chuan':0, 'e':1, 'gan':2, 'gan1':3, 'gui':4, 'gui1':5, 'hei':6, 'hu':7, 'ji':8, 'jin':9,
	'jing':10, 'jl':11, 'liao':12, 'lu':13, 'meng':14, 'min':15, 'ning':16, 'qing':17,	'qiong':18, 'shan':19,
	'su':20, 'sx':21, 'wan':22, 'xiang':23, 'xin':24, 'yu':25, 'yu1':26, 'yue':27, 'yun':28, 'zang':29,
	'zhe':30
}

#GPU设置
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

#加载数据集 逐一遍历所有文件夹下的图片 注意层数！！（套娃）
def load_data(dir_path): #一定要注意 路径和图片名称中最好不要包含中文！！！
    data = []
    labels = []
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        for item1 in os.listdir(item_path):
            item1_path = os.path.join(item_path, item1)
            for item2 in os.listdir(item1_path):
                item2_path = os.path.join(item1_path, item2)
                if os.path.isdir(item2_path):
                    for subitem in os.listdir(item2_path):
                        subitem_path = os.path.join(item2_path, subitem)
                        gray_image = cv.imread(subitem_path, cv.IMREAD_GRAYSCALE)
                        resized_image = cv.resize(gray_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                        data.append(resized_image.ravel())
                        labels.append(LABEL_DICT[item2])
    return np.array(data), np.array(labels)

#数据的正则化
def normalize_data(data):
    return (data - data.mean()) / data.max()

#独热编码 唯一性
def onehot_labels(labels):
    onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
    for i in np.arange(len(labels)):
        onehots[i, (labels[i])] = 1
    return onehots

#权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

##开始构建神经网络的基本结构（前向）
#卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])#特征矩阵x
y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT])
x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

#第一层
W_conv1 = weight_variable([5, 5, 1, 32])  #32个
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  #采用ReLu激活函数
h_pool1 = max_pool_2x2(h_conv1)  #池化 压缩模型

#第二层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 10x10
h_pool2 = max_pool_2x2(h_conv2)  # 10x10 => 5x5

#全连接层 可提升准确率
W_fc1 = weight_variable([5 * 5 * 64, 1024])  #第一个隐藏层 1024个神经元
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  #ReLU

#Dropout 操作 防止过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接神经网络输出层
W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT])
b_fc2 = bias_variable([CLASSIFICATION_COUNT])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#尝试了多个learning_rate和max_epochs的值，发现了识别准确率根据其大小变化的规律 大致为凸函数 epoch不能追求过大 可能导致过拟合
learning_rate = 1e-4 #-4和-5都试过了 还是-4吧
max_epochs = 32
batch_size = 50
check_step = 10

#交叉损失熵
logits = y_conv
y = tf.nn.softmax(logits=logits)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))

#反向传播BP
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#评估一下
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #单个字符识别的准确率基本大概在0.95以上

#开始训练 对话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("装载训练数据...")
    train_data, train_labels = load_data(TRAIN_DIR)
    train_data = normalize_data(train_data) #正则化
    train_labels = onehot_labels(train_labels)
    print("装载%d条数据，每条数据%d个特征" % (train_data.shape)) #好多数据
    train_samples_count = len(train_data)
    train_indicies = np.arange(train_samples_count)
    np.random.shuffle(train_indicies) #shuffle一下
    print("装载测试数据...")
    test_data, test_labels = load_data(TEST_DIR)
    test_data = normalize_data(test_data)
    test_labels = onehot_labels(test_labels)
    print("装载%d条数据，每条数据%d个特征" % (test_data.shape))
    iters = int(np.ceil(train_samples_count / batch_size)) #有必要取整 batch_size不一定被整除
    print("Training...")
    for epoch in range(1, max_epochs + 1):
        print("Epoch #", epoch)
        for i in range(1, iters + 1):
            start_idx = (i * batch_size) % train_samples_count
            idx = train_indicies[start_idx: start_idx + batch_size]
            batch_x = train_data[idx, :]
            batch_y = train_labels[idx, :]
            _, batch_accuracy = sess.run([train_step, accuracy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
            if i % check_step == 0:
                print("Iter:", i, "of", iters, "batch_accuracy=", batch_accuracy)
    print("Training completed.")

    #保存模型 这里存为.ckpt 在ckpt_convert_to_pb.py中固化为了pb格式
    print("Saving model...")
    saver = tf.train.Saver(max_to_keep=1)
    saved_file = saver.save(sess, MODEL_PATH,global_step=epoch)
    print('Model saved to ', saved_file)

    #测试一下准确率 基本在95%以上
    test_accuracy = accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})
    print('Test accuracy %g' % test_accuracy)


