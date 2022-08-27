#开始预测
import os
from detect import *
import numpy as np
import cv2 as cv
import tensorflow._api.v2.compat.v1 as tf
from matplotlib.pyplot import imread,imsave

tf.disable_v2_behavior()
MODEL_PATH_CHS='./Project/model_chs/cnn_chs/chs.ckpt-32'
MODEL_PATH_ENG='./Project/model/cnn_enu/enu.ckpt-30'

#英文图片重置宽、高
IMAGE_WIDTH=20
IMAGE_HEIGHT=20
CLASSIFICATION_COUNT1=34#英文
CLASSIFICATION_COUNT2=31
CLASSIFICATION_COUNT=65

ENGLISH_LABELS=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z']
CHINESE_LABELS = [
	"川","鄂","赣","甘","贵","桂","黑","沪","冀","津",
	"京","吉","辽","鲁","蒙","闽","宁","青","琼","陕",
	"苏","晋","皖","湘","新","豫","渝","粤","云","藏",
	"浙"]

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

#再来一遍
#正则化
def normalize_data(data):
    return (data - data.mean()) / data.max()

#权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#字母数字识别模型
def load_model_ENG():
    print('load_model_eng')
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])
    y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT1])
    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    #cnn第一层
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #cnn第二层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 10x10
    h_pool2 = max_pool_2x2(h_conv2)

    #全连接层
    W_fc1 = weight_variable([5 * 5 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #神经网络输出层
    W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT1])
    b_fc2 = bias_variable([CLASSIFICATION_COUNT1])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    learning_rate = 1e-5
    max_epochs = 30
    batch_size = 50
    check_step = 10

    logits = y_conv
    y = tf.nn.softmax(logits=logits)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))

    #反向传播
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess=tf.Session()
    saver=tf.train.Saver()
    saver.restore(sess,MODEL_PATH_ENG)
    return(sess,x,keep_prob,y_conv)

#汉字识别模型
def load_model_CHS():
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])
    y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT2])
    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    #cnn第一层
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #cnn第二层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #全连接层
    W_fc1 = weight_variable([5 * 5 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #神经网络输出层
    W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT2])
    b_fc2 = bias_variable([CLASSIFICATION_COUNT2])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    learning_rate = 1e-4
    max_epochs = 32
    batch_size = 50
    check_step = 10

    logits = y_conv
    y = tf.nn.softmax(logits=logits)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess=tf.Session()
    saver=tf.train.Saver()
    saver.restore(sess,MODEL_PATH_CHS)
    return(sess,x,keep_prob,y_conv)

#predict函数 预测单个字符
def predict_eng(char_image,model):
    origin_height,origin_width=char_image.shape #原始大小
    resize_height=IMAGE_HEIGHT-2 if origin_height>IMAGE_HEIGHT else origin_height #resize一下
    resize_width=IMAGE_WIDTH-2 if origin_width>IMAGE_WIDTH else origin_width
    resize_image=cv.resize(char_image,(resize_width,resize_height))

    working_image=np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH))
    x_idx=(IMAGE_WIDTH-resize_width)//2
    y_idx=(IMAGE_HEIGHT-resize_height)//2
    working_image[y_idx:y_idx+resize_height,x_idx:x_idx+resize_width]=resize_image

    working_image=normalize_data(working_image)
    data=[]
    data.append(working_image.ravel())

    sess,x,keep_prob,y_conv=model
    results=sess.run(y_conv,feed_dict={x:data,keep_prob:1.0})
    predict=np.argmax(results[0])
    return ENGLISH_LABELS[predict]

def predict_chs(char_image,model):
    origin_height,origin_width=char_image.shape
    resize_height=IMAGE_HEIGHT-2 if origin_height>IMAGE_HEIGHT else origin_height
    resize_width=IMAGE_WIDTH-2 if origin_width>IMAGE_WIDTH else origin_width
    resize_image=cv.resize(char_image,(resize_width,resize_height))

    working_image=np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH))
    x_idx=(IMAGE_WIDTH-resize_width)//2
    y_idx=(IMAGE_HEIGHT-resize_height)//2
    working_image[y_idx:y_idx+resize_height,x_idx:x_idx+resize_width]=resize_image

    working_image=normalize_data(working_image)
    data=[]
    data.append(working_image.ravel())

    sess,x,keep_prob,y_conv=model
    results=sess.run(y_conv,feed_dict={x:data,keep_prob:1.0})
    predict=np.argmax(results[0])
    return CHINESE_LABELS[predict]
#以上将在detect.py中的detect函数中被调用 实现字符分割后与识别的连接：循环读入单个的字符 最后输出字符数组