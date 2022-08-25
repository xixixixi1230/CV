##与汉字的训练基本相同 就不再做过多注释
import os
import numpy as np
import cv2 as cv
import joblib
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

#注意路径
MODEL_PATH = 'model/cnn_enu/enu.ckpt'
TRAIN_DIR = 'data/'
TEST_DIR = 'data/'

#部分车牌照片存在歪斜 导致分割好的字符可能会向左倾倒 所以在data_rotate.py中将部分训练集照片统一向左转了20° 再加入训练集
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
CLASSIFICATION_COUNT = 34
LABEL_DICT = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'J': 18, 'K': 19,
    'L': 20, 'M': 21, 'N': 22, 'P': 23, 'Q': 24, 'R': 25, 'S': 26, 'T': 27, 'U': 28, 'V': 29,
    'W': 30, 'X': 31, 'Y': 32, 'Z': 33
}

#GPU
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

def load_data(dir_path):
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

def normalize_data(data):
    return (data - data.mean()) / data.max()

def onehot_labels(labels):
    onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
    for i in np.arange(len(labels)):
        onehots[i, labels[i]] = 1
    return onehots

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])
y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT])
x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

#cnn第一层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# cnn第二层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层
W_fc1 = weight_variable([5 * 5 * 64, 1024])  #1024
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT])
b_fc2 = bias_variable([CLASSIFICATION_COUNT])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

learning_rate = 1e-4
max_epochs = 30 #修改训练集后发现不用迭代太多次就能达到较高准确率了
batch_size = 50
check_step = 10

logits = y_conv
y = tf.nn.softmax(logits=logits)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("装载训练数据...")
    train_data, train_labels = load_data(TRAIN_DIR)
    train_data = normalize_data(train_data)
    train_labels = onehot_labels(train_labels)
    print("装载%d条数据，每条数据%d个特征" % (train_data.shape))
    train_samples_count = len(train_data)
    train_indicies = np.arange(train_samples_count)
    np.random.shuffle(train_indicies)
    print("装载测试数据...")
    test_data, test_labels = load_data(TEST_DIR)
    test_data = normalize_data(test_data)
    test_labels = onehot_labels(test_labels)
    print("装载%d条数据，每条数据%d个特征" % (test_data.shape))
    iters = int(np.ceil(train_samples_count / batch_size))
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
    print("Saving model...")
    saver = tf.train.Saver(max_to_keep=1)
    saved_file = saver.save(sess, MODEL_PATH,global_step=epoch)
    print('Model saved to ', saved_file)
    test_accuracy = accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})
    print('Test accuracy %g' % test_accuracy)




