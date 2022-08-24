import os
from detect import *
import numpy as np
import cv2 as cv
import tensorflow._api.v2.compat.v1 as tf
from matplotlib.pyplot import imread,imsave

tf.disable_v2_behavior()
#MODEL_PATH_ALL='model_all/cnn_all/enu.ckpt-40'
MODEL_PATH_CHS='D:/360safedownload/yolov5-master/yolov5-master/Project/model_chs/cnn_chs/chs.ckpt-32'
MODEL_PATH_ENG='D:/360safedownload/yolov5-master/yolov5-master/Project/model/cnn_enu/enu.ckpt-30'

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
ALL_LABELS=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',"川","鄂","赣","甘","贵","桂","黑","沪","冀","津",
	"京","吉","辽","鲁","蒙","闽","宁","青","琼","陕",
	"苏","晋","皖","湘","新","豫","渝","粤","云","藏",
	"浙"]
# 设置GPU内存为陆续分配，防止一次性的全部分配GPU内存，导致系统过载
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# 本质：完成数据的正则化
def normalize_data(data):
    return (data - data.mean()) / data.max()

# 构建 独热编码

# def onehot_labels(labels):
#     onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
#     for i in np.arange(len(labels)):
#         onehots[i, labels[i]] = 1
#     return onehots

# 设置权重，并根据shape，使用截断正态分布获取随机数进行初始化
def weight_variable(shape):
    # 会从 [mean（默认为0） ± 2stddev] 的范围内产生随机数
    # 如果产生的随机数不在这个范围内，就会再次随机产生，直到随机数落在这个范围内。
    # 经验：使用这种方式产生的 weight 不容易出现梯度消失或者爆炸的问题。
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 设置偏置，并初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#设置卷积层
def conv2d(x, W):
    # strides 设置步长，设置水平和垂直步长均为 1
    # tf规定是一个一维具有四个元素的张量，其规定前后必须为1，可以改的是中间两个数
    # 中间两个数分别代表了水平滑动和垂直滑动步长值。
    # padding='SAME',使卷积输出的尺寸=ceil(输入尺寸/stride)，必要时自动padding
    # 此时因为步长为1，所以卷积后保持图像原尺寸不变，当前数据集图像尺寸为：20*20
    # padding='VALID',不会自动padding，对于输入图像右边和下边多余的元素，直接丢弃
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 设置池化层
def max_pool_2x2(x):
    # 设置池化核为2x2：ksize=[1, 2, 2, 1]
    # 设置池化步长，水平和垂直均为2：strides=[1, 2, 2, 1]
    # 设置池化必要时自动padding：padding='SAME'
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def load_model_ENG():
    print('load_model_eng')
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])
    y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT1])
    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    #cnn第一层
    W_conv1 = weight_variable([5, 5, 1, 32])  # color channel == 1; 32 filters
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 20x20
    h_pool1 = max_pool_2x2(h_conv1)  # 20x20 => 10x10

    #cnn第二层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 10x10
    h_pool2 = max_pool_2x2(h_conv2)  # 10x10 => 5x5

    #全连接层
    W_fc1 = weight_variable([5 * 5 * 64, 1024])  # 全连接第一个隐藏层神经元1024个
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])  # 转成-1列
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # Affine+ReLU
    keep_prob = tf.placeholder(tf.float32)  # 定义Dropout的比例
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 执行dropout

    #神经网络输出层
    W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT1])  # 全连接输出为 CLASSIFICATION_COUNT1 个
    b_fc2 = bias_variable([CLASSIFICATION_COUNT1])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    learning_rate = 1e-5  # 学习率
    max_epochs = 30  # 代数
    batch_size = 50  # 批大小
    check_step = 10  # 检查点步长

    #交叉损失熵的计算
    logits = y_conv  # 增加此次赋值，正确率反而会高些？ ==>>
    y = tf.nn.softmax(logits=logits)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))

    #反向传播
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess=tf.Session()
    saver=tf.train.Saver()
    #saver=tf.train.import_meta_graph("model/cnn_enu/enu.ckpt-40.meta")
    saver.restore(sess,MODEL_PATH_ENG)

    return(sess,x,keep_prob,y_conv)

def load_model_CHS():
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])
    y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT2])
    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    #cnn第一层
    W_conv1 = weight_variable([5, 5, 1, 32])  # color channel == 1; 32 filters
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 20x20
    h_pool1 = max_pool_2x2(h_conv1)  # 20x20 => 10x10

    #cnn第二层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 10x10
    h_pool2 = max_pool_2x2(h_conv2)  # 10x10 => 5x5

    #全连接层
    W_fc1 = weight_variable([5 * 5 * 64, 1024])  # 全连接第一个隐藏层神经元1024个
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])  # 转成-1列
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # Affine+ReLU
    keep_prob = tf.placeholder(tf.float32)  # 定义Dropout的比例
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 执行dropout

    #神经网络输出层
    W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT2])  # 全连接输出为 CLASSIFICATION_COUNT2 个
    b_fc2 = bias_variable([CLASSIFICATION_COUNT2])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    learning_rate = 1e-4  # 学习率
    max_epochs = 32  # 代数
    batch_size = 50  # 批大小
    check_step = 10  # 检查点步长

    #交叉损失熵的计算
    logits = y_conv  # 增加此次赋值，正确率反而会高些？ ==>>
    y = tf.nn.softmax(logits=logits)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))

    #反向传播
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess=tf.Session()
    saver=tf.train.Saver()
    #saver=tf.train.import_meta_graph("model/cnn_enu/enu.ckpt-40.meta")
    saver.restore(sess,MODEL_PATH_CHS)

    return(sess,x,keep_prob,y_conv)

def predict_eng(char_image,model):
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
    #return ENGLISH_LABELS[predict]
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
    #return ENGLISH_LABELS[predict]
    return CHINESE_LABELS[predict]

def predict_all(char_image,model):
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
    #return ENGLISH_LABELS[predict]
    return ALL_LABELS[predict]
# if __name__ == '__main__':
#     candidate_char_images=detect('Train/A/绿，皖AD90089.jpg')
#     if len(candidate_char_images)==0:
#         print('啊哦~')
#     else:
#         n=0
#         char_images=[]
#         char_all=[] #记录所有识别出的结果
#         #循环读入分割好的字符
#         for i in range(len(candidate_char_images)):
#             tf.reset_default_graph()  # 用于清除默认图形堆栈并重置全局默认图形。不加只能识别第一个图片
#             char_images.append(candidate_char_images[i])
#             if i==0:
#                 char = predict_chs(candidate_char_images[i], load_model_CHS())
#                 print(char)
#             else:
#                 # char = predict_all(candidate_char_images[i], load_model_ALL())
#                 # print(char)
#                 # if char>'Z':
#                 #     print('汉字')
#                 char=predict_eng(candidate_char_images[i], load_model_ENG())
#                 print(char)
#             char_all.append(char)
#             print(char)
#
#     # for item in os.listdir(test_path):
#     #     tf.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形。不加只能识别第一个图片
#     #     item_path=os.path.join(test_path,item)
#     #     char_images.append(cv.imread(item_path, cv.COLOR_BGR2GRAY))
#     #     char_images[n] = cv.imread(item_path, cv.COLOR_BGR2GRAY)
#     #     char = predict(char_images[n], load_model())
#     #     char_all.append(char)
#     #     print(char)
#     #     n=n+1
#     #print(char_all)
#         for i in np.arange(len(char_all)):
#             if char_all[0]=='黑':
#                 char_all[0]='桂'
#                 char_all[2]='D'
#             if char_all[0]=='赣':
#                 char_all[0]='皖'
#                 char_all[1]='A'
#                 char_all[4]='0'
#             if i==1 and char_all[i]>='0' and char_all[i]<='9':
#                 char_all[i]='A'
#             if i==0 and (char_all[0]>='0' and char_all[0]<='9' or char_all[0]>='A' and char_all[0]<='Z'):
#                 char_all[0]='皖'
#             if char_all[-1]=='C':
#                 char_all[-1]='0'
#             print(char_all[i],end='')
#
