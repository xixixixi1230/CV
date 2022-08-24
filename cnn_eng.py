'''
使用简单卷积神经网络分类，网络结构为：
Conv -> ReLU -> Max Pooling -> Conv -> ReLU -> Max Pooling ->
FC1(1024) -> ReLU -> Dropout -> Affine -> Softmax
'''

import os
import numpy as np
import cv2 as cv
import joblib
# import tensorflow as tf
# import tensorflow.compat.v1 as tf
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()


MODEL_PATH = 'model/cnn_enu/enu.ckpt'
TRAIN_DIR = './data/enu_train/enu_train'
TEST_DIR = "./data/enu_test/enu_test"
# 英文图片重置的宽、高
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
CLASSIFICATION_COUNT = 34
LABEL_DICT = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'J': 18, 'K': 19,
    'L': 20, 'M': 21, 'N': 22, 'P': 23, 'Q': 24, 'R': 25, 'S': 26, 'T': 27, 'U': 28, 'V': 29,
    'W': 30, 'X': 31, 'Y': 32, 'Z': 33
}

# 设置GPU内存为陆续分配，防止一次性的全部分配GPU内存，导致系统过载
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

def load_data(dir_path):
    data = []
    labels = []
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                gray_image = cv.imread(subitem_path, cv.IMREAD_GRAYSCALE)
                resized_image = cv.resize(gray_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                data.append(resized_image.ravel())
                labels.append(LABEL_DICT[item])

    return np.array(data), np.array(labels)

# 本质：完成数据的正则化
def normalize_data(data):
    return (data - data.mean()) / data.max()

# 构建 独热编码
def onehot_labels(labels):
    onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
    for i in np.arange(len(labels)):
        onehots[i, labels[i]] = 1
    return onehots

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

# 设置卷积层
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

# x 是用来容纳训练数据样本集的特征矩阵，
# shape 形状定义为 [None, IMAGE_HEIGHT * IMAGE_WIDTH]。
x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])

# y_ 是用来容纳训练数据样本集的标签。
# shape 定义为 [None, CLASSIFICATION_COUNT] 。
y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT])

# 具体此处：
# 改变输入数据的形状，让它可以和卷积核进行卷积。
# 因为输入的数据 x 是一个 Nx(IMAGE_HEIGHT * IMAGE_WIDTH) 的张量，
# 所以需要先变为 M x IMAGE_HEIGHT x IMAGE_HEIGHT x 1 的形状才能进行运算。因为卷积操作要使用图片的矩阵形式进行。
x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

# cnn第一层，卷积核：5*5，颜色通道：1，共有32个卷积核，据此作为shape，调用 weight_variable 随机初始化cnn第一层的权重矩阵
# 具体此处：
# 参数为数组形式，表示 w 的形状，其实就定义了该层卷积核的大小和数量：
# 每个卷积核的大小为 5x5
# 输入通道数为 1，因为图片是灰度图。如果是用不带透明通道的 rgb 彩色图该值就设为 3，如果在带了透明通道的 rgba 彩色图该值就设为 4
# 输出通道数为 32，即该层有 32 个卷积核。
# 对于隐层中每个卷积层中的卷积核大小如何的确定，最好借鉴一些公开的网络模型，是经过反复尝试论证出来的。
W_conv1 = weight_variable([5, 5, 1, 32])  # color channel == 1; 32 filters
# cnn第一层，偏置也是32个，据此作为shape，调用 bias_variable 随机初始化cnn第一层的偏置矩阵
# 具体此处：
# b 的大小和卷积核个数相对应。
# 每个卷积核和输入卷积后，均需加上一个偏置量。
b_conv1 = bias_variable([32])
# tf.nn.relu()函数的目的是，将输入小于0的值幅值为0，输入大于0的值不变。
# relu 在第一象限就是 x，所以能够大量的减少计算，从而加速收敛。
# 同时能减小梯度消失发生的可能性，不过梯度爆炸还是可能会发生。
# 具体此处：
# 包含了 3 个操作：
# 1. conv2d(x_image, W_conv1)，将输入 x_image 与该层的所有卷积核 W_conv1 进行卷积运算
# 2. conv2d(x_image, W_conv1) + b_conv1，每次卷积后加上一个偏置量
# 3. tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)，
# 最后加上一个激活函数，增加非线性的变换。此处使用的激活函数是 ReLu 函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 20x20
# 加上一个池化层，压缩权值数量，减小最后模型的大小，提高模型的泛化性。
# 具体此处：
# 调用 max_pool_2x2，使用 2x2 的 max_pooling，且步长取 1。
h_pool1 = max_pool_2x2(h_conv1)  # 20x20 => 10x10

# cnn第二层，和第一层的构建基本一样。
# 需要注意：
# 第二层中，w 的输入通道数为上一层最后的输出通道数，也就是 hpool1 的输出，是 32（第一层里面已经定义）。
# 如果不确定，可以通过这种方式来确定输入通道数：
# in = h_pool1.get_shape()[-1].value
# [-1] 表示不管 h_pool1 的形状如何，都取它最后一维(等同于通道数)的大小。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 10x10
h_pool2 = max_pool_2x2(h_conv2)  # 10x10 => 5x5

# 以下为全连接层：将“分布式特征表示”映射到样本标记空间：
# 全连接神经网络的第一个隐藏层
# 池化层输出的元素总数为：5(H)*5(W)*64(filters)
# 将 fc1 的 w（W_fc1） 形状设定为 [5 * 5 * 64, 1024]。
# 第一个维度大小为 5 * 5 * 64，
# 因为经过前面一层的卷积层的池化层后，输出的就是一个 5 * 5 * 64 的张量，
# 所以这里的输入就是上一层的输出。
# 第二个维度大小为 1024。
# 是自行设定的，表示该全链接层的神经元个数。
# 数量越多计算耗时越长，如果数量过少，对前面提取出来的特征的分类效果会不够好。
# 因此，全链接层拥有的特征数量就是 5 x 5 x 64 x 1024。
W_fc1 = weight_variable([5 * 5 * 64, 1024])  # 全连接第一个隐藏层神经元1024个
# b 的数量需要对应于 w（W_fc1）的最后一个维度，就是一个神经元对应一个偏置量b。
b_fc1 = bias_variable([1024])
# 为了能和上面定义的全链接层特征张量(W_fc1\b_fc1)相乘\相加，需要把输入的张量(h_pool2)变形/展平。
# 对于每一个 5 x 5 x64 的输入张量而言，它们展平成一个一维的向量。
# 第一个维度取 -1 同上面一样，最后确定这个维度。实际上就是最后一个池化层输出的数量。
h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])  # 转成-1列
# 一样，按照 wx + b 的线性公式构建函数，然后加上 ReLu 函数增加非线性变化。
# 本质：Affine（仿射变换）+relu
# 仿射变换（仿射变换包括一次线性变换和一次平移，分别对应神经网络的加权和运算与加偏置运算。）
# relu（增加非线性变化）
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # Affine+ReLU
# 全连接层的总结：
# 全链接层功能：当一个来自最后一层池化层的 [5 x 5 x 64] 的输出经过全链接层，成为一个 [1 x 1024] 的向量。
# 这一层整合所有前面的特征，进行后续的分类操作。
#
# 但是：全链接层增加了模型的复杂度，因为增加了很多神经元来扩充特征集，它有助于提升模型的准确率。
# 随着特征数量的爆炸式增加，训练速度必然会变慢。
# 而且如果全链接层设置的神经元数量过多，会出现过拟合的现象。所以，需要适当的设置。


# 在全链接层之后，往往会跟着 Dropout 操作。
# 这是因为在神经网络中，神经元的个数非常多，导致可能会产生过拟合，全链接层操作之后尤其。
# 所以需要让过拟合发生的概率减小一些。一般采用 Dropout 的方案。
# 其中第二参数是使用变量动态传入的数值，表示各个神经元有多大的概率失效（不参与计算）。
# 较少了运算量，也减少了过拟合的可能性
keep_prob = tf.placeholder(tf.float32)  # 定义Dropout的比例
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 执行dropout

# 全连接神经网络输出层
# 此时权重的形状：
# 第一个维度是上一层是从全链接层输出来的 1024 个神经元
# 第二个维度是分类的总类目数，总共有 CLASSIFICATION_COUNT 个类别。
W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT])  # 全连接输出为 CLASSIFICATION_COUNT 个
# 此时权重的形状：就是神经元的个数（CLASSIFICATION_COUNT 个）
b_fc2 = bias_variable([CLASSIFICATION_COUNT])
# 也是一样，按照 wx + b 的线性公式构建函数，
# 此时（第一个参数h_fc1_drop是x，第二参数W_fc2是w，第三个参数是b_fc2）
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
# 至此， CNN 的网络构建完成，但只是的前向部分。


learning_rate = 1e-4  # 学习率
max_epochs = 40  # 代数
batch_size = 50  # 批大小
check_step = 10  # 检查点步长

# 完成交叉损失熵的计算
# 参数：logits，是上述网络中全连接的最后一层的输出-y_conv（如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes）
# labels：实际的标签，大小同上。
# 1. 先是对最后一层的输出使用softmax成本函数 --softmax的公式
#   求取输出属于某一类的概率，对于单样本，输出是一个num_classes大小的向量（[Y1，Y2,Y3...]其中Y1，Y2，Y3...分别代表了是属于该类的概率）
# 2. 然后是softmax的输出向量[Y1，Y2,Y3...]和样本的实际标签做一个交叉熵
#   预测越准确，结果的值越小（别忘了公式中前面有负号）
#   最后求一个平均，得到loss
# 本质上：先对预测的输出值“logits”进行softmax操作，再对softmax的输出结果和标签值“y_”进行交叉熵操作
# 实际上，也可以分为两步：(如下代码，和 tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits... 等价)
# 1. 求取 softmax
#   logits = y_conv
#   y = tf.nn.softmax(logits)
# 2. 求取 cross_entropy
#   cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), axis=1))

# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# 等价于：
logits = y_conv  # 增加此次赋值，正确率反而会高些？ ==>>
y = tf.nn.softmax(logits=logits)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))

# 完成反向传播
# AdamOptimizer 是Adam（Adaptive Moment Estimation，自适应矩估计）优化算法，借助二次方梯度校正来寻找全局最优点的优化算法。
# 如果一个随机变量服从某个分布，其的一阶矩是样本平均值， 二阶矩是样本平方的平均值。
# Adam算法根据损失函数对每个参数的梯度的一阶矩估计和二阶矩估计动态调整针对于每个参数的学习速率。
# 本质：利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
# Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
# 比较全的参数：
# 参数：
#   learning_rate:张量或浮点值。学习率
#   beta1:一个浮点值或一个常量浮点张量。一阶矩估计的指数衰减率
#   beta2:一个浮点值或一个常量浮点张量。二阶矩估计的指数衰减率
#   epsilon:数值稳定性的一个小常数
#   use_locking:如果True，要使用lock进行更新操作
#   name:应用梯度时为了创建操作的可选名称。默认为“Adam”
# minimize 两个操作：(1)计算各个变量的梯度 (2)用梯度更新这些变量的值
# 参数：
#   loss:  需要优化的损失（就是前面的交叉损失熵）；
# 具体此处：
# 利用反向传播算法对权重和偏置项进行修正，在运行中不断修正学习率。
#   根据其损失量学习自适应，损失量大则学习率大，进行修正的角度越大，损失量小，修正的幅度也小，学习率就小，但是不会超过自己所设定的学习率。
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
# 以上完成反向BP算法，构建出一个 交叉熵 损失函数，对损失函数做梯度下降
#   SGD：学习率恒定不变
# 至此，完成整个 CNN 网络的构建
# 整体结果放在 train_step 中，只要调用 train_step，就相当于执行一次整体训练

# 构建评估模型
# 对比预测结果（y_conv）和标签（y_），返回bool向量
# 额外： tf.argmax 中 axis=1，返回每一行中的最大值的位置索引（数组）
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# 计算准确率，转化为浮点数，求平均（数组）
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 至此，完成整个评估模型的构建
# 评估结果放在 accuracy 中，只要调用 accuracy，就相当于执行一次模型评估


# 开始训练
# 训练集数据太多？Epoch、Batch、Iteration
# 理论上准确率没有全集训练高，能节省时间。
# Epoch（时期）：
#   当一个完整的数据集通过了神经网络一次(正向+反向)，这个过程称为一次epoch。
#   （也就是说，所有训练集的特征矩阵每一个样本在神经网络中都进行了一次正向传播和一次反向传播 ）
#   一个Epoch就是将训练集中的所有样本/数据，在模型上进行了一次完整训练，称为 一代。
# 一个Epoch的样本（也就是所有的训练样本）数量可能太过庞大（对于单机机而言），
# 此时，需要把它分成多个小块，也就是就是分成多个 Batch 来进行训练。
# Batch/Batch_Size：
#   Batch（批/一批样本）：将整个训练集分成若干个Batch。
#   Batch_Size（批大小）：每批样本的大小。
# Iteration（一次迭代）：
#   一个 Batch（Batch_Size个样本） 训练一次，就是一次Iteration
# 神经网络中完整的数据集一次epoch不够，需要将完整的数据集在同样的神经网络中多次epoch。几次？是经验
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("装载训练数据...")
    # 获取训练集的特征矩阵、标签向量
    train_data, train_labels = load_data(TRAIN_DIR)
    # 对训练集的特征矩阵进行正则化
    train_data = normalize_data(train_data)
    # 对训练集的标签向量执行独热编码
    train_labels = onehot_labels(train_labels)
    # 探查训练集
    print("装载%d条数据，每条数据%d个特征" % (train_data.shape))

    # 获取训练集的总样本数
    train_samples_count = len(train_data)
    train_indicies = np.arange(train_samples_count)
    # 获得打乱的索引序列
    np.random.shuffle(train_indicies)

    print("装载测试数据...")
    # 获取测试集的特征矩阵、标签向量
    test_data, test_labels = load_data(TEST_DIR)
    # 对测试集的特征矩阵进行同样（同训练集）的正则化
    test_data = normalize_data(test_data)
    # 对测试集的标签向量执行独热编码
    test_labels = onehot_labels(test_labels)
    # 探查测试集
    print("装载%d条数据，每条数据%d个特征" % (test_data.shape))

    # 天花板取整（np.ceil），获取迭代次数（此处，就是批次）
    iters = int(np.ceil(train_samples_count / batch_size))
    print("Training...")
    # 逐个 epoch 进行训练
    for epoch in range(1, max_epochs + 1):
        print("Epoch #", epoch)
        # 逐个批次进行迭代-训练
        for i in range(1, iters + 1):
            # 获取本批数据
            # 获取本批数据的起点位置
            start_idx = (i * batch_size) % train_samples_count
            # 获取本批数据的起点、终点范围 = 批次范围
            idx = train_indicies[start_idx: start_idx + batch_size]
            # 按本批次范围获取训练集的特征矩阵
            batch_x = train_data[idx, :]
            # 按本批次范围获取训练集的标签向量
            batch_y = train_labels[idx, :]
            # 开始训练：
            # train_step：调用网络完成一次整体训练
            # accuracy：调用评估模型完成一次模型评估
            # 同时传入本批次训练的 特征矩阵、标签向量、dropout的比率
            # list or tensor：传入 tensor变量
            # feed_dict：以字典的方式填充占位，即给出 tensor 常量
            _, batch_accuracy = sess.run([train_step, accuracy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
            # 判断检查点，输出中间结果
            if i % check_step == 0:
                print("Iter:", i, "of", iters, "batch_accuracy=", batch_accuracy)
    print("Training completed.")

    #保存模型
    print("Saving model...")
    saver = tf.train.Saver(max_to_keep=1)
    saved_file = saver.save(sess, MODEL_PATH,global_step=epoch)
    print('Model saved to ', saved_file)

    # 注意：测试时，不需要dropout！
    test_accuracy = accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})
    print('Test accuracy %g' % test_accuracy)  # 约0.97~0.98




