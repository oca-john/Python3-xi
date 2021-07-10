# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf 
from tensorflow import keras
# 导包部分不是先后依赖的，第一步导入TensorFlow作为tf了，但第二部导包不能从tf中导入，因为python仍会从tensorflow文件夹（模块）中查找子模块

# %%
# 回归问题-散点图线性回归

import numpy as np 
np.random.seed(1337)        # 为重现性，设置随机种子
from keras.models import Sequential # 按顺序方式搭建模型
from keras.models import Input      # 最新版本需要用单独的Input模块
from keras.layers import Dense      # 全连接层
import matplotlib.pyplot as plt 

# 构造数据
X = np.linspace(-1,1,200)
np.random.shuffle(X)        # 随机化处理数据
Y = 0.5*X + 2 + np.random.normal(0,0.05,(200,)) # 内部的元组？？
# plt.scatter(X,Y)
# 数据集分割
X_train, Y_train = X[:160], Y[:160] # 0-160个数据
X_test, Y_test = X[160:], Y[160:]   # 160-200个数据

# 模型搭建
# 定义神经网络每一层            ####### 和莫烦教程有较大变化
model = Sequential()
model.add(Input(shape=(1,)))        # 最新的Keras不是单独用Dense定义层，有单独的Input
                                    # Dense定义也发生变化，需要定义输出unit
model.add(Dense(1, activation=None))   # 定义输出维数unit即可，激活函数可不选
# 选择损失函数和优化算法
model.compile(loss='mse', optimizer='sgd')

# 训练过程
print('Training ...')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)   # 用train_on_batch模块来训练数据
    if step % 100 == 0:
        print('train cost:', cost)                  # 每100次迭代输出一次信息

# 测试
print('\nTesting ...')
cost = model.evaluate(X_test, Y_test, batch_size=40)# evaluate模块用测试数据评估
print('test cost:', cost)
W,b = model.layers[0].get_weights()                 # 输出模型参数
print('weight=',W,'\nbias=',b)

# 可视化展示训练结果
Y_pred = model.predict(X_test)  # 计算预测值
plt.scatter(X_test,Y_test)      # 真实值
plt.plot(X_test,Y_pred)         # 预测值


# %%
# 分类问题-MNIST手写数字数据集

import numpy as np 
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import Input          # 最新版本需要用单独的Input模块
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop    # 优化器

# MNIST数据集下载和数据集分割
# X shape(60000, 28*28), y shape(10000)
(X_train, y_train),(X_test, y_test) = mnist.load_data()
# 数据预处理-标准化、分类标签
X_train = X_train.reshape(X_train.shape[0],-1)/255  # 像素/255颜色归一化理解
                                # -1的解释，此函数来自numpy，具有ndarray维度推理的能力
                                # 此处参数应是正数，-1是错误的，意为根据另一维度来推导此维度
X_test = X_test.reshape(X_test.shape[0],-1)/255
y_train = np_utils.to_categorical(y_train, num_classes=10)  # 分类标签，处理为'one-hot'
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 模型搭建
# 全连接层
model = Sequential([    # 直接在Sequential内部搭建，用[]作为数组传入
    Input(784),         # 输入维度时28*28=784（像素数）
    Dense(32),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])
# 优化器/优化算法
rmsprop = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# 嵌入优化器
model.compile(
    optimizer=rmsprop,      # 优化器是上面定义的rmsprop
    loss='categorical_crossentropy',    # 损失函数是分类交叉熵
    metrics=['accuracy'],   # 优化过程中同时计算准确率信息
)

# 训练过程
print('Training ...')
model.fit(X_train, y_train, batch_size=32, epochs=2)    # epochs和老版本不同

# 测试
print('Testing ...')
loss,accuracy = model.evaluate(X_test, y_test)
print('test loss:',loss)
print('test accuracy:',accuracy)


# %%


# %% [markdown]
# ## 神经网络模型搭建实例

# %%
# CNN 卷积神经网络

import numpy as np 
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import Input          # 最新版本需要用单独的Input模块
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam       # 优化器

# MNIST数据集下载
(X_train, y_train),(X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1,1,28,28)
X_test = X_test.reshape(-1,1,28,28)
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 模型搭建
# 卷积层1
model = Sequential()
model.add(Convolution2D(
    filters=32,         # 过滤器，输出的大小
    kernel_size=5,      # 卷积核大小
    strides=1,
    padding='same',     # 结果大小相同，即卷积前先边缘填充一圈0
    input_shape=(1,
                28,28),
))
# 激活函数
model.add(Activation('relu'))
# 最大池化层
model.add(MaxPooling2D(
    pool_size=(2,2),
    strides=(2,2),
    padding='same',
))
# 卷积层2
model.add(Convolution2D(64,5,1,padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2,2),
    padding='same',
))
# 全连接层1
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
# 全连接层2
model.add(Dense(10))
model.add(Activation('softmax'))

# 优化器/优化算法
adam = Adam(learning_rate=1e-4)
# 嵌入优化器
model.compile(
    optimizer=adam,      # 优化器是上面定义的rmsprop
    loss='categorical_crossentropy',    # 损失函数是分类交叉熵
    metrics=['accuracy'],   # 优化过程中同时计算准确率信息
)

# 训练过程
print('\nTraining ...')
model.fit(X_train, y_train, batch_size=32, epochs=1)    # epochs和老版本不同

# 测试
print('\nTesting ...')
loss,accuracy = model.evaluate(X_test, y_test)
print('\ntest loss:',loss)
print('\ntest accuracy:',accuracy)


# %%
# 更紧凑的神经网络搭建方式

from keras.models import Sequential
# 模型搭建
model = Input(shape=(28,28,3))  # 定义输入层和维度

func1 = Convolution2D(filters=32, kernel_size=(1,1), strides=1, padding='same', activation='relu',)     # 过滤器，卷积核，移动步长，边缘填充，激活函数
func2 = MaxPooling2D(pool_size=(2,2), strides=1, padding='same',)
output = keras.layers.concatenate([func1, func2], axis=1)


# %%
# 模型可视化

from keras.utils import plot_model
plot_model(model, to_file='model.png')
# plot_model接收两个参数：
    # show_shapes:      是否显示输出数据的形状，默认False
    # show_layer_names: 是否显示层名称，默认True

# 获得pydot.Graph对象，在ipython中展示：
# pip install pydot-ng & brew install graphviz
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot',format='svg'))


# %%


# %% [markdown]
# ## Keras 官方英文文档
# %% [markdown]
# ### Keras 开发者文档 https://keras.io/guides/
# 
# 

# %%
# Keras 研究人员基础
# https://keras.io/getting_started/intro_to_keras_for_researchers/

# 导包
import tensorflow as tf 
from tensorflow import keras    # 作为tf的子包使用

# Tensor 张量
x = tf.constant([[5,2],[1,3]])  # 常量 张量
# print(x)            # 打印张量
# print(x.numpy())    # 打印张量的值（转为numpy对象了）
x = tf.ones(shape=(2,1))        # 全一 张量
# print(x)
x = tf.zeros(shape=(2,1))       # 全零 张量
# print(x)
x = tf.random.normal(shape=(2,2), mean=0.0, stddev=1.0)    # 创建正态分布随机常量张量，指定均值和标准差

# Variables 变量（TF）
int_value = tf.random.normal(shape=(2,2))
x = tf.Variable(int_value)      # 将数组“变量”化 - 转为TF专用变量
# print(x)

# 增加和减少变量值
new_value = tf.random.normal(shape=(2,2))   # 设定更新的变量值
x.assign(new_value)                         # 更新x的值
for i in range(2):                          # 两个嵌套for循环逐个更新两个维度的数据
    for j in range(2):
        assert x[i,j] == new_value[i,j]
# print(x)
added_value = tf.random.normal(shape=(2,2)) # 设定增量
x.assign_add(added_value)                     # 增加的x的值
for i in range(2):                          # for循环逐个更新
    for j in range(2):
        assert x[i,j] == new_value[i,j] + added_value[i,j]
# print(x)
# 减少用x.assign_sub()方法

# 数学计算
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))
c = a + b           # 加法，直接计算
d = tf.square(c)    # 平方
e = tf.exp(d)       # 指数
# print(c, d, e)

# 自动计算并监视梯度变化
a = tf.Variable(a)
with tf.GradientTape() as tape:     # Tape 译为磁带/录音，表示记录梯度
    c = tf.sqrt(tf.square(a) + tf.square(b))    # 对a,b进行操作，同时记录梯度（不用手动.watch）
    dc_da = tape.gradient(c, a)                 # 提取梯度信息到变量中
    # print(dc_da)

# Keras 定义 Layer
# class Linear(keras.layers.Layer):   # 用 keras.layer.Layer 定义一个线性方程
#     """y = w.x + b"""
#     def __init__(self, units=32, input_dim=32): # 初始化参数
#         super(Linear, self).__init__()          # 超类
#         w_init = tf.random_normal_initializer() # 随机初始化 w
#         self.w = tf.Variable(                   # w 方法，生成输入的同维 w 矩阵，可训练（参数）
#             initial_value=w_init(shape=(input_dim, units), dtype="float32"),
#             trainable=True,
#         )
#         b_init = tf.zeros_initializer()         # 初始化 b
#         self.b = tf.Variable(
#             initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
#         )
#     def call(self, inputs):     # 定义线性方程
#         return tf.matmul(inputs, self.w) + self.b
# 调用线性方程
# linear_layer = Linear(units=4, input_dim=2)     # 初始化线性方程对象（定义的是类）
# y = linear_layer(tf.ones((2, 2)))   # 用全一矩阵调用方程
# assert y.shape == (2, 4)            # assert 用于断言，此例未指定输出，输出为布尔值

# 基于上例，创建该层参数的权重
# class Linear(keras.layers.Layer):
#     """y = w.x + b"""
#     def __init__(self, units=32):
#         super(Linear, self).__init__()
#         self.units = units
#     def build(self, input_shape):           # 将w,b参数包进build方法中
#         self.w = self.add_weight(           # .add_weight 为 w 参数增加权重
#             shape=(input_shape[-1], self.units),
#             initializer="random_normal",
#             trainable=True,
#         )
#         self.b = self.add_weight(           # 同样计算 b 的权重
#             shape=(self.units,), initializer="random_normal", trainable=True
#         )
#     def call(self, inputs):
#         return tf.matmul(inputs, self.w) + self.b
# linear_layer = Linear(4)                    # 实例化为对象
# y = linear_layer(tf.ones((2, 2)))           # 调用线性方程时会自动调用该方法计算权重

# 层的自动求梯度

# 包含层的层

# 跟踪所有层的损失

# 跟踪训练指标

# 编译函数

# 训练模型和推理模式

# 功能性API快速搭建模型
inputs = tf.keras.Input(shape=(16,), dtype="float32")
x = Linear(32)(inputs)      # 重用本页之前定义的线性方程
x = Dropout(0.5)(x)         # 重用本页之前定义的
outputs = Linear(10)(x)
model = tf.keras.Model(inputs, outputs)
assert len(model.weights) == 4
y = model(tf.ones((2, 16)))
assert y.shape == (2, 10)
y = model(tf.ones((2, 16)), training=True)

# -------------------------------------------------
# 自动编码器 - 处理MNIST手写数字识别 - Keras API版本
from tensorflow.keras import layers

original_dim = 784
intermediate_dim = 64
latent_dim = 32

# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
x = layers.Dense(intermediate_dim, activation="relu")(original_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
outputs = layers.Dense(original_dim, activation="sigmoid")(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

# Define VAE model.
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")

# Add KL divergence regularization loss.
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

# Loss and optimizer.
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Prepare a dataset.
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    x_train.reshape(60000, 784).astype("float32") / 255
)
dataset = dataset.map(lambda x: (x, x))  # Use x_train as both inputs & targets
dataset = dataset.shuffle(buffer_size=1024).batch(32)

# Configure the model for training.
vae.compile(optimizer, loss=loss_fn)

# Actually training the model.
vae.fit(dataset, epochs=1)

# %% [markdown]
# ### Keras 应用程序接口 https://keras.io/api/
# 
# 
# %% [markdown]
# ### Keras 代码示例 https://keras.io/examples/

# %%



