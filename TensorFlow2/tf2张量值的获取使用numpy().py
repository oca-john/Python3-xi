import tensorflow as tf

# 定义两个2×2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])

print(A.shape)      # 输出(2, 2)，即矩阵的长和宽均为2
print(A.dtype)      # 输出<dtype: 'float32'>
print(A.numpy())    # 输出[[1. 2.]

print(A)    # 直接输出A则展示包括张量A的值、形状、数据类型等所有张量信息
            # 仅获得值，使用.numpy()，使张量向量化（去除其他信息）
            # 仅获得形状，使用.shape属性
            # 仅获得类型，使用.dtype属性
