import tensorflow as tf
import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)

num_epoch = 10000
learning_rate = 1e-3
for e in range(num_epoch):
    # 前向传播
    y_pred = a * X + b
    loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y)) # loss = 0.5 * np.sum(np.square(a * X + b - y))

    # 反向传播，手动计算变量（模型参数）的梯度
    grad_a = tf.reduce_sum((y_pred - y) * X)
    grad_b = tf.reduce_sum(y_pred - y)

    # 更新参数
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b

print(a, b)