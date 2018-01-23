import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

X = tf.constant(X)
y = tf.constant(y)

a = tfe.Variable(0., name='a')
b = tfe.Variable(0., name='b')

def model(x):
    return a * x + b

def loss(X_, y_):
    return 0.5 * tf.reduce_sum(tf.square(model(X_) - y_))

grad_fn = tfe.implicit_gradients(loss)
num_epoch = 10000
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
for e in range(num_epoch):
    # 前向传播
    y_pred = a * X + b

    # 反向传播，利用Eager模式下的tfe.implicit_gradients()自动计算梯度
    grad = grad_fn(X, y)
    optimizer.apply_gradients(grad)

    # 更新参数
    # a, b = a - learning_rate * grad_a, b - learning_rate * grad_b

print(a, b)