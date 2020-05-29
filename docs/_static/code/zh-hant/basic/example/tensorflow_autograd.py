import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017])
y_raw = np.array([12000, 14000, 15000, 16500, 17500])

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

import tensorflow as tf

learning_rate_ = tf.placeholder(dtype=tf.float32)
X_ = tf.placeholder(dtype=tf.float32, shape=[5])
y_ = tf.placeholder(dtype=tf.float32, shape=[5])
a = tf.get_variable('a', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)

y_pred = a * X_ + b
loss = tf.constant(0.5) * tf.reduce_sum(tf.square(y_pred - y_))

# 反向传播，利用TensorFlow的梯度下降优化器自动计算并更新变量（模型参数）的梯度
train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_).minimize(loss)

num_epoch = 10000
learning_rate = 1e-3
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for e in range(num_epoch):
        sess.run(train_op, feed_dict={X_: X, y_: y, learning_rate_: learning_rate})
    print(sess.run([a, b]))