import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017])
y_raw = np.array([12000, 14000, 15000, 16500, 17500])

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

import tensorflow as tf

# Define data flow gragh
learning_rate_ = tf.placeholder(dtype=tf.float32)
X_ = tf.placeholder(dtype=tf.float32, shape=[5])
y_ = tf.placeholder(dtype=tf.float32, shape=[5])
a = tf.get_variable('a', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)

y_pred = a * X_ + b
loss = tf.constant(0.5) * tf.reduce_sum(tf.square(y_pred - y_))

# Back propagation, calculate gradient of variables(model parameters) manually
grad_a = tf.reduce_sum((y_pred - y_) * X_)
grad_b = tf.reduce_sum(y_pred - y_)

# Gradient descent, update parameters manually
new_a = a - learning_rate_ * grad_a
new_b = b - learning_rate_ * grad_b
update_a = tf.assign(a, new_a)
update_b = tf.assign(b, new_b)

train_op = [update_a, update_b] 
# End of defining of data flow gragh
# Attention, until now, we haven't do any actually data calculation, just defined a data flow gragh

num_epoch = 10000
learning_rate = 1e-3
with tf.Session() as sess:
    # Initialize variables a and b
    tf.global_variables_initializer().run()
    # Put data in the data flow gragh created above to calculate and update variables
    for e in range(num_epoch):
        sess.run(train_op, feed_dict={X_: X, y_: y, learning_rate_: learning_rate})
    print(sess.run([a, b]))
