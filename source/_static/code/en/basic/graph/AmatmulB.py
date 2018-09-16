import tensorflow as tf

A = tf.ones(shape=[2, 3])   # tf.ones(shape) defines a all one matrix with shape
B = tf.ones(shape=[3, 2])
C = tf.matmul(A, B)

sess = tf.Session()
C_ = sess.run(C)
print(C_)
