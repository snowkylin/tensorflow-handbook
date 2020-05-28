import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

A = tf.ones(shape=[2, 3])   # tf.ones(shape)定義了一个形狀為shape的全1矩陣
B = tf.ones(shape=[3, 2])
C = tf.matmul(A, B)

sess = tf.Session()
C_ = sess.run(C)
print(C_)