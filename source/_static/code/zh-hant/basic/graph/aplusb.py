import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

a = tf.placeholder(dtype=tf.int32)  # 定义一个占位符Tensor
b = tf.placeholder(dtype=tf.int32)
c = a + b

a_ = int(input("a = "))  # 从终端读入一个整数并放入变量a_
b_ = int(input("b = "))

sess = tf.Session()
c_ = sess.run(c, feed_dict={a: a_, b: b_})  # feed_dict参数传入为了计算c所需要的张量的值
print("a + b = %d" % c_)