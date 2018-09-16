import tensorflow as tf

a = tf.placeholder(dtype=tf.int32)  # Define a placeholder Tensor
b = tf.placeholder(dtype=tf.int32)
c = a + b

a_ = input("a = ")  # Read an Integer from terminal and put it into a_
b_ = input("b = ")

sess = tf.Session()
c_ = sess.run(c, feed_dict={a: a_, b: b_})  # feed_dict will input Tensors' value needed by computing c
print("a + b = %d" % c_)
