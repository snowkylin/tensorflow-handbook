import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

a = tf.placeholder(dtype=tf.int32)  # 定義一個字符串Tensor
b = tf.placeholder(dtype=tf.int32)
c = a + b

a_ = int(input("a = "))  # 從使用者讀入一個整數並放入變數a_
b_ = int(input("b = "))

sess = tf.Session()
c_ = sess.run(c, feed_dict={a: a_, b: b_})  # feed_dict參數傳入為了計算c所需要的變數的值
print("a + b = %d" % c_)