import tensorflow as tf

# 定义一个“计算图”
a = tf.constant(1)  # 定义一个常量Tensor（张量）
b = tf.constant(1)
c = a + b  # 等价于 c = tf.add(a, b)，c是张量a和张量b通过Add这一Operation（操作）所形成的新张量

sess = tf.Session()     # 实例化一个Session（会话）
c_ = sess.run(c)        # 通过Session的run()方法对计算图里的节点（张量）进行实际的计算
print(c_)