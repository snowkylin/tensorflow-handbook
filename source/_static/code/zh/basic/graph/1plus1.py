import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# 以下三行定义了一个简单的“计算图”
a = tf.constant(1)  # 定义一个常量张量（Tensor）
b = tf.constant(1)
c = a + b           # 等价于 c = tf.add(a, b)，c是张量a和张量b通过 tf.add 这一操作（Operation）所形成的新张量
# 到此为止，计算图定义完毕，然而程序还没有进行任何实质计算。
# 如果此时直接输出张量 c 的值，是无法获得 c = 2 的结果的

sess = tf.Session()     # 实例化一个会话（Session）
c_ = sess.run(c)        # 通过会话的 run() 方法对计算图里的节点（张量）进行实际的计算
print(c_)