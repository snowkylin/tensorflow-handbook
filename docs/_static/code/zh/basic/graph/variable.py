import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

a = tf.get_variable(name='a', shape=[])
initializer = tf.assign(a, 0.0)   # tf.assign(x, y)返回一个“将张量y的值赋给变量x”的操作
plus_one_op = tf.assign(a, a + 1.0)

sess = tf.Session()
sess.run(initializer)
for i in range(5):
    sess.run(plus_one_op)       # 对变量a执行加一操作
    print(sess.run(a))          # 输出此时变量a在当前会话的计算图中的值
