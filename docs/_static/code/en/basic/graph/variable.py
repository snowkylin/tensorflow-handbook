import tensorflow as tf

a = tf.get_variable(name='a', shape=[])
initializer = tf.assign(a, 0)   # tf.assign(x, y)返回一个“将张量y的值赋给变量x”的操作
a_plus_1 = a + 1    # 等价于 a + tf.constant(1)
plus_one_op = tf.assign(a, a_plus_1)

sess = tf.Session()
sess.run(initializer)
for i in range(5):
    sess.run(plus_one_op)                   # 对变量a执行加一操作
    a_ = sess.run(a)                        # 获得变量a的值并存入a_
    print(a_)
