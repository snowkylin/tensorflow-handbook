import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

a = tf.get_variable(name='a', shape=[], 
    initializer=tf.zeros_initializer)   # 指定初始化器为全0初始化
plus_one_op = tf.assign(a, a + 1.0)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 初始化所有变量
for i in range(5):
    sess.run(plus_one_op)
    print(sess.run(a))
