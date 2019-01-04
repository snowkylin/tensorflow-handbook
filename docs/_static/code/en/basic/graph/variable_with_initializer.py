import tensorflow as tf

a = tf.get_variable(name='a', shape=[], initializer=tf.zeros_initializer)   # Made initializer as a all zero initializer
a_plus_1 = a + 1
plus_one_op = tf.assign(a, a_plus_1)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # Initailize all the 
for i in range(5):
    sess.run(plus_one_op)
    a_ = sess.run(a)
    print(a_)
