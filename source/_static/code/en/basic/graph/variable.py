import tensorflow as tf

a = tf.get_variable(name='a', shape=[])
initializer = tf.assign(a, 0)   # tf.assign(x, y) will return a operation “assign Tensor y's value to Tensor x”
a_plus_1 = a + 1    # Equal to a + tf.constant(1)
plus_one_op = tf.assign(a, a_plus_1)

sess = tf.Session()
sess.run(initializer)
for i in range(5):
    sess.run(plus_one_op)                   # Do plus one operation to a
    a_ = sess.run(a)                        # Calculate a‘s value and put the result to a_
    print(a_)
