import tensorflow as tf

zero = tf.fill([2, 3], tf.constant(0))  # == tf.zeros([2, 3])
tf.get_default_graph().collections
a = tf.Variable(initial_value=zero)     # name="Variable:0"
print(tf.GraphKeys.GLOBAL_VARIABLES)    # "variables"
a_initializer = tf.assign(a, a._initial_value)
a_plus_1 = tf.add(a, 1)
tf.Graph
# b = tf.get_variable()

with tf.Session() as sess:
    sess.run(a_initializer)
    print(sess.run(a_plus_1))