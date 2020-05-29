import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

x = tf.get_variable('x', dtype=tf.float32, shape=[], initializer=tf.constant_initializer(3.))
y = tf.square(x)    # y = x ^ 2
y_grad = tf.gradients(y, x)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
y_, y_grad_ = sess.run([y, y_grad])
print(y_, y_grad_)