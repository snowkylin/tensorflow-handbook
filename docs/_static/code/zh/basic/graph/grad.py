import tensorflow as tf

x = tf.Variable(initial_value=1.)
y = tf.square(x)    # y = x ^ 2
y_grad = tf.gradients(y, x)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
y_, y_grad_ = sess.run([y, y_grad])
print([y_, y_grad_])