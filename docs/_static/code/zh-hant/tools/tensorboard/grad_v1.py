import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

log_dir = 'tensorboard_v1'

x = tf.get_variable('x', dtype=tf.float32, shape=[], initializer=tf.constant_initializer(3.))
y = tf.square(x)
y_grad = tf.gradients(y, x)

print(x, y, y_grad)

sess = tf.Session()
writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)

