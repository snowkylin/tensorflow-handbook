import tensorflow as tf
import numpy as np

eager = True
X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
y = np.array([[10.0], [20.0]], dtype=np.float32)


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(LinearLayer, self).__init__()

    def build(self, input_shape):     # here input_shape is a TensorShape
        self.w = self.add_variable(name='w',
            shape=[input_shape[-1], 1], initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b',
            shape=[1], initializer=tf.zeros_initializer())

    def call(self, X):
        y_pred = tf.matmul(X, self.w) + self.b
        return y_pred


class Linear(tf.keras.Model):
    def __init__(self):
        super(Linear, self).__init__()
        self.layer = LinearLayer()

    def call(self, input):
        output = self.layer(input)
        return output


if eager:
    tf.enable_eager_execution()
    X = tf.constant(X)
    y = tf.constant(y)
    model = Linear()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_mean(tf.square(y_pred - y))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    print(model.variables)
else:
    model = Linear()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    X_placeholder = tf.placeholder(name='X', shape=[None, 3], dtype=tf.float32)
    y_placeholder = tf.placeholder(name='y', shape=[None, 1], dtype=tf.float32)
    y_pred = model(X_placeholder)
    loss = tf.reduce_mean(tf.square(y_pred - y_placeholder))
    train_op = optimizer.minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            sess.run(train_op, feed_dict={X_placeholder: X, y_placeholder: y})
        print(sess.run(model.variables))