import tensorflow as tf
import numpy as np

eager = True
X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
y = np.array([[10.0], [20.0]], dtype=np.float32)


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):     # 这里 input_shape 是第一次运行call()时参数inputs的形状
        self.w = self.add_variable(name='w',
            shape=[input_shape[-1], self.units], initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b',
            shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred


class LinearModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = LinearLayer(units=1)

    def call(self, inputs):
        output = self.layer(inputs)
        return output


class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))


if eager:
    X = tf.constant(X)
    y = tf.constant(y)
    model = LinearModel()
    mse = MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = mse(y, y_pred)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    print(model.variables)
else:
    model = LinearModel()
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