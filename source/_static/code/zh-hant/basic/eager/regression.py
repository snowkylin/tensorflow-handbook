import tensorflow as tf

X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])

w = tf.Variable(initial_value=tf.zeros([3., 1.]))
b = tf.Variable(initial_value=[0.])
variables = [w, b]

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = tf.matmul(X, w) + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
print(variables)
