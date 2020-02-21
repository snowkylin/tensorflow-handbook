import tensorflow as tf

x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:     # All calculation steps will be recorded within the context of tf.GradientTape() for differentiation.
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # Compute the derivative of y with respect to x.
print([y, y_grad])

X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])        # Compute the partial derivative of L(w, b) with respect to w and b.
print([L.numpy(), w_grad.numpy(), b_grad.numpy()])