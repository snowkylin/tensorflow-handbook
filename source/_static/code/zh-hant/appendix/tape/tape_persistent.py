import tensorflow as tf

x_1 = tf.Variable(3.)
x_2 = tf.Variable(2.)
with tf.GradientTape(persistent=True) as tape:
    y_1 = tf.square(x_1)
    y_2 = tf.pow(x_2, 3)
y_grad_1 = tape.gradient(y_1, x_1)  # 6.0 = 2 * 3.0
y_grad_2 = tape.gradient(y_2, x_2)  # 12.0 = 3 * 2.0 ^ 2
del tape
print(y_grad_1, y_grad_2)