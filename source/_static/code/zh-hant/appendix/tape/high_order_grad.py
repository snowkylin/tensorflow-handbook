import tensorflow as tf

x = tf.Variable(3.)
with tf.GradientTape() as tape_1:
    with tf.GradientTape() as tape_2:
        y = tf.square(x)
    dy_dx = tape_2.gradient(y, x)   # 值為 6.0
d2y_dx2 = tape_1.gradient(dy_dx, x) # 值為 2.0