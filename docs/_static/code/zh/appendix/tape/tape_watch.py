import tensorflow as tf

x = tf.constant(3.)                 # x 为常量类型张量，默认无法对其求导
with tf.GradientTape() as tape:
    y = tf.square(x)
y_grad_1 = tape.gradient(y, x)      # 求导结果为 None
with tf.GradientTape() as tape:
    tape.watch(x)                   # 使用 tape.watch 手动将 x 加入监视列表
    y = tf.square(x)
y_grad_2 = tape.gradient(y, x)      # 求导结果为 tf.Tensor(6.0, shape=(), dtype=float32)
print(y_grad_1, y_grad_2)