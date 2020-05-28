import tensorflow as tf

x = tf.constant(3.)                 # x 常數類型張量，預設無法對其求導
with tf.GradientTape() as tape:
    y = tf.square(x)
y_grad_1 = tape.gradient(y, x)      # 求導結果為 None
with tf.GradientTape() as tape:
    tape.watch(x)                   # 使用 tape.watch 手動將 x 加入監視列表
    y = tf.square(x)
y_grad_2 = tape.gradient(y, x)      # 求導結果為 tf.Tensor(6.0, shape=(), dtype=float32)
print(y_grad_1, y_grad_2)