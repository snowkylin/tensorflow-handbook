import tensorflow as tf

x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文內，所有計算步驟都會被記錄以用於推導
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 計算y關於x的導數
print(y, y_grad)

X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])        # 計算L(w, b)關於w, b的偏導數
print(L, w_grad, b_grad)