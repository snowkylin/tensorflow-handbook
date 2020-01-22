import tensorflow as tf

@tf.function
def graph():
    a = tf.constant(1)  # 定义一个常量张量（Tensor）
    b = tf.constant(1)
    c = a + b
    return c

c_ = graph()
print(c_.numpy())