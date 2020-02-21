import tensorflow as tf

# 以下被 @tf.function 修饰的函数定义了一个计算图
@tf.function
def graph():
    a = tf.constant(1)
    b = tf.constant(1)
    c = a + b
    return c
# 到此为止，计算图定义完毕。由于 graph() 是一个函数，在其被调用之前，程序是不会进行任何实质计算的。
# 只有调用函数，才能通过函数返回值，获得 c = 2 的结果

c_ = graph()
print(c_.numpy())