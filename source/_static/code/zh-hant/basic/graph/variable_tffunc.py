import tensorflow as tf

a = tf.Variable(0.0)

@tf.function
def plus_one_op():
    a.assign(a + 1.0)
    return a

for i in range(5):
    plus_one_op()
    print(a.numpy())