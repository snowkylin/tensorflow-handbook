import tensorflow as tf

a = tf.Variable(0.0)

@tf.function
def g():
    a.assign(a + 1.0)
    return a

print(g())
print(g())
print(g())