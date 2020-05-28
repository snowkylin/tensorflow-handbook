import tensorflow as tf

@tf.function
def graph(a, b):
    c = a + b
    return c

a_ = int(input("a = "))
b_ = int(input("b = "))
c_ = graph(a_, b_)
print("a + b = %d" % c_)