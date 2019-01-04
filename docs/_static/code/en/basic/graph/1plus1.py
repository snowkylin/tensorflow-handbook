import tensorflow as tf

# Defince a "Computation Graph"
a = tf.constant(1)  # Defince a constant Tensor
b = tf.constant(1)
c = a + b  # Equal to c = tf.add(a, b)，c is a new Tensor created by Tensor a and Tesor b's add Operation

sess = tf.Session()     # Initailize a Session
c_ = sess.run(c)        # Session的run() will do actually computation to the nodes (Tensor) in the Computation Graph
print(c_)
