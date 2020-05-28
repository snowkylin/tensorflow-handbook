import tensorflow as tf
import time

A = tf.random.uniform(shape=(1000, 10000))
B = tf.random.uniform(shape=(10000, 10000))
start_time = time.time()
C = tf.matmul(A, B)
print('time consumed by tf.matmul:', time.time() - start_time)