import tensorflow as tf
import time

A = tf.random.uniform(shape=(1000, 500, 1000))
B = tf.random.uniform(shape=(1000, 1000))

start_time = time.time()
C = tf.einsum('ijk,kl->ijl', A, B)
print('time consumed by tf.einsum:', time.time() - start_time)  # 0.280s

start_time = time.time()
C = []
for i in range(1000):
    C.append(tf.matmul(A[i], B))
C = tf.stack(C, axis=0)
print('time consumed by for loop:', time.time() - start_time)   # 0.401s


