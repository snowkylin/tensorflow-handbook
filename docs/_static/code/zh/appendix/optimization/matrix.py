import tensorflow as tf
import numpy as np
import time
import pyximport; pyximport.install()
import matrix_cython

A = np.random.uniform(size=(10000, 10000))
B = np.random.uniform(size=(10000, 10000))

start_time = time.time()
C = np.zeros(shape=(10000, 10000))
for i in range(10000):
    for j in range(10000):
        for k in range(10000):
            C[i, j] += A[i, k] * B[k, j]
print('time consumed by Python for loop:', time.time() - start_time)    # ~700000s

start_time = time.time()
C = matrix_cython.matmul(A, B)      # Cython 代码为上述 Python 代码的 C 语言版本，此处省略
print('time consumed by Cython for loop:', time.time() - start_time)    # ~8400s

start_time = time.time()
C = np.dot(A, B)
print('time consumed by np.dot:', time.time() - start_time)     # 5.61s

A = tf.constant(A)
B = tf.constant(B)
start_time = time.time()
C = tf.matmul(A, B)
print('time consumed by tf.matmul:', time.time() - start_time)  # 0.77s