import tensorflow as tf
import numpy as np
import scipy.sparse
import timeit

# A = tf.random.uniform(shape=(1000, 10000))
# b = tf.random.uniform(shape=(10000, 1))
# print(timeit.timeit('tf.matmul(A, b)', setup='import tensorflow as tf; from __main__ import A, b;', number=1))

# A_np = A.numpy()
# b_np = b.numpy()
# print(A_np.shape)
# print(timeit.timeit('np.dot(A_np, b_np)', setup='import numpy as np; from __main__ import A_np, b_np;', number=1))

# A = scipy.sparse.rand(100000, 100000, 0.001, dtype=np.float)
# b = np.random.rand(10000, 1)
m, n, density = 1000000, 100000, 0.01
nnz = int(m * n * density)
number = 10
row = np.random.randint(0, m, nnz)
col = np.random.randint(0, n, nnz)
data = np.random.rand(nnz)
A = scipy.sparse.csr_matrix((data, (row, col)), shape=(m, n))
b = np.random.rand(n, 1)
print(timeit.timeit('A.dot(b)', setup='import scipy.sparse; from __main__ import A, b;', number=number))
A = tf.SparseTensor(np.stack([row, col], axis=1), data, dense_shape=(m, n))
b = tf.constant(b)
print(timeit.timeit('tf.sparse.sparse_dense_matmul(A, b)', setup='import tensorflow as tf; from __main__ import A, b;', number=number))
