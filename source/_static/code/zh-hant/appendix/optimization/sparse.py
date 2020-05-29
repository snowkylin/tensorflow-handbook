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