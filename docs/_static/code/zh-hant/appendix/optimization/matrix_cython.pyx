import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def matmul(A, B):
    C = np.zeros(shape=(10000, 10000))
    cdef int i, j, k
    cdef double[:, :] A_view = A
    cdef double[:, :] B_view = B
    cdef double[:, :] C_view = C
    for i in range(10000):
        for j in range(10000):
            for k in range(10000):
                C_view[i, j] += A_view[i, k] * B_view[k, j]
    return C