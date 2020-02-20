import tensorflow as tf

# Declare a random float (scalar).
random_float = tf.random.uniform(shape=())

# Declare a zero vector with two elements.
zero_vector = tf.zeros(shape=(2))

# Declare two 2*2 constant matrices A and B.
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])

# View the shape, type and value of matrix A.
print(A.shape)      # Output (2, 2), which means the number of rows and cols are both 2.
print(A.dtype)      # Output <dtype: 'float32'>.
print(A.numpy())    # Output [[1. 2.]
                    #         [3. 4.]].

C = tf.add(A, B)    # Compute the elementwise sum of A and B.
D = tf.matmul(A, B) # Compute the multiplication of A and B.

print(C)
print(D)
