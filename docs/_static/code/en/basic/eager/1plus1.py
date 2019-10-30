import tensorflow as tf

# declare a random float (scalar)
random_float = tf.random.uniform(shape=())

# declare a zero vector with two elements
zero_vector = tf.zeros(shape=(2))

# declare 2x2 constant matrices A and B
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])

# view the shape, type and value of matrix A
print(A.shape)      # output (2, 2), which means the number of rows and cols are both 2
print(A.dtype)      # output <dtype: 'float32'>
print(A.numpy())    # output [[1. 2.]
                    #         [3. 4.]]

C = tf.add(A, B)    # calculate the element-wise sum of A and B 
D = tf.matmul(A, B) # calculate the multiplication of A and B 

print(C)
print(D)
