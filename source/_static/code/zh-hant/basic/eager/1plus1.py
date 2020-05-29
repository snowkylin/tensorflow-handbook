import tensorflow as tf

# 定義一個隨機數（純量）
random_float = tf.random.uniform(shape=())

# 定義一個有2個元素的零向量
zero_vector = tf.zeros(shape=(2))

# 定義兩個2×2的常量矩陣
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])

# 查看矩陣A的形狀、類型和值
print(A.shape)      # 輸出(2, 2)，即矩陣的長和寬均為2
print(A.dtype)      # 輸出<dtype: 'float32'>
print(A.numpy())    # 輸出[[1. 2.]
                    #      [3. 4.]]

C = tf.add(A, B)    # 計算矩陣A和B的和
D = tf.matmul(A, B) # 計算矩陣A和B的乘積

print(C)
print(D)
