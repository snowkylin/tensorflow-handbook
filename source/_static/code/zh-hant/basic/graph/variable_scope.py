import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_eager_execution()

def dense(inputs, num_units):
    weight = tf.get_variable(name='weight', shape=[inputs.shape[1], num_units])
    bias = tf.get_variable(name='bias', shape=[num_units])
    return tf.nn.relu(tf.matmul(inputs, weight) + bias)

def model(inputs):
    with tf.variable_scope('dense1'):   # 限定变量的作用域为 dense1
        x = dense(inputs, 10)           # 声明了 dense1/weight 和 dense1/bias 两个变量
    with tf.variable_scope('dense2'):   # 限定变量的作用域为 dense2
        x = dense(x, 10)                # 声明了 dense2/weight 和 dense2/bias 两个变量
    with tf.variable_scope('dense2', reuse=True):   # 第三层复用第二层的变量
        x = dense(x, 10)
    return x

inputs = tf.placeholder(shape=[10, 32], dtype=tf.float32)
outputs = model(inputs)
print(tf.global_variables())    # 输出当前计算图中的所有变量节点
sess = tf.Session()
sess.run(tf.global_variables_initializer())
outputs_ = sess.run(outputs, feed_dict={inputs: np.random.rand(10, 32)})
print(outputs_)