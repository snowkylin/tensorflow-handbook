import tensorflow.compat.v1 as tf
import time
from zh.model.utils import MNISTLoader
from zh.model.mnist.cnn import CNN

num_batches = 400
batch_size = 50
learning_rate = 0.001
data_loader = MNISTLoader()


    

X_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
y_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])
train_op = model(X_placeholder, y_placeholder)
start_time = time.time()
sess = tf.Session()
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    sess.run(train_op, feed_dict={X_placeholder: X, y_placeholder: y})
end_time = time.time()
print(end_time - start_time)      