import tensorflow as tf
import numpy as np
from zh.model.mlp.mlp import MLP
from zh.model.cnn.cnn import CNN

tf.enable_eager_execution()
model_type = 'CNN'
num_batches = 1000
batch_size = 50
learning_rate = 0.001


class DataLoader():
    def __init__(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.train_data = mnist.train.images                                 # np.array [55000, 784]
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)   # np.array [55000] of int32
        self.eval_data = mnist.test.images                                   # np.array [10000, 784]
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)     # np.array [10000] of int32

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_labels[index]


if model_type == 'MLP':
    model = MLP()
else:
    model = CNN()
data_loader = DataLoader()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_logit_pred = model(tf.convert_to_tensor(X))
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

num_eval_samples = np.shape(data_loader.eval_labels)[0]
y_pred = model.predict(data_loader.eval_data).numpy()
print("test accuracy: %f" % (sum(y_pred == data_loader.eval_labels) / num_eval_samples))

num_correct_pred = 0
for batch_index in range(num_eval_samples // batch_size):
    y_pred = model.predict(data_loader.eval_data[batch_index * batch_size: (batch_index + 1) * batch_size]).numpy()
    num_correct_pred += sum(y_pred == data_loader.eval_labels[batch_index * batch_size: (batch_index + 1) * batch_size])
print("test accuracy: %f" % (num_correct_pred / np.shape(data_loader.eval_labels)[0]))


