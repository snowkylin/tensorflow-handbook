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
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]


if model_type == 'MLP':
    model = MLP()
else:
    model = CNN()
data_loader = DataLoader()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_logit_pred = model(X)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

num_test_samples = np.shape(data_loader.test_label)[0]
y_pred = model.predict(data_loader.test_data).numpy()
print("test accuracy: %f" % (sum(y_pred == data_loader.test_label) / num_test_samples))

num_correct_pred = 0
for batch_index in range(num_test_samples // batch_size):
    y_pred = model.predict(data_loader.test_data[batch_index * batch_size: (batch_index + 1) * batch_size]).numpy()
    num_correct_pred += sum(y_pred == data_loader.test_label[batch_index * batch_size: (batch_index + 1) * batch_size])
print("test accuracy: %f" % (num_correct_pred / np.shape(data_loader.test_label)[0]))


