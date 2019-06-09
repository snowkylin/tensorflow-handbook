import tensorflow as tf
import numpy as np
from zh.model.mlp.mlp import MLP
from zh.model.cnn.cnn import CNN
from zh.model.utils import MNISTLoader

model_type = 'CNN'
num_batches = 200
batch_size = 50
learning_rate = 0.001


if __name__ == '__main__':
    if model_type == 'MLP':
        model = MLP()
    else:
        model = CNN()
    data_loader = MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
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


