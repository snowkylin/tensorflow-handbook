import tensorflow as tf
import numpy as np
from zh.model.utils import MNISTLoader

num_epochs = 10
batch_size = 50
learning_rate = 0.001
data_loader = MNISTLoader()
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)
print(model.evaluate(data_loader.test_data, data_loader.test_label))