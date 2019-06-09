import tensorflow as tf
from zh.model.utils import MNISTLoader


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(100, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10),
#     tf.keras.layers.Softmax()
# ])
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(
#         filters=32,             # 卷积核数目
#         kernel_size=[5, 5],     # 感受野大小
#         padding="same",         # padding策略
#         activation=tf.nn.relu   # 激活函数
#     ),
#     tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
#     tf.keras.layers.Conv2D(
#         filters=64,
#         kernel_size=[5, 5],
#         padding="same",
#         activation=tf.nn.relu
#     ),
#     tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
#     tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,)),
#     tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
#     tf.keras.layers.Dense(units=10),
#     tf.keras.layers.Softmax()
# ])
from zh.model.cnn.cnn import CNN
from 

model = CNN()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)


data_loader = MNISTLoader()
model.fit(data_loader.train_data, data_loader.train_label, epochs=1)
print(model.evaluate(data_loader.test_data, data_loader.test_label))