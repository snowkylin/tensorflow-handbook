import tensorflow as tf
import tensorflow_datasets as tfds

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

num_epochs = 5
batch_size_per_replica = 32
learning_rate = 0.001

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

# 载入数据集并预处理
def resize(image, label):
    image = tf.image.resize(image, [224, 224]) / 255.0
    return image, label

# 当as_supervised为True时，返回image和label两个键值
dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(resize).shuffle(1024).batch(batch_size)

with strategy.scope():
    model = tf.keras.applications.MobileNetV2()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

model.fit(dataset, epochs=num_epochs)