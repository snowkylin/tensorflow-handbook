import tensorflow as tf
import time
from zh.model.mnist.cnn import CNN

num_batches = 100
batch_size = 50
learning_rate = 0.001

model = CNN()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def prepare_mnist_features_and_labels(x, y):
    x = tf.expand_dims(tf.cast(x, tf.float32), axis=-1) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y

# @tf.function
def train():    
    # 使用TensorFlow内置的tf.data预处理数据集
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(prepare_mnist_features_and_labels)
    dataset = dataset.take(20000).shuffle(20000).batch(batch_size)
    batch_index = 0
    for X, y in dataset:        
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
        tf.print("batch", batch_index, ", loss", loss)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        batch_index += 1

if __name__ == '__main__':    
    start_time = time.time()
    train()
    end_time = time.time()
    print(end_time - start_time)
        