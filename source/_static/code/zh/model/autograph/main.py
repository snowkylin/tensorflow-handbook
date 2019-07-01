import tensorflow as tf
import time
from zh.model.mnist.cnn import CNN
from zh.model.utils import MNISTLoader

num_batches = 400
batch_size = 50
learning_rate = 0.001
data_loader = MNISTLoader()

@tf.function
def train_one_step(X, y):    
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        # 注意这里使用了Graph Execution下的tf.print()
        tf.print("batch", batch_index, "loss", loss)  
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

if __name__ == '__main__':
    model = CNN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    start_time = time.time()
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        train_one_step(X, y)
    end_time = time.time()
    print(end_time - start_time)      