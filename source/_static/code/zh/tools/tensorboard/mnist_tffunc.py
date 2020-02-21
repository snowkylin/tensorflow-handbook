import tensorflow as tf
import time
from zh.model.mnist.mlp import MLP
from zh.model.utils import MNISTLoader

num_batches = 400
batch_size = 50
learning_rate = 0.001
log_dir = 'tensorboard'
data_loader = MNISTLoader()
model = MLP()
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
summary_writer = tf.summary.create_file_writer(log_dir) 
tf.summary.trace_on(graph=True, profiler=True) 

@tf.function
def train_one_step(X, y):    
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        # 注意这里使用了TensorFlow内置的tf.print()。@tf.function不支持Python内置的print方法
        tf.print("loss", loss)
    grads = tape.gradient(loss, model.variables)    
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

start_time = time.time()
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    train_one_step(X, y)
end_time = time.time()
print(end_time - start_time)
with summary_writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)
