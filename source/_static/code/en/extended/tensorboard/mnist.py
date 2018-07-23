import tensorflow as tf
import numpy as np
from model.mlp.mlp import MLP
from model.mlp.utils import DataLoader

tf.enable_eager_execution()
num_batches = 10000
batch_size = 50
learning_rate = 0.001
model = MLP()
data_loader = DataLoader()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
summary_writer = tf.contrib.summary.create_file_writer('./tensorboard')     # 实例化记录器
with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_logit_pred = model(tf.convert_to_tensor(X))
            loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
            tf.contrib.summary.scalar("loss", loss, step=batch_index)       # 记录当前loss
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
