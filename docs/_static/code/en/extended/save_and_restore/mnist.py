import tensorflow as tf
import numpy as np
from en.model.mlp.mlp import MLP
from en.model.mlp.utils import DataLoader

tf.enable_eager_execution()
mode = 'test'
num_batches = 1000
batch_size = 50
learning_rate = 0.001
data_loader = DataLoader()


def train():
    model = MLP()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)      # instantiate a Checkpoint, set `model` as object to be saved
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_logit_pred = model(tf.convert_to_tensor(X))
            loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        if (batch_index + 1) % 100 == 0:                        # save every 100 batches
            checkpoint.save('./save/model.ckpt')                # save model to .ckpt file


def test():
    model_to_be_restored = MLP()
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)      # instantiate a Checkpoint, set newly initialized model `model_to_be_restored` to be the object to be restored
    checkpoint.restore(tf.train.latest_checkpoint('./save'))    # restore parameters of model from file
    num_eval_samples = np.shape(data_loader.eval_labels)[0]
    y_pred = model_to_be_restored.predict(tf.constant(data_loader.eval_data)).numpy()
    print("test accuracy: %f" % (sum(y_pred == data_loader.eval_labels) / num_eval_samples))


if __name__ == '__main__':
    if mode == 'train':
        train()
    if mode == 'test':
        test()
