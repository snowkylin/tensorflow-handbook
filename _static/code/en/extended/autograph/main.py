import tensorflow as tf
import numpy as np
from tensorflow.contrib import autograph


class RNN(tf.keras.Model):
    def __init__(self, num_chars, batch_size, num_units):
        super().__init__()
        self.num_chars = num_chars
        self.num_units = num_units
        batch_size = batch_size
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    @autograph.convert()
    def call(self, inputs, seq_length):
        # batch_size, seq_length = tf.shape(inputs)
        inputs = tf.one_hot(inputs, depth=self.num_chars)       # [batch_size, seq_length, num_chars]
        state = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        output = tf.zeros(shape=[self.batch_size, self.num_units], dtype=tf.float32)
        for t in range(seq_length):
            output, state = self.cell(inputs[:, t, :], state)
        output = self.dense(output)
        return output

    def predict(self, inputs, seq_length, temperature=1.):
        logits = self(inputs, seq_length)
        prob = tf.nn.softmax(logits / temperature)
        return prob


class DataLoader():
    def __init__(self):
        path = tf.keras.utils.get_file('nietzsche.txt',
            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.text = [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index:index+seq_length])
            next_char.append(self.text[index+seq_length])
        return np.array(seq), np.array(next_char)       # [num_batch, seq_length], [num_batch]


if __name__ == '__main__':
    num_batches = 100
    seq_length = 40
    batch_size = 50
    rnn_size = 256
    learning_rate = 1e-3

    data_loader = DataLoader()
    model = RNN(len(data_loader.chars), rnn_size)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    X_placeholder = tf.placeholder(name='X', shape=[None, seq_length], dtype=tf.int32)
    y_placeholder = tf.placeholder(name='y', shape=[None], dtype=tf.int32)
    seq_length_placeholder = tf.placeholder(name='seq_length', shape=None, dtype=tf.int32)
    y_logit_pred = model(X_placeholder, batch_size, seq_length_placeholder)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_placeholder, logits=y_logit_pred)
    train_op = optimizer.minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for batch_index in range(num_batches):
            X, y = data_loader.get_batch(seq_length, batch_size)
            _, train_loss = sess.run([train_op, loss], feed_dict={X_placeholder: X, y_placeholder: y, seq_length_placeholder: seq_length})
            print("batch %d: loss %f" % (batch_index, train_loss))

    prob = model.predict(X_placeholder)
    X_, _ = data_loader.get_batch(seq_length, 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        X = X_
        print("diversity %f:" % diversity)
        for t in range(400):
            test_prob = sess.run(prob, feed_dict={X_placeholder: X})
            y_pred = np.array([np.random.choice(len(data_loader.chars), p=test_prob[i, :])
                               for i in range(batch_size)])
            print(data_loader.indices_char[y_pred[0]], end='', flush=True)
            X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)