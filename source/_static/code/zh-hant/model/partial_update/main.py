import tensorflow as tf
from zh.model.mnist.mlp import MLP
from zh.model.utils import MNISTLoader

num_epochs = 5
batch_size = 50
learning_rate_1 = 0.001
learning_rate_2 = 0.01

model = MLP()
data_loader = MNISTLoader()
# 宣告兩個優化器，設定不同的學習率，分別用於更新MLP模型的第一層和第二層
optimizer_1 = tf.keras.optimizers.Adam(learning_rate=learning_rate_1)
optimizer_2 = tf.keras.optimizers.Adam(learning_rate=learning_rate_2)
num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape(persistent=True) as tape:  # 宣告一個永久的GradientTape，允許我們多次呼叫tape.gradient方法
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.dense1.variables)    # 單獨求第一層參數的梯度
    optimizer_1.apply_gradients(grads_and_vars=zip(grads, model.dense1.variables)) # 單獨對第一層參數更新，學習率0.001
    grads = tape.gradient(loss, model.dense2.variables)    # 單獨求第二層參數的梯度
    optimizer_2.apply_gradients(grads_and_vars=zip(grads, model.dense2.variables)) # 單獨對第二層參數更新，學習率0.01