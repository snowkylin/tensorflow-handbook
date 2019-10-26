import tensorflow as tf
from zh.model.mnist.mlp import MLP
from zh.model.utils import MNISTLoader

num_epochs = 5
batch_size = 50
learning_rate_1 = 0.001
learning_rate_2 = 0.01

model = MLP()
data_loader = MNISTLoader()
# 声明两个优化器，设定不同的学习率，分别用于更新MLP模型的第一层和第二层
optimizer_1 = tf.keras.optimizers.Adam(learning_rate=learning_rate_1)
optimizer_2 = tf.keras.optimizers.Adam(learning_rate=learning_rate_2)
num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape(persistent=True) as tape:  # 声明一个持久的GradientTape，允许我们多次调用tape.gradient方法
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.dense1.variables)    # 单独求第一层参数的梯度
    optimizer_1.apply_gradients(grads_and_vars=zip(grads, model.dense1.variables)) # 单独对第一层参数更新，学习率0.001
    grads = tape.gradient(loss, model.dense2.variables)    # 单独求第二层参数的梯度
    optimizer_1.apply_gradients(grads_and_vars=zip(grads, model.dense2.variables)) # 单独对第二层参数更新，学习率0.01