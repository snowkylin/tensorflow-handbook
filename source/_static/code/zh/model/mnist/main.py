import tensorflow as tf
import numpy as np
from zh.model.utils import MNISTLoader, SparseCategoricalAccuracy

model = 'CNN'           # 'MLP' or 'CNN'
mode = 'sequential'     # 'sequential', 'functional' or 'subclassing'
training_loop = 'custom' # 'keras', 'custom' or 'graph'
num_epochs = 5
batch_size = 50
learning_rate = 0.001

if training_loop == 'graph':
    tf.compat.v1.disable_eager_execution()


if model == 'MLP':
    if mode == 'sequential':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=tf.nn.relu),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Softmax()
        ])
    if mode == 'functional':
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Dense(units=10)(x)
        outputs = tf.keras.layers.Softmax()(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    if mode == 'subclassing':
        from zh.model.mnist.mlp import MLP
        model = MLP()
if model == 'CNN':
    if mode == 'sequential':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                filters=32,             # 卷积核数目
                kernel_size=[5, 5],     # 感受野大小
                padding="same",         # padding策略
                activation=tf.nn.relu   # 激活函数
            ),
            tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu
            ),
            tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
            tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,)),
            tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=10),
            tf.keras.layers.Softmax()
        ])
    if mode == 'functional':
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(
            filters=32,             # 卷积核数目
            kernel_size=[5, 5],     # 感受野大小
            padding="same",         # padding策略
            activation=tf.nn.relu   # 激活函数
        )(inputs)
        x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)
        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)
        x = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))(x)
        x = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Dense(units=10)(x)
        outputs = tf.keras.layers.Softmax()(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    if mode == 'subclassing':
        from zh.model.mnist.cnn import CNN
        model = CNN()


data_loader = MNISTLoader()
if training_loop == 'keras':
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)
    print(model.evaluate(data_loader.test_data, data_loader.test_label))
if training_loop == 'custom':
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index: end_index])
        sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())
if training_loop == 'graph':
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    # 建立计算图
    X_placeholder = tf.compat.v1.placeholder(name='X', shape=[None, 28, 28, 1], dtype=tf.float32)
    y_placeholder = tf.compat.v1.placeholder(name='y', shape=[None], dtype=tf.int32)
    y_pred = model(X_placeholder)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y_placeholder, y_pred=y_pred)
    loss = tf.reduce_mean(loss)
    train_op = optimizer.minimize(loss)
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    # 建立Session
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for batch_index in range(num_batches):
            X, y = data_loader.get_batch(batch_size)
            # 使用Session.run()将数据送入计算图节点，进行训练以及计算损失函数
            _, loss_value = sess.run([train_op, loss], feed_dict={X_placeholder: X, y_placeholder: y})
            print("batch %d: loss %f" % (batch_index, loss_value))

        num_batches = int(data_loader.num_test_data // batch_size)
        for batch_index in range(num_batches):
            start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
            y_pred = model.predict(data_loader.test_data[start_index: end_index])
            sess.run(sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred))
        print("test accuracy: %f" % sess.run(sparse_categorical_accuracy.result()))