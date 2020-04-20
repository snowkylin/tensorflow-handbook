import cirq
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

q = cirq.GridQubit(0, 0)

# 准备量子数据集(q_data, label)
add_noise = lambda x: x + np.random.normal(0, 0.25 * np.pi)
q_data = tfq.convert_to_tensor(
    [cirq.Circuit(
        cirq.rx(add_noise(0.5 * np.pi))(q), 
        cirq.ry(add_noise(0))(q)
        ) for _ in range(100)] + 
    [cirq.Circuit(
        cirq.rx(add_noise(1.5 * np.pi))(q), 
        cirq.ry(add_noise(0))(q)
        ) for _ in range(100)]
)
label = np.array([0] * 100 + [1] * 100)

# 建立参数化的量子线路（PQC）
theta = sympy.Symbol('theta')
q_model = cirq.Circuit(cirq.rx(theta)(q))

# 建立量子层和经典全连接层
q_layer = tfq.layers.PQC(q_model, cirq.Z(q))
dense_layer = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)

# 使用Keras建立训练流程。量子数据首先通过PQC，然后通过经典的全连接模型
q_data_input = tf.keras.Input(shape=() ,dtype=tf.dtypes.string)
expectation_output = q_layer(q_data_input)
classifier_output = dense_layer(expectation_output)
model = tf.keras.Model(inputs=q_data_input, outputs=classifier_output)

# 编译模型，指定优化器、损失函数和评估指标，并进行训练
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)
model.fit(x=q_data, y=label, epochs=200)

# 输出量子层参数（即theta）的训练结果
print(q_layer.get_weights())