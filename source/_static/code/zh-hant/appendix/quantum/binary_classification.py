import cirq
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

q = cirq.GridQubit(0, 0)

# 準備量子資料集(q_data, label)
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

# 建立參數化的量子電路（PQC）
theta = sympy.Symbol('theta')
q_model = cirq.Circuit(cirq.rx(theta)(q))

# 建立量子層和經典全連接層
q_layer = tfq.layers.PQC(q_model, cirq.Z(q))
dense_layer = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)

# 使用Keras建立訓練流程。量子資料首先通過PQC，然後通過經典的全連接模型
q_data_input = tf.keras.Input(shape=() ,dtype=tf.dtypes.string)
expectation_output = q_layer(q_data_input)
classifier_output = dense_layer(expectation_output)
model = tf.keras.Model(inputs=q_data_input, outputs=classifier_output)

# 編譯模型，指定優化器、損失函數和評估指標，並進行訓練
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)
model.fit(x=q_data, y=label, epochs=200)

# 輸出量子層參數（即theta）的訓練結果
print(q_layer.get_weights())