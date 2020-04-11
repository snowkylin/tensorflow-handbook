import cirq
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

q = cirq.GridQubit(0, 0)
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

theta = sympy.Symbol('theta')
q_model = cirq.Circuit()
q_data_input = tf.keras.Input(shape=() ,dtype=tf.dtypes.string)
expectation_output = tfq.layers.PQC(q_model, cirq.Z(q))(q_data_input)
classifier_output = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)(expectation_output)
model = tf.keras.Model(inputs=q_data_input, outputs=classifier_output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)
model.fit(x=q_data, y=label, epochs=20)

print(model.predict(q_data))