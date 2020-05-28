import tensorflow as tf
import numpy as np

class Dense(tf.keras.layers.Layer):
    def __init__(self, num_units, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units

    def build(self, input_shape):
        self.weight = self.add_variable(name='weight', shape=[input_shape[-1], self.num_units])
        self.bias = self.add_variable(name='bias', shape=[self.num_units])

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.weight) + self.bias
        return y_pred

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(num_units=10, name='dense1')
        self.dense2 = Dense(num_units=10, name='dense2')

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(inputs)
        x = self.dense2(inputs)
        return x

model = Model()
print(model(np.random.rand(10, 32)))
graph = model.call.get_concrete_function(np.random.rand(10, 32))
print(graph.variables)
