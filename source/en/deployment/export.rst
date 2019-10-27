TensorFlow Model Saving
====================================================

.. _savedmodel:

Using SavedModel to export a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/beta/guide/saved_model

在部署模型时，我们的第一步往往是将训练好的整个模型完整导出为一系列标准格式的文件，然后即可在不同的平台上部署模型文件。这时，TensorFlow为我们提供了SavedModel这一格式。与前面介绍的Checkpoint不同，SavedModel包含了一个TensorFlow程序的完整信息： **不仅包含参数的权值，还包含计算的流程（即计算图）** 。当模型导出为SavedModel文件时，无需建立模型的源代码即可再次运行模型，这使得SavedModel尤其适用于模型的分享和部署。后文的TensorFlow Serving（服务器端部署模型）、TensorFlow Lite（移动端部署模型）以及TensorFlow.js都会用到这一格式。

Keras模型均可方便地导出为SavedModel格式。不过需要注意的是，因为SavedModel基于计算图，所以对于使用继承 ``tf.keras.Model`` 类建立的Keras模型，其需要导出到SavedModel格式的方法（比如 ``call`` ）都需要使用 ``@tf.function`` 修饰（ ``@tf.function`` 的使用方式见 :ref:`前文 <tffunction>` ）。然后，假设我们有一个名为 ``model`` 的Keras模型，使用下面的代码即可将模型导出为SavedModel：

.. code-block:: python

    tf.saved_model.save(model, "保存的目标文件夹名称")

在需要载入SavedModel文件时，使用

.. code-block:: python

    model = tf.saved_model.load("保存的目标文件夹名称")

即可。

.. hint:: 对于使用继承 ``tf.keras.Model`` 类建立的Keras模型 ``model`` ，使用SavedModel载入后将无法使用 ``model()`` 直接进行推断，而需要使用 ``model.call()`` 。

以下是一个简单的示例，将 :ref:`前文MNIST手写体识别的模型 <mlp>` 进行导出和导入。

导出模型到 ``saved/1`` 文件夹：

.. literalinclude:: /_static/code/zh/savedmodel/keras/train_and_export.py
    :emphasize-lines: 22

将 ``saved/1`` 中的模型导入并测试性能：

.. literalinclude:: /_static/code/zh/savedmodel/keras/load_savedmodel.py
    :emphasize-lines: 6, 12

输出::

    test accuracy: 0.952000

使用继承 ``tf.keras.Model`` 类建立的Keras模型同样可以以相同方法导出，唯须注意 ``call`` 方法需要以 ``@tf.function`` 修饰，以转化为SavedModel支持的计算图，代码如下：

.. code-block:: python
    :emphasize-lines: 8

    class MLP(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
            self.dense2 = tf.keras.layers.Dense(units=10)

        @tf.function
        def call(self, inputs):         # [batch_size, 28, 28, 1]
            x = self.flatten(inputs)    # [batch_size, 784]
            x = self.dense1(x)          # [batch_size, 100]
            x = self.dense2(x)          # [batch_size, 10]
            output = tf.nn.softmax(x)
            return output

    model = MLP()
    ...

模型导入并测试性能的过程也相同，唯须注意模型推断时需要显式调用 ``call`` 方法，即使用：

.. code-block:: python
    :emphasize-lines: 2

        ...
        y_pred = model.call(data_loader.test_data[start_index: end_index])
        ...

Keras Sequential save (Jinpeng)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们以keras模型训练和保存为例进行讲解，如下是keras官方的mnist模型训练样例。

源码地址::
    
    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

以上代码，是基于keras的Sequential构建了多层的卷积神经网络，并进行训练。

为了方便起见可使用如下命令拷贝到本地::

    curl -LO https://raw.githubusercontent.com/keras-team/keras/master/examples/mnist_cnn.py

然后，在最后加上如下一行代码（主要是对keras训练完毕的模型进行保存）::

    model.save('mnist_cnn.h5')


在终端中执行mnist_cnn.py文件，如下::

    python mnist_cnn.py

.. warning:: 该过程需要连接网络获取 ``mnist.npz`` 文件（https://s3.amazonaws.com/img-datasets/mnist.npz），会被保存到 ``$HOME/.keras/datasets/`` 。如果网络连接存在问题，可以通过其他方式获取 ``mnist.npz`` 后，直接保存到该目录即可。

执行过程会比较久，执行结束后，会在当前目录产生 ``mnist_cnn.h5`` 文件（HDF5格式），就是keras训练后的模型，其中已经包含了训练后的模型结构和权重等信息。

在服务器端，可以直接通过 ``keras.models.load_model("mnist_cnn.h5")`` 加载，然后进行推理；在移动设备需要将HDF5模型文件转换为TensorFlow Lite的格式，然后通过相应平台的Interpreter加载，然后进行推理。
