TensorFlow Model Export
====================================================

In order to deploy trained machine learning models to various target platforms (e.g. servers, mobile, embedded devices, browsers, etc.), our first step is often to export (serialize) the entire trained model into a series of files with standard format. TensorFlow provides a unified model export format, SavedModel, which allows us to deploy our trained models on a variety of platforms using this format as an intermediary. It is the main export format we use in TensorFlow 2. Also, for historical reasons, Keras's Sequential and Functional models have their own model export formats, which we will also introduce later.

.. _en_savedmodel:

Export models by SavedModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/beta/guide/saved_model

In the previous section we introduced :ref:`Checkpoint <en_chechpoint>`, which helps us save and recover the weights in the model. SavedModel, as a model export format, goes one step further and contains complete information about a TensorFlow program: not only the weights of the model, but also the computation process (i.e., the dataflow graph). When the model is exported as a SavedModel file, the model can be run again without source code, which makes SavedModel especially suitable for model sharing and deployment. This format is used later in TensorFlow Serving (server-side deployment), TensorFlow Lite (mobile-side deployment), and TensorFlow.js.

All Keras models can be easily exported to SavedModel format. It should be noted, however, that since SavedModel is based on graph execution mode, any method (e.g. ``call``) that needs to be exported to SavedModel format requires to be decorated by ``@tf.function`` (see :ref:` previous <tffunction>` for the usage of ``@tf.function``. Models built with sequantial or function API is not required for the decoration). Then, assuming we have a Keras model named ``model``, the model can be exported as SavedModel using the following code.

.. code-block:: python

    tf.saved_model.save(model, "target export folder")

When you need to load a SavedModel file, use

.. code-block:: python

    model = tf.saved_model.load("target export folder")

.. admonition:: Hint

    For the Keras model ``model`` inheriting ``tf.keras.Model`` class, loaded instance using the SavedModel will not allow direct inference using ``model()`, but will require the use of ``model.call()``.

Here is a simple example of exporting and importing the model of :ref:`previous MNIST digit classification task <en_mlp>`.

Export the model to the ``saved/1`` folder.

.. literalinclude:: /_static/code/zh/savedmodel/keras/train_and_export.py
    :emphasize-lines: 22

Import and test the performance of the exported model in ``saved/1``.

.. literalinclude:: /_static/code/zh/savedmodel/keras/load_savedmodel.py
    :emphasize-lines: 6, 12

Output::

    test accuracy: 0.952000

Keras models inheriting ``tf.keras.Model`` class can also be exported in the same way, but note that the ``call`` method requires a ``@tf.function`` modification to translate into a SavedModel-supported dataflow graph. The following code is an example

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

The process of importing the model is the same, except that model inference requires an explicit call to the ``call`` method, i.e. using.

.. code-block:: python

        y_pred = model.call(data_loader.test_data[start_index: end_index])

Built-in export format in Keras（Jinpeng）
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

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 192 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>