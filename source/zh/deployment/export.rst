TensorFlow模型导出
====================================================

.. _savedmodel:

使用SavedModel完整导出模型
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


----------------------------------------------------------------

Keras Sequential save方法（Jinpeng）
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
