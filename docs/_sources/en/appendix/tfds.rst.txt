TensorFlow Datasets: Ready-to-use Datasets
==========================================

`TensorFlow Datasets <https://www.tensorflow.org/datasets/>`_ 是一个开箱即用的数据集集合，包含数十种常用的机器学习数据集。通过简单的几行代码即可将数据以 ``tf.data.Datasets`` 的格式载入。关于 ``tf.data.Datasets`` 的使用可参考 :ref:`tf.data <tfdata>`。

该工具是一个独立的Python包，可以通过::

    pip install tensorflow-datasets

安装。

在使用时，首先使用import导入该包

.. code-block:: python

    import tensorflow as tf
    import tensorflow_datasets as tfds

然后，最基础的用法是使用 ``tfds.load`` 方法，载入所需的数据集，如：

.. code-block:: python

    dataset = tfds.load("mnist", split=tfds.Split.TRAIN)
    dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
    dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)

该方法返回一个 ``tf.data.Datasets`` 对象。部分重要的参数如下：

..
    https://www.tensorflow.org/datasets/api_docs/python/tfds/load

- ``as_supervised`` ：若为True，则根据数据集的特性返回为 ``(input, label)`` 格式，否则返回所有特征的字典。
- ``split``：指定返回数据集的特定部分，若无则返回整个数据集。一般有 ``tfds.Split.TRAIN`` （训练集）和 ``tfds.Split.TEST`` （测试集）选项。

当前支持的数据集可在 `官方文档 <https://www.tensorflow.org/datasets/datasets>`_ 或使用 ``tfds.list_builders()`` 查看。

当得到了 ``tf.data.Datasets`` 类型的数据集后，我们即可使用 ``tf.data`` 对数据集进行各种预处理以及读取数据。例如：

.. code-block:: python
    
    # 使用 TessorFlow Datasets 载入“tf_flowers”数据集
    dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
    # 对 dataset 进行大小调整、打散和分批次操作
    dataset = dataset.map(lambda img, label: (tf.image.resize(img, [224, 224]) / 255.0, label)) \
        .shuffle(1024) \
        .batch(32)
    # 迭代数据
    for images, labels in dataset:
        # 对images和labels进行操作

详细操作说明可见 :ref:`本文档的 tf.data 一节 <tfdata>` 。

.. hint:: 在使用 TensorFlow Datasets 时，可能需要设置代理。较为简易的方式是设置 ``TFDS_HTTPS_PROXY`` 环境变量，即

    ::

        export TFDS_HTTPS_PROXY=http://代理服务器IP:端口