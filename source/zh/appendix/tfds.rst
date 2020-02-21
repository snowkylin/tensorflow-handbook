TensorFlow Datasets 数据集载入
============================================

`TensorFlow Datasets <https://www.tensorflow.org/datasets/>`_ 是一个开箱即用的数据集集合，包含数十种常用的机器学习数据集。通过简单的几行代码即可将数据以 ``tf.data.Dataset`` 的格式载入。关于 ``tf.data.Dataset`` 的使用可参考 :ref:`tf.data <tfdata>`。

该工具是一个独立的Python包，可以通过::

    pip install tensorflow-datasets

安装。

在使用时，首先使用import导入该包

.. code-block:: python

    import tensorflow as tf
    import tensorflow_datasets as tfds

然后，最基础的用法是使用 ``tfds.load`` 方法，载入所需的数据集。例如，以下三行代码分别载入了MNIST、猫狗分类和 ``tf_flowers`` 三个图像分类数据集：

.. code-block:: python

    dataset = tfds.load("mnist", split=tfds.Split.TRAIN)
    dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
    dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)

当第一次载入特定数据集时，TensorFlow Datasets 会自动从云端下载数据集到本地，并显示下载进度。例如，载入MNIST数据集时，终端输出提示如下：

::

    Downloading and preparing dataset mnist (11.06 MiB) to C:\Users\snowkylin\tensorflow_datasets\mnist\3.0.0...
    WARNING:absl:Dataset mnist is hosted on GCS. It will automatically be downloaded to your
    local data directory. If you'd instead prefer to read directly from our public
    GCS bucket (recommended if you're running on GCP), you can instead set
    data_dir=gs://tfds-data/datasets.

    Dl Completed...: 100%|██████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:10<00:00,  2.93s/ file] 
    Dl Completed...: 100%|██████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:10<00:00,  2.73s/ file] 
    Dataset mnist downloaded and prepared to C:\Users\snowkylin\tensorflow_datasets\mnist\3.0.0. Subsequent calls will reuse this data.

.. hint:: 在使用 TensorFlow Datasets 时，可能需要设置代理。较为简易的方式是设置 ``TFDS_HTTPS_PROXY`` 环境变量，即

    ::

        export TFDS_HTTPS_PROXY=http://代理服务器IP:端口

``tfds.load`` 方法返回一个 ``tf.data.Dataset`` 对象。部分重要的参数如下：

..
    https://www.tensorflow.org/datasets/api_docs/python/tfds/load

- ``as_supervised`` ：若为True，则根据数据集的特性，将数据集中的每行元素整理为有监督的二元组 ``(input, label)`` （即“数据+标签”）形式，否则数据集中的每行元素为包含所有特征的字典。
- ``split``：指定返回数据集的特定部分。若不指定，则返回整个数据集。一般有 ``tfds.Split.TRAIN`` （训练集）和 ``tfds.Split.TEST`` （测试集）选项。

TensorFlow Datasets 当前支持的数据集可在 `官方文档 <https://www.tensorflow.org/datasets/datasets>`_ 查看，或者也可以使用 ``tfds.list_builders()`` 查看。

当得到了 ``tf.data.Dataset`` 类型的数据集后，我们即可使用 ``tf.data`` 对数据集进行各种预处理以及读取数据。例如：

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

详细操作说明可见 :ref:`本手册的 tf.data 一节 <tfdata>` 。同时，本手册的 :doc:`分布式训练 <../appendix/distributed>` 一章也使用了 TensorFlow Datasets 载入数据集。可以参考这些章节的示例代码以进一步了解 TensorFlow Datasets 的使用方法。