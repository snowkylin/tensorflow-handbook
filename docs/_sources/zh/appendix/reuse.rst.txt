TensorFlow资源重用
============================================

TensorFlow Hub （jinpeng）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TF Hub目的是为了更好复用已训练好的模型，可节省海量的训练时间和计算资源。预训练好的模型，可以进行直接部署，也可以进行迁移学习（Transfer Learning）。

本书是使用TF Hub中的Inception V3模型，针对mnist图像进行迁移学习。
mnist图像资源获取地址：https://github.com/dpinthinker/mnist_image_png_jpeg

具体操作方法参考https://www.tensorflow.org/hub/tutorials/image_retraining，如下

1. 获取retrain.py脚本

.. code-block:: bash

    curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py

2. 进行迁移学习，该过程会持续很长时间，停止后可以增量训练

.. code-block:: bash

    python retrain.py --image_dir your_mnist_img_path

默认生成的模型在/tmp/output_graph.pb，对应的label数据在/tmp/output_labels.txt。

3. 获取label_imange.py脚本

.. code-block:: bash

    curl -LO https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py

4. 对迁移学习训练完的模型进行验证

.. code-block:: bash

    python label_image.py \
    --graph=/tmp/output_graph.pb \
    --labels=/tmp/output_labels.txt \
    --input_layer=Placeholder \
    --output_layer=final_result \
    --image=your_mnist_test_img_<number>.jpg

结果如下：

.. code-block:: bash

    3 0.92819667
    2 0.027902907
    5 0.018210107
    9 0.010902734
    7 0.0056838035

这里产生的模型output_graph.pb，在移动端部署章节会被使用到。

TensorFlow Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/datasets/

TensorFlow Datasets是一个开箱即用的数据集集合，包含数十种常用的机器学习数据集。通过简单的几行代码即可将数据以 ``tf.data.Datasets`` 的格式载入。

该工具是一个独立的Python包，可以通过::

    pip install tensorflow-datasets

安装。

在使用时，首先使用import导入该包

.. code-block:: python

    import tensorflow as tf
    import tensorflow_datasets as tfds

然后，最基础的用法是使用tfds.load方法，载入所需的数据集，如：

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

Tensor2Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^