Model Construction and Training
===============================

.. _linear:

本章介绍如何使用 TensorFlow 快速搭建动态模型。

- 模型的构建： ``tf.keras.Model`` 和 ``tf.keras.layers``
- 模型的损失函数： ``tf.keras.losses``
- 模型的优化器： ``tf.keras.optimizer``
- 模型的评估： ``tf.keras.metrics``

.. admonition:: 前置知识

    * `Python 面向对象编程 <http://www.runoob.com/python3/python3-class.html>`_ （在 Python 内定义类和方法、类的继承、构造和析构函数，`使用 super() 函数调用父类方法 <http://www.runoob.com/python/python-func-super.html>`_ ，`使用__call__() 方法对实例进行调用 <https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014319098638265527beb24f7840aa97de564ccc7f20f6000>`_ 等）；
    * 多层感知机、卷积神经网络、循环神经网络和强化学习（每节之前给出参考资料）。
    *  `Python 的函数装饰器 <https://www.runoob.com/w3cnote/python-func-decorators.html>`_ （非必须）


Models and layers
^^^^^^^^^^^^^^^^^
..  https://www.tensorflow.org/programmers_guide/eager

在 TensorFlow 中，推荐使用 Keras（ ``tf.keras`` ）构建模型。Keras 是一个广为流行的高级神经网络 API，简单、快速而不失灵活性，现已得到 TensorFlow 的官方内置和全面支持。

Keras 有两个重要的概念： **模型（Model）** 和 **层（Layer）** 。层将各种计算流程和变量进行了封装（例如基本的全连接层，CNN 的卷积层、池化层等），而模型则将各种层进行组织和连接，并封装成一个整体，描述了如何将输入数据通过各种层以及运算而得到输出。在需要模型调用的时候，使用 ``y_pred = model(X)`` 的形式即可。Keras 在 ``tf.keras.layers`` 下内置了深度学习中大量常用的的预定义层，同时也允许我们自定义层。

Keras 模型以类的形式呈现，我们可以通过继承 ``tf.keras.Model`` 这个 Python 类来定义自己的模型。在继承类中，我们需要重写 ``__init__()`` （构造函数，初始化）和 ``call(input)`` （模型调用）两个方法，同时也可以根据需要增加自定义的方法。

.. code-block:: python

    class MyModel(tf.keras.Model):
        def __init__(self):
            super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
            # 此处添加初始化代码（包含 call 方法中会用到的层），例如
            # layer1 = tf.keras.layers.BuiltInLayer(...)
            # layer2 = MyCustomLayer(...)

        def call(self, input):
            # 此处添加模型调用的代码（处理输入并返回输出），例如
            # x = layer1(input)
            # output = layer2(x)
            return output

        # 还可以添加自定义的方法

.. figure:: /_static/image/model/model.png
    :width: 50%
    :align: center

    Keras 模型类定义示意图

继承 ``tf.keras.Model`` 后，我们同时可以使用父类的若干方法和属性，例如在实例化类 ``model = Model()`` 后，可以通过 ``model.variables`` 这一属性直接获得模型中的所有变量，免去我们一个个显式指定变量的麻烦。

上一章中简单的线性模型 ``y_pred = a * X + b`` ，我们可以通过模型类的方式编写如下：

.. literalinclude:: /_static/code/zh/model/linear/linear.py

这里，我们没有显式地声明 ``a`` 和 ``b`` 两个变量并写出 ``y_pred = a * X + b`` 这一线性变换，而是建立了一个继承了 ``tf.keras.Model`` 的模型类 ``Linear`` 。这个类在初始化部分实例化了一个 **全连接层** （ ``tf.keras.layers.Dense`` ），并在 call 方法中对这个层进行调用，实现了线性变换的计算。如果需要显式地声明自己的变量并使用变量进行自定义运算，或者希望了解 Keras 层的内部原理，请参考 :ref:`自定义层 <custom_layer>`。

.. admonition:: Keras 的全连接层：线性变换 + 激活函数

    `全连接层 <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`_ （Fully-connected Layer，``tf.keras.layers.Dense`` ）是 Keras 中最基础和常用的层之一，对输入矩阵 :math:`A` 进行 :math:`f(AW + b)` 的线性变换 + 激活函数操作。如果不指定激活函数，即是纯粹的线性变换 :math:`AW + b`。具体而言，给定输入张量 ``input = [batch_size, input_dim]`` ，该层对输入张量首先进行 ``tf.matmul(input, kernel) + bias`` 的线性变换（ ``kernel`` 和 ``bias`` 是层中可训练的变量），然后对线性变换后张量的每个元素通过激活函数 ``activation`` ，从而输出形状为 ``[batch_size, units]`` 的二维张量。

    .. figure:: /_static/image/model/dense.png
        :width: 60%
        :align: center

    其包含的主要参数如下：

    * ``units`` ：输出张量的维度；
    * ``activation`` ：激活函数，对应于 :math:`f(AW + b)` 中的 :math:`f` ，默认为无激活函数（ ``a(x) = x`` ）。常用的激活函数包括 ``tf.nn.relu`` 、 ``tf.nn.tanh`` 和 ``tf.nn.sigmoid`` ；
    * ``use_bias`` ：是否加入偏置向量 ``bias`` ，即 :math:`f(AW + b)` 中的 :math:`b`。默认为 ``True`` ；
    * ``kernel_initializer`` 、 ``bias_initializer`` ：权重矩阵 ``kernel`` 和偏置向量 ``bias`` 两个变量的初始化器。默认为 ``tf.glorot_uniform_initializer`` [#glorot]_ 。设置为 ``tf.zeros_initializer`` 表示将两个变量均初始化为全 0；

    该层包含权重矩阵 ``kernel = [input_dim, units]`` 和偏置向量 ``bias = [units]`` [#broadcast]_ 两个可训练变量，对应于 :math:`f(AW + b)` 中的 :math:`W` 和 :math:`b`。

    这里着重从数学矩阵运算和线性变换的角度描述了全连接层。基于神经元建模的描述可参考 :ref:`后文介绍 <neuron>` 。

    .. [#glorot] Keras 中的很多层都默认使用 ``tf.glorot_uniform_initializer`` 初始化变量，关于该初始化器可参考 https://www.tensorflow.org/api_docs/python/tf/glorot_uniform_initializer 。
    .. [#broadcast] 你可能会注意到， ``tf.matmul(input, kernel)`` 的结果是一个形状为 ``[batch_size, units]`` 的二维矩阵，这个二维矩阵要如何与形状为 ``[units]`` 的一维偏置向量 ``bias`` 相加呢？事实上，这里是 TensorFlow 的 Broadcasting 机制在起作用，该加法运算相当于将二维矩阵的每一行加上了 ``Bias`` 。Broadcasting 机制的具体介绍可见 https://www.tensorflow.org/xla/broadcasting 。

.. admonition:: 为什么模型类是重载 ``call()`` 方法而不是  ``__call__()`` 方法？

    在 Python 中，对类的实例 ``myClass`` 进行形如 ``myClass()`` 的调用等价于 ``myClass.__call__()`` （具体请见本章初 “前置知识” 的 ``__call__()`` 部分）。那么看起来，为了使用 ``y_pred = model(X)`` 的形式调用模型类，应该重写 ``__call__()`` 方法才对呀？原因是 Keras 在模型调用的前后还需要有一些自己的内部操作，所以暴露出一个专门用于重载的 ``call()`` 方法。 ``tf.keras.Model`` 这一父类已经包含 ``__call__()`` 的定义。 ``__call__()`` 中主要调用了 ``call()`` 方法，同时还需要在进行一些 keras 的内部操作。这里，我们通过继承 ``tf.keras.Model`` 并重载 ``call()`` 方法，即可在保持 keras 结构的同时加入模型调用的代码。

.. _mlp:

A basic example: Multilayer Perceptron (MLP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们从编写一个最简单的 `多层感知机 <https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E5%99%A8>`_ （Multilayer Perceptron, MLP），或者说 “多层全连接神经网络” 开始，介绍 TensorFlow 的模型编写方式。在这一部分，我们依次进行以下步骤：

- 使用 ``tf.keras.datasets`` 获得数据集并预处理
- 使用 ``tf.keras.Model`` 和 ``tf.keras.layers`` 构建模型
- 构建模型训练流程，使用 ``tf.keras.losses`` 计算损失函数，并使用 ``tf.keras.optimizer`` 优化模型
- 构建模型评估流程，使用 ``tf.keras.metrics`` 计算评估指标

.. admonition:: 基础知识和原理

    * UFLDL 教程 `Multi-Layer Neural Network <http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/>`_ 一节；
    * 斯坦福课程 `CS231n: Convolutional Neural Networks for Visual Recognition <http://cs231n.github.io/>`_ 中的 “Neural Networks Part 1 ~ 3” 部分。

这里，我们使用多层感知机完成 MNIST 手写体数字图片数据集 [LeCun1998]_ 的分类任务。

.. figure:: /_static/image/model/mnist_0-9.png
    :align: center

    MNIST 手写体数字图片示例

Data collection and preprocessing: ``tf.keras.datasets``
--------------------------------------------------------

先进行预备工作，实现一个简单的 ``MNISTLoader`` 类来读取 MNIST 数据集数据。这里使用了 ``tf.keras.datasets`` 快速载入 MNIST 数据集。

.. literalinclude:: /_static/code/zh/model/utils.py
    :lines: 5-19

.. hint:: ``mnist = tf.keras.datasets.mnist`` 将从网络上自动下载 MNIST 数据集并加载。如果运行时出现网络连接错误，可以从 https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz 或 https://s3.amazonaws.com/img-datasets/mnist.npz 下载 MNIST 数据集 ``mnist.npz`` 文件，并放置于用户目录的 ``.keras/dataset`` 目录下（Windows 下用户目录为 ``C:\Users\用户名`` ，Linux 下用户目录为 ``/home/用户名`` ）。

.. admonition:: TensorFlow 的图像数据表示

    在 TensorFlow 中，图像数据集的一种典型表示是 ``[图像数目，长，宽，色彩通道数]`` 的四维张量。在上面的 ``DataLoader`` 类中， ``self.train_data`` 和 ``self.test_data`` 分别载入了 60,000 和 10,000 张大小为 ``28*28`` 的手写体数字图片。由于这里读入的是灰度图片，色彩通道数为 1（彩色 RGB 图像色彩通道数为 3），所以我们使用 ``np.expand_dims()`` 函数为图像数据手动在最后添加一维通道。

.. _mlp_model:

Model construction: ``tf.keras.Model`` and ``tf.keras.layers``
--------------------------------------------------------------

多层感知机的模型类实现与上面的线性模型类似，使用 ``tf.keras.Model`` 和 ``tf.keras.layers`` 构建，所不同的地方在于层数增加了（顾名思义，“多层” 感知机），以及引入了非线性激活函数（这里使用了 `ReLU 函数 <https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E6%95%B4%E6%B5%81%E5%87%BD%E6%95%B0>`_ ， 即下方的 ``activation=tf.nn.relu`` ）。该模型输入一个向量（比如这里是拉直的 ``1×784`` 手写体数字图片），输出 10 维的向量，分别代表这张图片属于 0 到 9 的概率。

.. literalinclude:: /_static/code/zh/model/mnist/mlp.py
    :lines: 4-

.. admonition:: softmax 函数

    这里，因为我们希望输出 “输入图片分别属于 0 到 9 的概率”，也就是一个 10 维的离散概率分布，所以我们希望这个 10 维向量至少满足两个条件：

    * 该向量中的每个元素均在 :math:`[0, 1]` 之间；
    * 该向量的所有元素之和为 1。

    为了使得模型的输出能始终满足这两个条件，我们使用 `Softmax 函数 <https://zh.wikipedia.org/wiki/Softmax%E5%87%BD%E6%95%B0>`_ （归一化指数函数， ``tf.nn.softmax`` ）对模型的原始输出进行归一化。其形式为 :math:`\sigma(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}` 。不仅如此，softmax 函数能够凸显原始向量中最大的值，并抑制远低于最大值的其他分量，这也是该函数被称作 softmax 函数的原因（即平滑化的 argmax 函数）。

.. figure:: /_static/image/model/mlp.png
    :width: 80%
    :align: center

    MLP 模型示意图

Model training: ``tf.keras.losses`` and ``tf.keras.optimizer``
--------------------------------------------------------------

定义一些模型超参数：

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 8-10

实例化模型和数据读取类，并实例化一个 ``tf.keras.optimizer`` 的优化器（这里使用常用的 Adam 优化器）：

.. code-block:: python

    model = MLP()
    data_loader = MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

然后迭代进行以下步骤：

- 从 DataLoader 中随机取一批训练数据；
- 将这批数据送入模型，计算出模型的预测值；
- 将模型预测值与真实值进行比较，计算损失函数（loss）。这里使用 ``tf.keras.losses`` 中的交叉熵函数作为损失函数；
- 计算损失函数关于模型变量的导数；
- 将求出的导数值传入优化器，使用优化器的 ``apply_gradients`` 方法更新模型参数以最小化损失函数（优化器的详细使用方法见 :ref:`前章 <optimizer>`  ）。

具体代码实现如下：

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 93-102

.. admonition:: 交叉熵（cross entropy）与 ``tf.keras.losses``

    你或许注意到了，在这里，我们没有显式地写出一个损失函数，而是使用了 ``tf.keras.losses`` 中的 ``sparse_categorical_crossentropy`` （交叉熵）函数，将模型的预测值 ``y_pred`` 与真实的标签值 ``y`` 作为函数参数传入，由 Keras 帮助我们计算损失函数的值。

    交叉熵作为损失函数，在分类问题中被广泛应用。其离散形式为 :math:`H(y, \hat{y}) = -\sum_{i=1}^{n}y_i \log(\hat{y_i})` ，其中 :math:`y` 为真实概率分布， :math:`\hat{y}` 为预测概率分布， :math:`n` 为分类任务的类别个数。预测概率分布与真实分布越接近，则交叉熵的值越小，反之则越大。更具体的介绍及其在机器学习中的应用可参考 `这篇博客文章 <https://blog.csdn.net/tsyccnh/article/details/79163834>`_ 。

    在 ``tf.keras`` 中，有两个交叉熵相关的损失函数 ``tf.keras.losses.categorical_crossentropy`` 和 ``tf.keras.losses.sparse_categorical_crossentropy`` 。其中 sparse 的含义是，真实的标签值 ``y_true`` 可以直接传入 int 类型的标签类别。具体而言：

    .. code-block:: python

        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)

    与

    .. code-block:: python

        loss = tf.keras.losses.categorical_crossentropy(
            y_true=tf.one_hot(y, depth=tf.shape(y_pred)[-1]),
            y_pred=y_pred
        )

    的结果相同。

Model evaluation: ``tf.keras.metrics``
--------------------------------------

最后，我们使用测试集评估模型的性能。这里，我们使用 ``tf.keras.metrics`` 中的 ``SparseCategoricalAccuracy`` 评估器来评估模型在测试集上的性能，该评估器能够对模型预测的结果与真实结果进行比较，并输出预测正确的样本数占总样本数的比例。我们迭代测试数据集，每次通过 ``update_state()`` 方法向评估器输入两个参数： ``y_pred`` 和 ``y_true`` ，即模型预测出的结果和真实结果。评估器具有内部变量来保存当前评估指标相关的参数数值（例如当前已传入的累计样本数和当前预测正确的样本数）。迭代结束后，我们使用 ``result()`` 方法输出最终的评估指标值（预测正确的样本数占总样本数的比例）。

在以下代码中，我们实例化了一个 ``tf.keras.metrics.SparseCategoricalAccuracy`` 评估器，并使用 For 循环迭代分批次传入了测试集数据的预测结果与真实结果，并输出训练后的模型在测试数据集上的准确率。

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 104-110

输出结果::

    test accuracy: 0.947900

可以注意到，使用这样简单的模型，已经可以达到 95% 左右的准确率。

.. _neuron:

.. admonition:: 神经网络的基本单位：神经元 [#order]_

    如果我们将上面的神经网络放大来看，详细研究计算过程，比如取第二层的第 k 个计算单元，可以得到示意图如下：

    .. figure:: /_static/image/model/neuron.png
        :width: 80%
        :align: center

    该计算单元 :math:`Q_k` 有 100 个权值参数 :math:`w_{0k}, w_{1k}, ..., w_{99k}` 和 1 个偏置参数 :math:`b_k` 。将第 1 层中所有的 100 个计算单元 :math:`P_0, P_1, ..., P_{99}` 的值作为输入，分别按权值 :math:`w_{ik}` 加和（即 :math:`\sum_{i=0}^{99} w_{ik} P_i` ），并加上偏置值 :math:`b_k` ，然后送入激活函数 :math:`f` 进行计算，即得到输出结果。

    事实上，这种结构和真实的神经细胞（神经元）类似。神经元由树突、胞体和轴突构成。树突接受其他神经元传来的信号作为输入（一个神经元可以有数千甚至上万树突），胞体对电位信号进行整合，而产生的信号则通过轴突传到神经末梢的突触，传播到下一个（或多个）神经元。

    .. figure:: /_static/image/model/real_neuron.png
        :width: 80%
        :align: center

        神经细胞模式图（修改自 Quasar Jarosz at English Wikipedia [CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0)]）

    上面的计算单元，可以被视作对神经元结构的数学建模。在上面的例子里，第二层的每一个计算单元（人工神经元）有 100 个权值参数和 1 个偏置参数，而第二层计算单元的数目是 10 个，因此这一个全连接层的总参数量为 100*10 个权值参数和 10 个偏置参数。事实上，这正是该全连接层中的两个变量 ``kernel`` 和 ``bias`` 的形状。仔细研究一下，你会发现，这里基于神经元建模的介绍与上文基于矩阵计算的介绍是等价的。

    .. [#order] 事实上，应当是先有神经元建模的概念，再有基于人工神经元和层结构的人工神经网络。但由于本手册着重介绍 TensorFlow 的使用方法，所以调换了介绍顺序。

Convolutional neural network (CNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`卷积神经网络 <https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C>`_ （Convolutional Neural Network, CNN）是一种结构类似于人类或动物的 `视觉系统 <https://zh.wikipedia.org/wiki/%E8%A7%86%E8%A7%89%E7%B3%BB%E7%BB%9F>`_ 的人工神经网络，包含一个或多个卷积层（Convolutional Layer）、池化层（Pooling Layer）和全连接层（Fully-connected Layer）。

.. admonition:: 基础知识和原理

    * 台湾大学李宏毅教授的《机器学习》课程的 `Convolutional Neural Network <https://www.bilibili.com/video/av10590361/?p=21>`_ 一章；
    * UFLDL 教程 `Convolutional Neural Network <http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/>`_ 一节；
    * 斯坦福课程 `CS231n: Convolutional Neural Networks for Visual Recognition <http://cs231n.github.io/>`_ 中的 “Module 2: Convolutional Neural Networks” 部分。


Implementing CNN with Keras
---------------------------

卷积神经网络的一个示例实现如下所示，和 :ref:`上节中的多层感知机 <mlp_model>` 在代码结构上很类似，只是新加入了一些卷积层和池化层。这里的网络结构并不是唯一的，可以增加、删除或调整 CNN 的网络结构和参数，以达到更好的性能。

.. literalinclude:: /_static/code/zh/model/mnist/cnn.py
    :lines: 4-

.. figure:: /_static/image/model/cnn.png
    :align: center

    示例代码中的 CNN 结构图示

将前节的 ``model = MLP()`` 更换成 ``model = CNN()`` ，输出如下::

    test accuracy: 0.988100

可以发现准确率相较于前节的多层感知机有非常显著的提高。事实上，通过改变模型的网络结构（比如加入 Dropout 层防止过拟合），准确率还有进一步提升的空间。

Using predefined classic CNN structure in Keras
-----------------------------------------------

``tf.keras.applications`` 中有一些预定义好的经典卷积神经网络结构，如 ``VGG16`` 、 ``VGG19`` 、 ``ResNet`` 、 ``MobileNet`` 等。我们可以直接调用这些经典的卷积神经网络结构（甚至载入预训练的参数），而无需手动定义网络结构。

例如，我们可以使用以下代码来实例化一个 ``MobileNetV2`` 网络结构：

.. code-block:: python

    model = tf.keras.applications.MobileNetV2()

当执行以上代码时，TensorFlow会自动从网络上下载 ``MobileNetV2`` 网络结构，因此在第一次执行代码时需要具备网络连接。每个网络结构具有自己特定的详细参数设置，一些共通的常用参数如下：

- ``input_shape`` ：输入张量的形状（不含第一维的Batch），大多默认为 ``224 × 224 × 3`` 。一般而言，模型对输入张量的大小有下限，长和宽至少为 ``32 × 32`` 或 ``75 × 75`` ；
- ``include_top`` ：在网络的最后是否包含全连接层，默认为 ``True`` ；
- ``weights`` ：预训练权值，默认为 ``'imagenet'`` ，即为当前模型载入在ImageNet数据集上预训练的权值。如需随机初始化变量可设为 ``None`` ；
- ``classes`` ：分类数，默认为1000。修改该参数需要 ``include_top`` 参数为 ``True`` 且 ``weights`` 参数为 ``None`` 。

各网络模型参数的详细介绍可参考 `Keras文档 <https://keras.io/applications/>`_ 。

以下展示一个例子，使用 ``MobileNetV2`` 网络在 ``tf_flowers`` 五分类数据集上进行训练（为了代码的简短高效，在该示例中我们使用了 :doc:`TensorFlow Datasets <../appendix/tfds>` 和 :ref:`tf.data <tfdata>` 载入和预处理数据）。通过将 ``weights`` 设置为 ``None`` ，我们随机初始化变量而不使用预训练权值。同时将 ``classes`` 设置为5，对应于5分类的数据集。

.. literalinclude:: /_static/code/zh/model/cnn/mobilenet.py
    :emphasize-lines: 10    

后文的部分章节（如 :doc:`分布式训练 <../appendix/distributed>` ）中，我们也会直接调用这些经典的网络结构来进行训练。

.. admonition:: 卷积层和池化层的工作原理

    卷积层（Convolutional Layer，以 ``tf.keras.layers.Conv2D`` 为代表）是 CNN 的核心组件，其结构与大脑的视觉皮层有类似之处。

    回忆我们之前建立的 :ref:`神经细胞的计算模型 <neuron>` 以及全连接层，我们默认每个神经元与上一层的所有神经元相连。不过，在视觉皮层的神经元中，情况并不是这样。你或许在生物课上学习过 **感受野** （Receptive Field）这一概念，即视觉皮层中的神经元并非与前一层的所有神经元相连，而只是感受一片区域内的视觉信号，并只对局部区域的视觉刺激进行反应。CNN 中的卷积层正体现了这一特性。

    例如，下图是一个 7×7 的单通道图片信号输入：

    .. figure:: /_static/image/model/conv_image.png
        :align: center

    如果使用之前基于全连接层的模型，我们需要让每个输入信号对应一个权值，即建模一个神经元需要 7×7=49 个权值（加上偏置项是50个），并得到一个输出信号。如果一层有 N 个神经元，我们就需要 49N 个权值，并得到 N 个输出信号。

    而在 CNN 的卷积层中，我们这样建模一个卷积层的神经元：

    .. figure:: /_static/image/model/conv_field.png
        :align: center

    图中 3×3 的红框代表该神经元的感受野。由此，我们只需 3×3=9 个权值 :math:`W = \begin{bmatrix}w_{1, 1} & w_{1, 2} & w_{1, 3} \\w_{2, 1} & w_{2, 2} & w_{2, 3} \\w_{3, 1} & w_{3, 2} & w_{3, 3}\end{bmatrix}`  ，外加1个偏置项 :math:`b`  ，即可得到一个输出信号。例如，对于红框所示的位置，输出信号即为对矩阵 :math:`\begin{bmatrix}0 \times w_{1, 1} & 0 \times w_{1, 2} & 0 \times w_{1, 3} \\0 \times w_{2, 1} & 1 \times w_{2, 2} & 0 \times w_{2, 3} \\0 \times w_{3, 1} & 0 \times w_{3, 2} & 2 \times w_{3, 3}\end{bmatrix}` 的所有元素求和并加上偏置项 :math:`b`，记作 :math:`a_{1, 1}`  。

    不过，3×3 的范围显然不足以处理整个图像，因此我们使用滑动窗口的方法。使用相同的参数 :math:`W` ，但将红框在图像中从左到右滑动，进行逐行扫描，每滑动到一个位置就计算一个值。例如，当红框向右移动一个单位时，我们计算矩阵 :math:`\begin{bmatrix}0 \times w_{1, 1} & 0 \times w_{1, 2} & 0 \times w_{1, 3} \\1 \times w_{2, 1} & 0 \times w_{2, 2} & 1 \times w_{2, 3} \\0 \times w_{3, 1} & 2 \times w_{3, 2} & 1 \times w_{3, 3}\end{bmatrix}` 的所有元素的和加上偏置项 :math:`b`，记作 :math:`a_{1, 2}` 。由此，和一般的神经元只能输出 1 个值不同，这里的卷积层神经元可以输出一个 5×5 的矩阵 :math:`A = \begin{bmatrix}a_{1, 1} & \cdots & a_{1, 5} \\ \vdots & & \vdots \\ a_{5, 1} & \cdots & a_{5, 5}\end{bmatrix}`  。

    .. figure:: /_static/image/model/conv_procedure.png
        :align: center

        卷积示意图。一个单通道的 7×7 图像在通过一个感受野为 3×3 ，参数为10个的卷积层神经元后，得到 5×5 的矩阵作为卷积结果。

    下面，我们使用TensorFlow来验证一下上图的计算结果。

    将上图中的输入图像、权值矩阵 :math:`W` 和偏置项 :math:`b` 表示为NumPy数组 ``image`` , ``W`` , ``b`` 如下：

    .. literalinclude:: /_static/code/zh/model/cnn/cnn_example.py
        :lines: 4-21

    然后建立一个仅有一个卷积层的模型，用 ``W`` 和 ``b`` 初始化 [#sequential]_ ：

    .. literalinclude:: /_static/code/zh/model/cnn/cnn_example.py
        :lines: 23-30

    最后将图像数据 ``image`` 输入模型，打印输出：

    .. literalinclude:: /_static/code/zh/model/cnn/cnn_example.py
        :lines: 32-33

    程序运行结果为：

    ::

        tf.Tensor(
        [[ 6.  5. -2.  1.  2.]
         [ 3.  0.  3.  2. -2.]
         [ 4.  2. -1.  0.  0.]
         [ 2.  1.  2. -1. -3.]
         [ 1.  1.  1.  3.  1.]], shape=(5, 5), dtype=float32)

    可见与上图中矩阵 :math:`A`  的值一致。
    
    还有一个问题，以上假设图片都只有一个通道（例如灰度图片），但如果图像是彩色的（例如有 RGB 三个通道）该怎么办呢？此时，我们可以为每个通道准备一个 3×3 的权值矩阵，即一共有 3×3×3=27 个权值。对于每个通道，均使用自己的权值矩阵进行处理，输出时将多个通道所输出的值进行加和即可。

    可能有读者会注意到，按照上述介绍的方法，每次卷积后的结果相比于原始图像而言，四周都会“少一圈”。比如上面 7×7 的图像，卷积后变成了 5×5 ，这有时会为后面的工作带来麻烦。因此，我们可以设定padding策略。在 ``tf.keras.layers.Conv2D`` 中，当我们将 ``padding`` 参数设为 ``same`` 时，会将周围缺少的部分使用0补齐，使得输出的矩阵大小和输入一致。

    最后，既然我们可以使用滑动窗口的方法进行卷积，那么每次滑动的步长是不是可以设置呢？答案是肯定的。通过 ``tf.keras.layers.Conv2D`` 的 ``strides`` 参数即可设置步长（默认为1）。比如，在上面的例子中，如果我们将步长设定为2，输出的卷积结果即会是一个3×3的矩阵。

    ..
        一个动态演示如下图所示。其中红色的矩阵为多通道的图像（这里展示为 2 个通道），绿色的矩阵为图像的每个通道所对应的权值矩阵 :math:`W` ，蓝色的矩阵为输出矩阵 :math:`A`  。

        .. figure:: /_static/image/model/conv_sliding_window.gif
            :align: center

            卷积示意图（来源： https://blog.csdn.net/huachao1001/article/details/79120521 ）

    事实上，卷积的形式多种多样，以上的介绍只是其中最简单和基础的一种。更多卷积方式的示例可见 `Convolution arithmetic <https://github.com/vdumoulin/conv_arithmetic>`_ 。

    池化层（Pooling Layer）的理解则简单得多，其可以理解为对图像进行降采样的过程，对于每一次滑动窗口中的所有值，输出其中的最大值（MaxPooling）、均值或其他方法产生的值。例如，对于一个三通道的 16×16 图像（即一个 ``16*16*3`` 的张量），经过感受野为 2×2，滑动步长为 2 的池化层，则得到一个 ``8*8*3`` 的张量。

    .. [#sequential] 这里使用了较为简易的Sequential模式建立模型，具体介绍见 :ref:`后文 <sequential_functional>`  。

Recurrent neural network (RNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

循环神经网络（Recurrent Neural Network, RNN）是一种适宜于处理序列数据的神经网络，被广泛用于语言模型、文本生成、机器翻译等。

.. admonition:: 基础知识和原理

    - `Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs <http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/>`_
    - 台湾大学李宏毅教授的《机器学习》课程的 `Recurrent Neural Network (part 1) <https://www.bilibili.com/video/av10590361/?p=36>`_ `Recurrent Neural Network (part 2) <https://www.bilibili.com/video/av10590361/?p=37>`_ 两部分。
    - LSTM 原理：`Understanding LSTM Networks <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_
    - RNN 序列生成：[Graves2013]_

这里，我们使用 RNN 来进行尼采风格文本的自动生成。 [#rnn_reference]_

这个任务的本质其实预测一段英文文本的接续字母的概率分布。比如，我们有以下句子::

    I am a studen

这个句子（序列）一共有 13 个字符（包含空格）。当我们阅读到这个由 13 个字符组成的序列后，根据我们的经验，我们可以预测出下一个字符很大概率是 “t”。我们希望建立这样一个模型，逐个输入一段长为 ``seq_length`` 的序列，输出这些序列接续的下一个字符的概率分布。我们从下一个字符的概率分布中采样作为预测值，然后滚雪球式地生成下两个字符，下三个字符等等，即可完成文本的生成任务。

首先，还是实现一个简单的 ``DataLoader`` 类来读取文本，并以字符为单位进行编码。设字符种类数为 ``num_chars`` ，则每种字符赋予一个 0 到 ``num_chars - 1`` 之间的唯一整数编号 i。

.. literalinclude:: /_static/code/zh/model/text_generation/rnn.py
    :lines: 35-53

接下来进行模型的实现。在 ``__init__`` 方法中我们实例化一个常用的 ``LSTMCell`` 单元，以及一个线性变换用的全连接层，我们首先对序列进行“One Hot”操作，即将序列中的每个字符的编码 i 均变换为一个 ``num_char`` 维向量，其第 i 位为 1，其余均为 0。变换后的序列张量形状为 ``[seq_length, num_chars]`` 。然后，我们初始化 RNN 单元的状态，存入变量 ``state`` 中。接下来，将序列从头到尾依次送入 RNN 单元，即在 t 时刻，将上一个时刻 t-1 的 RNN 单元状态 ``state`` 和序列的第 t 个元素 ``inputs[t, :]`` 送入 RNN 单元，得到当前时刻的输出 ``output`` 和 RNN 单元状态。取 RNN 单元最后一次的输出，通过全连接层变换到 ``num_chars`` 维，即作为模型的输出。

.. figure:: /_static/image/model/rnn_single.jpg
    :width: 50%
    :align: center

    ``output, state = self.cell(inputs[:, t, :], state)`` 图示

.. figure:: /_static/image/model/rnn.jpg
    :width: 100%
    :align: center

    RNN 流程图示

具体实现如下：

.. literalinclude:: /_static/code/zh/model/text_generation/rnn.py
    :lines: 7-25

定义一些模型超参数：

.. literalinclude:: /_static/code/zh/model/text_generation/rnn.py
    :lines: 57-60

训练过程与前节基本一致，在此复述：

- 从 ``DataLoader`` 中随机取一批训练数据；
- 将这批数据送入模型，计算出模型的预测值；
- 将模型预测值与真实值进行比较，计算损失函数（loss）；
- 计算损失函数关于模型变量的导数；
- 使用优化器更新模型参数以最小化损失函数。

.. literalinclude:: /_static/code/zh/model/text_generation/rnn.py
    :lines: 62-73

关于文本生成的过程有一点需要特别注意。之前，我们一直使用 ``tf.argmax()`` 函数，将对应概率最大的值作为预测值。然而对于文本生成而言，这样的预测方式过于绝对，会使得生成的文本失去丰富性。于是，我们使用 ``np.random.choice()`` 函数按照生成的概率分布取样。这样，即使是对应概率较小的字符，也有机会被取样到。同时，我们加入一个 ``temperature`` 参数控制分布的形状，参数值越大则分布越平缓（最大值和最小值的差值越小），生成文本的丰富度越高；参数值越小则分布越陡峭，生成文本的丰富度越低。

.. literalinclude:: /_static/code/zh/model/text_generation/rnn.py
    :lines: 27-32

通过这种方式进行 “滚雪球” 式的连续预测，即可得到生成文本。

.. literalinclude:: /_static/code/zh/model/text_generation/rnn.py
    :lines: 75-83

生成的文本如下::

    diversity 0.200000:
    conserted and conseive to the conterned to it is a self--and seast and the selfes as a seast the expecience and and and the self--and the sered is a the enderself and the sersed and as a the concertion of the series of the self in the self--and the serse and and the seried enes and seast and the sense and the eadure to the self and the present and as a to the self--and the seligious and the enders

    diversity 0.500000:
    can is reast to as a seligut and the complesed
    has fool which the self as it is a the beasing and us immery and seese for entoured underself of the seless and the sired a mears and everyther to out every sone thes and reapres and seralise as a streed liees of the serse to pease the cersess of the selung the elie one of the were as we and man one were perser has persines and conceity of all self-el

    diversity 1.000000:
    entoles by
    their lisevers de weltaale, arh pesylmered, and so jejurted count have foursies as is
    descinty iamo; to semplization refold, we dancey or theicks-welf--atolitious on his
    such which
    here
    oth idey of pire master, ie gerw their endwit in ids, is an trees constenved mase commars is leed mad decemshime to the mor the elige. the fedies (byun their ope wopperfitious--antile and the it as the f

    diversity 1.200000:
    cain, elvotidue, madehoublesily
    inselfy!--ie the rads incults of to prusely le]enfes patuateded:.--a coud--theiritibaior "nrallysengleswout peessparify oonsgoscess teemind thenry ansken suprerial mus, cigitioum: 4reas. whouph: who
    eved
    arn inneves to sya" natorne. hag open reals whicame oderedte,[fingo is
    zisternethta simalfule dereeg hesls lang-lyes thas quiin turjentimy; periaspedey tomm--whach

.. [#rnn_reference] 此处的任务及实现参考了 https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

.. admonition:: 循环神经网络的工作过程

    循环神经网络是一个处理时间序列数据的神经网络结构，也就是说，我们需要在脑海里有一根时间轴，循环神经网络具有初始状态 :math:`s_0` ，在每个时间点 :math:`t` 迭代对当前时间的输入 :math:`x_t` 进行处理，修改自身的状态 :math:`s_t` ，并进行输出 :math:`o_t` 。

    循环神经网络的核心是状态 :math:`s` ，是一个特定维数的向量，类似于神经网络的 “记忆”。在 :math:`t=0` 的初始时刻，:math:`s_0` 被赋予一个初始值（常用的为全 0 向量）。然后，我们用类似于递归的方法来描述循环神经网络的工作过程。即在 :math:`t` 时刻，我们假设 :math:`s_{t-1}` 已经求出，关注如何在此基础上求出 :math:`s_{t}` ：

    - 对输入向量 :math:`x_t` 通过矩阵 :math:`U` 进行线性变换，:math:`U x_t` 与状态 s 具有相同的维度；
    - 对 :math:`s_{t-1}` 通过矩阵 :math:`W` 进行线性变换，:math:`W s_{t-1}` 与状态 s 具有相同的维度；
    - 将上述得到的两个向量相加并通过激活函数，作为当前状态 :math:`s_t` 的值，即 :math:`s_t = f(U x_t + W s_{t-1})`。也就是说，当前状态的值是上一个状态的值和当前输入进行某种信息整合而产生的；
    - 对当前状态 :math:`s_t` 通过矩阵 :math:`V` 进行线性变换，得到当前时刻的输出 :math:`o_t`。

    .. figure:: /_static/image/model/rnn_cell.jpg
        :align: center

        RNN 工作过程图示（来自 http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/）

    我们假设输入向量 :math:`x_t` 、状态 :math:`s` 和输出向量 :math:`o_t` 的维度分别为 :math:`m`、:math:`n`、:math:`p`，则 :math:`U \in \mathbb{R}^{m \times n}`、:math:`W \in \mathbb{R}^{n \times n}`、:math:`V \in \mathbb{R}^{n \times p}`。

    上述为最基础的 RNN 原理介绍。在实际使用时往往使用一些常见的改进型，如LSTM（长短期记忆神经网络，解决了长序列的梯度消失问题，适用于较长的序列）、GRU等。

Deep reinforcement learning (DRL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`强化学习 <https://zh.wikipedia.org/wiki/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0>`_ （Reinforcement learning，RL）强调如何基于环境而行动，以取得最大化的预期利益。结合了深度学习技术后的强化学习更是如虎添翼。这两年广为人知的 AlphaGo 即是深度强化学习的典型应用。

.. admonition:: 基础知识和原理

    - `Demystifying Deep Reinforcement Learning <https://ai.intel.com/demystifying-deep-reinforcement-learning/>`_ （`中文编译 <https://snowkylin.github.io/rl/2017/01/04/Reinforcement-Learning.html>`_）
    - [Mnih2013]_

这里，我们使用深度强化学习玩 CartPole（平衡杆）游戏。简单说，我们需要让模型控制杆的左右运动，以让其一直保持竖直平衡状态。

.. only:: html

    .. figure:: /_static/image/model/cartpole.gif
        :width: 500
        :align: center

        CartPole 游戏

.. only:: latex

    .. figure:: /_static/image/model/cartpole.png
        :width: 500
        :align: center

        CartPole 游戏

我们使用 `OpenAI 推出的 Gym 环境库 <https://gym.openai.com/>`_ 中的 CartPole 游戏环境，具体安装步骤和教程可参考 `官方文档 <https://gym.openai.com/docs/>`_ 和 `这里 <https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-4-gym/>`_ 。Gym 的基本调用方法如下：

.. code-block:: python

    import gym

    env = gym.make('CartPole-v1')       # 实例化一个游戏环境，参数为游戏名称
    state = env.reset()                 # 初始化环境，获得初始状态
    while True:
        env.render()                    # 对当前帧进行渲染，绘图到屏幕
        action = model.predict(state)   # 假设我们有一个训练好的模型，能够通过当前状态预测出这时应该进行的动作
        next_state, reward, done, info = env.step(action)   # 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
        if done:                        # 如果游戏结束则退出循环
            break

那么，我们的任务就是训练出一个模型，能够根据当前的状态预测出应该进行的一个好的动作。粗略地说，一个好的动作应当能够最大化整个游戏过程中获得的奖励之和，这也是强化学习的目标。

以下代码展示了如何使用深度强化学习中的 Deep Q-Learning 方法来训练模型。

.. literalinclude:: /_static/code/zh/model/rl/rl.py

对于不同的任务（或者说环境），我们需要根据任务的特点，设计不同的状态以及采取合适的网络来拟合 Q 函数。例如，如果我们考虑经典的打砖块游戏（Gym 环境库中的  `Breakout-v0 <https://gym.openai.com/envs/Breakout-v0/>`_ ），每一次执行动作（挡板向左、向右或不动），都会返回一个 ``210 * 160 * 3`` 的 RGB 图片，表示当前屏幕画面。为了给打砖块游戏这个任务设计合适的状态表示，我们有以下分析：

* 砖块的颜色信息并不是很重要，画面转换成灰度也不影响操作，因此可以去除状态中的颜色信息（即将图片转为灰度表示）；
* 小球移动的信息很重要，如果只知道单帧画面而不知道小球往哪边运动，即使是人也很难判断挡板应当移动的方向。因此，必须在状态中加入表征小球运动方向的信息。一个简单的方式是将当前帧与前面几帧的画面进行叠加，得到一个 ``210 * 160 * X`` （X 为叠加帧数）的状态表示；
* 每帧的分辨率不需要特别高，只要能大致表征方块、小球和挡板的位置以做出决策即可，因此对于每帧的长宽可做适当压缩。

而考虑到我们需要从图像信息中提取特征，使用 CNN 作为拟合 Q 函数的网络将更为适合。将上面的 ``QNetwork`` 更换为 CNN 网络，即可用于玩一些简单的视频游戏。

Keras Pipeline *
^^^^^^^^^^^^^^^^

..
    https://medium.com/tensorflow/what-are-symbolic-and-imperative-apis-in-tensorflow-2-0-dfccecb01021
    https://www.tensorflow.org/beta/guide/keras/overview
    https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models

以上示例均使用了 Keras 的 Subclassing API 建立模型，即对 ``tf.keras.Model`` 类进行扩展以定义自己的新模型，同时手工编写了训练和评估模型的流程。这种方式灵活度高，且与其他流行的深度学习框架（如 PyTorch、Chainer）共通，是本手册所推荐的方法。不过在很多时候，我们只需要建立一个结构相对简单和典型的神经网络（比如上文中的 MLP 和 CNN），并使用常规的手段进行训练。这时，Keras 也给我们提供了另一套更为简单高效的内置方法来建立、训练和评估模型。

.. _sequential_functional:

Using Keras Sequential/Functional API mode to construct models
--------------------------------------------------------------

最典型和常用的神经网络结构是将一堆层按特定顺序叠加起来，那么，我们是不是只需要提供一个层的列表，就能由 Keras 将它们自动首尾相连，形成模型呢？Keras 的 Sequential API 正是如此。通过向 ``tf.keras.models.Sequential()`` 提供一个层的列表，就能快速地建立一个 ``tf.keras.Model`` 模型并返回：

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 18-23

不过，这种层叠结构并不能表示任意的神经网络结构。为此，Keras 提供了 Functional API，帮助我们建立更为复杂的模型，例如多输入 / 输出或存在参数共享的模型。其使用方法是将层作为可调用的对象并返回张量（这点与之前章节的使用方法一致），并将输入向量和输出向量提供给 ``tf.keras.Model`` 的 ``inputs`` 和 ``outputs`` 参数，示例如下：

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 25-30
..
    https://www.tensorflow.org/alpha/guide/keras/functional

Using ``compile``, ``fit`` and ``evaluate`` method in Keras Model to train and evaluate models
----------------------------------------------------------------------------------------------

当模型建立完成后，通过 ``tf.keras.Model`` 的 ``compile`` 方法配置训练过程：

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 84-88

``tf.keras.Model.compile`` 接受 3 个重要的参数：

 - ``oplimizer`` ：优化器，可从 ``tf.keras.optimizers`` 中选择；
 - ``loss`` ：损失函数，可从 ``tf.keras.losses`` 中选择；
 - ``metrics`` ：评估指标，可从 ``tf.keras.metrics`` 中选择。

接下来，可以使用 ``tf.keras.Model`` 的 ``fit`` 方法训练模型：

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 89

``tf.keras.Model.fit`` 接受 5 个重要的参数：

 - ``x`` ：训练数据；
 - ``y`` ：目标数据（数据标签）；
 - ``epochs`` ：将训练数据迭代多少遍；
 - ``batch_size`` ：批次的大小；
 - ``validation_data`` ：验证数据，可用于在训练过程中监控模型的性能。

Keras 支持使用 ``tf.data.Dataset`` 进行训练，详见 :ref:`tf.data <tfdata>` 。

最后，使用 ``tf.keras.Model.evaluate`` 评估训练效果，提供测试数据及标签即可：

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 90

..
    https://www.tensorflow.org/beta/guide/keras/training_and_evaluation

Custom layers, losses and evaluations*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可能你还会问，如果现有的这些层无法满足我的要求，我需要定义自己的层怎么办？事实上，我们不仅可以继承 ``tf.keras.Model`` 编写自己的模型类，也可以继承 ``tf.keras.layers.Layer`` 编写自己的层。

.. _custom_layer:

Custom layers
-------------------------------------------

自定义层需要继承 ``tf.keras.layers.Layer`` 类，并重写 ``__init__`` 、 ``build`` 和 ``call`` 三个方法，如下所示：

.. code-block:: python

    class MyLayer(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            # 初始化代码

        def build(self, input_shape):     # input_shape 是一个 TensorShape 类型对象，提供输入的形状
            # 在第一次使用该层的时候调用该部分代码，在这里创建变量可以使得变量的形状自适应输入的形状
            # 而不需要使用者额外指定变量形状。
            # 如果已经可以完全确定变量的形状，也可以在__init__部分创建变量
            self.variable_0 = self.add_weight(...)
            self.variable_1 = self.add_weight(...)

        def call(self, inputs):
            # 模型调用的代码（处理输入并返回输出）
            return output

例如，如果我们要自己实现一个 :ref:`本章第一节 <linear>` 中的全连接层（ ``tf.keras.layers.Dense`` ），可以按如下方式编写。此代码在 ``build`` 方法中创建两个变量，并在 ``call`` 方法中使用创建的变量进行运算：

.. literalinclude:: /_static/code/zh/model/custom/linear.py
    :lines: 9-22

在定义模型的时候，我们便可以如同 Keras 中的其他层一样，调用我们自定义的层 ``LinearLayer``：

.. literalinclude:: /_static/code/zh/model/custom/linear.py
    :lines: 25-32

Custom loss functions and evaluation criteria
---------------------------------------------

..
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/losses/Loss

自定义损失函数需要继承 ``tf.keras.losses.Loss`` 类，重写 ``call`` 方法即可，输入真实值 ``y_true`` 和模型预测值 ``y_pred`` ，输出模型预测值和真实值之间通过自定义的损失函数计算出的损失值。下面的示例为均方差损失函数：

.. code-block:: python

    class MeanSquaredError(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            return tf.reduce_mean(tf.square(y_pred - y_true))

..
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/metrics/Metric

自定义评估指标需要继承 ``tf.keras.metrics.Metric`` 类，并重写 ``__init__`` 、 ``update_state`` 和 ``result`` 三个方法。下面的示例对前面用到的 ``SparseCategoricalAccuracy`` 评估指标类做了一个简单的重实现：

.. literalinclude:: /_static/code/zh/model/utils.py
    :lines: 22-34

.. [LeCun1998] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998. http://yann.lecun.com/exdb/mnist/
.. [Graves2013] Graves, Alex. “Generating Sequences With Recurrent Neural Networks.” ArXiv:1308.0850 [Cs], August 4, 2013. http://arxiv.org/abs/1308.0850.
.. [Mnih2013] Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. “Playing Atari with Deep Reinforcement Learning.” ArXiv:1312.5602 [Cs], December 19, 2013. http://arxiv.org/abs/1312.5602.



