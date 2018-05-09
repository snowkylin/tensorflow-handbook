TensorFlow基本概念
======================

.. https://www.datacamp.com/community/tutorials/tensorflow-tutorial

TensorFlow，顾名思义，就是Tensor（张量）进行Flow（流动）的过程。所谓张量，即对向量（一维）和矩阵（二维）的一种推广，类似于多维数组。而张量的流动则是基于数据流图（Dataflow Graph，也称计算图Computational Graph）。一个典型的TensorFlow程序由以下几个部分组成：

1. 定义一个数据流图（在深度学习中往往称之为“模型”），其中往往包含大量的变量（深度学习中“模型的待训练参数”）；
2. 反复进行以下步骤：

   1. 将训练数据转换为张量，并送入数据流图进行计算（前向传播）；
   #. 计算损失函数的值，并对各变量求偏导数（反向传播）；
   #. 使用梯度下降或其他优化器（Optimizer）对变量进行更新以减小损失函数的值（即“对参数进行训练”）。

在步骤2重复足够多的次数（训练足够长的时间）后，损失函数达到较小的值并保持稳定，即完成了模型的训练。

在对TensorFlow的具体概念，如张量（Tensor）、数据流图（Dataflow Graph）、变量（Variable）、优化器（Optimizer）等进行具体介绍之前，本手册先举一个具体的例子，以让读者能对TensorFlow的基本运作方式有一个直观的理解。

基础示例：TensorFlow 1+1
^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow本质上是一个符号式的（基于计算图的）计算框架。这里以计算1+1作为Hello World的示例。

.. literalinclude:: ../source/_static/code/basic/1plus1.py      

输出::
    
    2

上面这个程序只能计算1+1，以下程序通过 ``tf.placeholder()`` （占位符张量）和 ``sess.run()`` 的 ``feed_dict=`` 参数展示了如何使用TensorFlow计算任意两个数的和：

.. literalinclude:: ../source/_static/code/basic/aplusb.py      

运行程序::

    >>> a = 2
    >>> b = 3
    a + b = 5

变量（Variable）是一种特殊类型的张量，使用 ``tf.get_variable()`` 建立，与编程语言中的变量很相似。使用变量前需要先初始化，变量的值可以在计算图的计算过程中被修改。以下示例如何建立一个变量，将其值初始化为0，并逐次累加1。

.. literalinclude:: ../source/_static/code/basic/variable.py

输出::

    1.0
    2.0
    3.0
    4.0
    5.0

以下代码和上述代码等价，在声明变量时指定初始化器，并通过 ``tf.global_variables_initializer()`` 一次性初始化所有变量，在实际工程中更常用：

.. literalinclude:: ../source/_static/code/basic/variable_with_initializer.py

矩阵乃至张量运算是科学计算（包括机器学习）的基本操作。以下程序展示如何计算两个矩阵 :math:`\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}` 和 :math:`\begin{bmatrix} 1 & 1 \\ 1 & 1 \\ 1 & 1 \end{bmatrix}` 的乘积：

.. literalinclude:: ../source/_static/code/basic/AmatmulB.py

输出::

    [[3. 3.]
     [3. 3.]]

Placeholder（占位符张量）和Variable（变量张量）也同样可以为向量、矩阵乃至更高维的张量。

基础示例：线性回归
^^^^^^^^^^^^^^^^^^^^^^^^^^^

考虑一个实际问题，某城市在2013年-2017年的房价如下表所示：

======  =====  =====  =====  =====  =====
年份    2013   2014   2015   2016   2017
房价    12000  14000  15000  16500  17500
======  =====  =====  =====  =====  =====

现在，我们希望通过对该数据进行线性回归，即使用线性模型 :math:`y = ax + b` 来拟合上述数据，此处 ``a`` 和 ``b`` 是待求的参数。

首先，我们定义数据，进行基本的归一化操作。

.. literalinclude:: ../source/_static/code/basic/example_numpy.py
    :lines: 1-7

接下来，我们使用梯度下降的方法来求两个参数的值 [注1]_。

NumPy：命令式编程
-----------------------

在介绍TensorFlow之前，先展示如何以我们最熟悉的命令式编程，通过代码的逐行执行演算来进行线性回归。在这里，我们使用NumPy这一通用的科学计算库来进行计算。NumPy提供了多维数组支持，可以表示向量、矩阵以及更高维的张量。同时，也提供了大量支持在多维数组上进行操作的函数（比如下面的 ``np.dot()`` 是求内积， ``np.sum()`` 是求和）。在这方面，NumPy和MATLAB比较类似。在以下代码中，我们手工求损失函数关于参数 ``a`` 和 ``b`` 的偏导数 [注2]_，并使用梯度下降法反复迭代，最终获得 ``a`` 和 ``b`` 的值。

.. literalinclude:: ../source/_static/code/basic/example_numpy.py
    :lines: 9-

TensorFlow：符号式编程
----------------------------

TensorFlow使用 **符号式编程** 来进行数值运算。首先，我们需要将待计算的过程抽象为数据流图，将输入、运算和输出都用符号化的节点来表达。然后，我们将数据不断地送入输入节点，让数据沿着数据流图进行计算和流动，最终到达我们需要的特定输出节点。以下代码展示了如何基于TensorFlow的符号式编程方法完成与前节相同的任务。其中， ``tf.placeholder()`` 即可以视为一种“符号化的输入节点”，使用 ``tf.get_variable()`` 定义模型的参数（Variable类型的张量可以使用 ``tf.assign()`` 进行赋值），而 ``sess.run(output_node, feed_dict={input_node: data})`` 可以视作将数据送入输入节点，沿着数据流图计算并到达输出节点并返回值的过程。

.. literalinclude:: ../source/_static/code/basic/example_tensorflow.py
    :lines: 9-

TensorFlow的自动求导机制
-------------------------------

在上面的两个示例中，我们都是手工计算获得损失函数关于各参数的偏导数。但当模型和损失函数都变得十分复杂时（尤其是深度学习模型），这种手动求导的工程量就难以接受了。TensorFlow提供了 **自动求导机制** ，免去了手工计算导数的繁琐。我们可以将上节中的两行手工计算导数的代码

.. literalinclude:: ../source/_static/code/basic/example_tensorflow.py
    :lines: 21-23

替换为

.. literalinclude:: ../source/_static/code/basic/example_tensorflow_autograd.py
    :lines: 23-24

计算结果将不会改变。

甚至不仅于此，TensorFlow附带有多种 **优化器** （optimizer），可以将求导和梯度更新一并完成。我们可以将上节的代码

.. literalinclude:: ../source/_static/code/basic/example_tensorflow.py
    :lines: 21-31

整体替换为

.. code-block:: python

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_)
    grad = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grad)

这里，我们先实例化了一个TensorFlow中的梯度下降优化器 ``tf.train.GradientDescentOptimizer()`` 并设置学习率。然后利用其 ``compute_gradients(loss)`` 方法求出 ``loss`` 对所有变量（参数）的梯度。最后通过 ``apply_gradients(grad)`` 方法，根据前面算出的梯度来梯度下降更新变量（参数）。

以上三行代码等价于下面一行代码：

.. code-block:: python

    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_).minimize(loss)

TensorFlow的动态图支持 *
------------------------------

.. [注1] 其实线性回归是有解析解的。这里使用梯度下降方法只是为了展示TensorFlow的运作方式。
.. [注2] 此处的损失函数为均方差 :math:`L(x) = \frac{1}{2} \sum_{i=1}^5 (ax_i + b - y_i)^2`。其关于参数 ``a`` 和 ``b`` 的偏导数为 :math:`\frac{\partial L}{\partial a} = \sum_{i=1}^5 (ax_i + b - y) x_i`，:math:`\frac{\partial L}{\partial b} = \sum_{i=1}^5 (ax_i + b - y)`

张量（变量、常量与占位符）
^^^^^^^^^^^^^^^^^^^^^^^^^^^

会话与计算图
^^^^^^^^^^^^^^^^^^

自动求导与优化器
^^^^^^^^^^^^^^^^^^^^^

变量的范围（Scope）
^^^^^^^^^^^^^^^^^^^^^^^^^^^
..  https://tensorflow.google.cn/versions/master/api_docs/python/tf/variable_scope

保存、恢复和持久化
^^^^^^^^^^^^^^^^^^^^^^^^^^^