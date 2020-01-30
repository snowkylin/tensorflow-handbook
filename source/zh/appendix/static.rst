图执行模式下的 TensorFlow
======================================

尽管 TensorFlow 2 建议以即时执行模式（Eager Execution）作为主要执行模式，然而，图执行模式（Graph Execution）作为 TensorFlow 2 之前的主要执行模式，依旧对于我们理解 TensorFlow 具有重要意义。尤其是当我们需要使用 :ref:`tf.function <tffunction>` 时，对图执行模式的理解更是不可或缺。以下我们即介绍 TensorFlow 在图执行模式下的基本使用方法。

.. note:: 为了使用图执行模式，建议使用 TensorFlow 1.X 的API进行操作，即使用 ``import tensorflow.compat.v1 as tf`` 导入TensorFlow，并通过 ``tf.disable_eager_execution()`` 禁用默认的即时执行模式。


TensorFlow 1+1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow 的图执行模式是一个符号式的（基于计算图的）计算框架。简而言之，如果你需要进行一系列计算，则需要依次进行如下两步：

- 建立一个“计算图”，这个图描述了如何将输入数据通过一系列计算而得到输出；
- 建立一个会话，并在会话中与计算图进行交互，即向计算图传入计算所需的数据，并从计算图中获取结果。

使用计算图与会话进行基本运算
-------------------------------------------

这里以计算1+1作为Hello World的示例。

.. literalinclude:: /_static/code/zh/basic/graph/1plus1.py      

输出::
    
    2

以上代码与下面基于 ``tf.function`` 的代码等价：

.. literalinclude:: /_static/code/zh/basic/graph/1plus1_tffunc.py  

占位符（Placeholder）与 ``feed_dict`` 
-------------------------------------------

上面这个程序只能计算1+1，以下程序通过 ``tf.placeholder()`` （占位符张量）和 ``sess.run()`` 的 ``feed_dict`` 参数展示了如何使用TensorFlow计算任意两个数的和：

.. literalinclude:: /_static/code/zh/basic/graph/aplusb.py      

运行程序::

    >>> a = 2
    >>> b = 3
    a + b = 5

以上代码与下面基于 ``tf.function`` 的代码等价：

.. literalinclude:: /_static/code/zh/basic/graph/aplusb_tffunc.py   

由以上例子，我们可以看出：

- ``tf.placeholder()`` 相当于 ``tf.function`` 的函数参数；
- ``sess.run()`` 的 ``feed_dict`` 参数相当于给被 ``@tf.function`` 修饰的函数传值。

变量（Variable）
-----------------------------

**变量** （Variable）是一种特殊类型的张量，使用 ``tf.get_variable()`` 建立，与编程语言中的变量很相似。使用变量前需要先初始化，变量内存储的值可以在计算图的计算过程中被修改。以下示例代码展示了如何建立一个变量，将其值初始化为0，并逐次累加1。

.. literalinclude:: /_static/code/zh/basic/graph/variable.py

输出::

    1.0
    2.0
    3.0
    4.0
    5.0

.. hint:: 为了初始化变量，也可以在声明变量时指定初始化器（initializer），并通过 ``tf.global_variables_initializer()`` 一次性初始化所有变量，在实际工程中更常用：

    .. literalinclude:: /_static/code/zh/basic/graph/variable_with_initializer.py

以上代码与下面基于 ``tf.function`` 的代码等价：

.. literalinclude:: /_static/code/zh/basic/graph/variable_tffunc.py  

矩阵及张量计算
-----------------------------

矩阵乃至张量运算是科学计算（包括机器学习）的基本操作。以下程序展示如何计算两个矩阵 :math:`\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}` 和 :math:`\begin{bmatrix} 1 & 1 \\ 1 & 1 \\ 1 & 1 \end{bmatrix}` 的乘积：

.. literalinclude:: /_static/code/zh/basic/graph/AmatmulB.py

输出::

    [[3. 3.]
     [3. 3.]]

占位符和变量也同样可以为向量、矩阵乃至更高维的张量。

基础示例：线性回归
^^^^^^^^^^^^^^^^^^^^^^^^^^^

与 :ref:`第一章的NumPy和即时执行模式 <linear-regression>` 不同，TensorFlow的图执行模式使用 **符号式编程** 来进行数值运算。首先，我们需要将待计算的过程抽象为计算图，将输入、运算和输出都用符号化的节点来表达。然后，我们将数据不断地送入输入节点，让数据沿着计算图进行计算和流动，最终到达我们需要的特定输出节点。

以下代码展示了如何基于TensorFlow的符号式编程方法完成与前节相同的任务。其中， ``tf.placeholder()`` 即可以视为一种“符号化的输入节点”，使用 ``tf.get_variable()`` 定义模型的参数（Variable类型的张量可以使用 ``tf.assign()`` 操作进行赋值），而 ``sess.run(output_node, feed_dict={input_node: data})`` 可以视作将数据送入输入节点，沿着计算图计算并到达输出节点并返回值的过程。

.. literalinclude:: /_static/code/zh/basic/example/tensorflow_manual_grad.py
    :lines: 9-

自动求导机制
-----------------------------

在上面的两个示例中，我们都是手工计算获得损失函数关于各参数的偏导数。但当模型和损失函数都变得十分复杂时（尤其是深度学习模型），这种手动求导的工程量就难以接受了。因此，在图执行模式中，TensorFlow同样提供了 **自动求导机制** 。类似于即时执行模式下的 ``tape.grad(ys, xs)`` ，可以利用TensorFlow的求导操作 ``tf.gradients(ys, xs)`` 求出损失函数 ``loss`` 关于 ``a`` ， ``b`` 的偏导数。由此，我们可以将上节中的两行手工计算导数的代码

.. literalinclude:: /_static/code/zh/basic/example/tensorflow_manual_grad.py
    :lines: 21-23

替换为

.. code-block:: python

    grad_a, grad_b = tf.gradients(loss, [a, b])

计算结果将不会改变。

优化器
-----------------------------

TensorFlow在图执行模式下也附带有多种 **优化器** （optimizer），可以将求导和梯度更新一并完成。我们可以将上节的代码

.. literalinclude:: /_static/code/zh/basic/example/tensorflow_manual_grad.py
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

使用自动求导机制和优化器简化后的代码如下：

.. literalinclude:: /_static/code/zh/basic/example/tensorflow_autograd.py
    :lines: 9-29
