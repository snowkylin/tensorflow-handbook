TensorFlow Under Graph Model
============================

TensorFlow 1+1
^^^^^^^^^^^^^^

TensorFlow本质上是一个符号式的（基于计算图的）计算框架。这里以计算1+1作为Hello World的示例。

.. literalinclude:: /_static/code/zh/basic/graph/1plus1.py      

输出::
    
    2

上面这个程序只能计算1+1，以下程序通过 ``tf.placeholder()`` （占位符张量）和 ``sess.run()`` 的 ``feed_dict=`` 参数展示了如何使用TensorFlow计算任意两个数的和：

.. literalinclude:: /_static/code/zh/basic/graph/aplusb.py      

运行程序::

    >>> a = 2
    >>> b = 3
    a + b = 5

**变量**（Variable）是一种特殊类型的张量，使用 ``tf.get_variable()`` 建立，与编程语言中的变量很相似。使用变量前需要先初始化，变量内存储的值可以在计算图的计算过程中被修改。以下示例如何建立一个变量，将其值初始化为0，并逐次累加1。

.. literalinclude:: /_static/code/zh/basic/graph/variable.py

输出::

    1.0
    2.0
    3.0
    4.0
    5.0

以下代码和上述代码等价，在声明变量时指定初始化器，并通过 ``tf.global_variables_initializer()`` 一次性初始化所有变量，在实际工程中更常用：

.. literalinclude:: /_static/code/zh/basic/graph/variable_with_initializer.py

矩阵乃至张量运算是科学计算（包括机器学习）的基本操作。以下程序展示如何计算两个矩阵 :math:`\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}` 和 :math:`\begin{bmatrix} 1 & 1 \\ 1 & 1 \\ 1 & 1 \end{bmatrix}` 的乘积：

.. literalinclude:: /_static/code/zh/basic/graph/AmatmulB.py

输出::

    [[3. 3.]
     [3. 3.]]

Placeholder（占位符张量）和Variable（变量张量）也同样可以为向量、矩阵乃至更高维的张量。

A basic example: Linear regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

与前面的NumPy和Eager Execution模式不同，TensorFlow的Graph Execution模式使用 **符号式编程** 来进行数值运算。首先，我们需要将待计算的过程抽象为数据流图，将输入、运算和输出都用符号化的节点来表达。然后，我们将数据不断地送入输入节点，让数据沿着数据流图进行计算和流动，最终到达我们需要的特定输出节点。以下代码展示了如何基于TensorFlow的符号式编程方法完成与前节相同的任务。其中， ``tf.placeholder()`` 即可以视为一种“符号化的输入节点”，使用 ``tf.get_variable()`` 定义模型的参数（Variable类型的张量可以使用 ``tf.assign()`` 进行赋值），而 ``sess.run(output_node, feed_dict={input_node: data})`` 可以视作将数据送入输入节点，沿着数据流图计算并到达输出节点并返回值的过程。

.. literalinclude:: /_static/code/zh/basic/example/tensorflow_manual_grad.py
    :lines: 9-

在上面的两个示例中，我们都是手工计算获得损失函数关于各参数的偏导数。但当模型和损失函数都变得十分复杂时（尤其是深度学习模型），这种手动求导的工程量就难以接受了。TensorFlow提供了 **自动求导机制** ，免去了手工计算导数的繁琐。利用TensorFlow的求导函数 ``tf.gradients(ys, xs)`` 求出损失函数loss关于a，b的偏导数。由此，我们可以将上节中的两行手工计算导数的代码

.. literalinclude:: /_static/code/zh/basic/example/tensorflow_manual_grad.py
    :lines: 21-23

替换为

.. code-block:: python

    grad_a, grad_b = tf.gradients(loss, [a, b])

计算结果将不会改变。

甚至不仅于此，TensorFlow附带有多种 **优化器** （optimizer），可以将求导和梯度更新一并完成。我们可以将上节的代码

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

简化后的代码如下：

.. literalinclude:: /_static/code/zh/basic/example/tensorflow_autograd.py
    :lines: 9-29
