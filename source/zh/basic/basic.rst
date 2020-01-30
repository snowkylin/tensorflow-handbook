TensorFlow基础
======================

.. 
    https://www.datacamp.com/community/tutorials/tensorflow-tutorial

    TensorFlow，顾名思义，就是Tensor（张量）进行Flow（流动）的过程。所谓张量，即对向量（一维）和矩阵（二维）的一种推广，类似于多维数组。而张量的流动则是基于数据流图（Dataflow Graph，也称计算图Computational Graph）。一个典型的TensorFlow程序由以下几个部分组成：

    1. 定义一个数据流图（在深度学习中往往称之为“模型”），其中往往包含大量的变量（深度学习中“模型的待训练参数”）；
    2. 反复进行以下步骤：

    1. 将训练数据转换为张量，并送入数据流图进行计算（前向传播）；
    #. 计算损失函数的值，并对各变量求偏导数（反向传播）；
    #. 使用梯度下降或其他优化器（Optimizer）对变量进行更新以减小损失函数的值（即“对参数进行训练”）。

    在步骤2重复足够多的次数（训练足够长的时间）后，损失函数达到较小的值并保持稳定，即完成了模型的训练。

    在对TensorFlow的具体概念，如张量（Tensor）、数据流图（Dataflow Graph）、变量（Variable）、优化器（Optimizer）等进行具体介绍之前，本手册先举一个具体的例子，以让读者能对TensorFlow的基本运作方式有一个直观的理解。

    https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

本章介绍TensorFlow的基本操作。

.. admonition:: 前置知识

    * `Python基本操作 <http://www.runoob.com/python3/python3-tutorial.html>`_ （赋值、分支及循环语句、使用import导入库）；
    * `Python的With语句 <https://www.ibm.com/developerworks/cn/opensource/os-cn-pythonwith/index.html>`_ ；
    * `NumPy <https://docs.scipy.org/doc/numpy/user/quickstart.html>`_ ，Python下常用的科学计算库。TensorFlow与之结合紧密；
    * `向量 <https://zh.wikipedia.org/wiki/%E5%90%91%E9%87%8F>`_ 和 `矩阵 <https://zh.wikipedia.org/wiki/%E7%9F%A9%E9%98%B5>`_ 运算（矩阵的加减法、矩阵与向量相乘、矩阵与矩阵相乘、矩阵的转置等。测试题：:math:`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = ?`）；
    * `函数的导数 <http://old.pep.com.cn/gzsx/jszx_1/czsxtbjxzy/qrzptgjzxjc/dzkb/dscl/>`_ ，`多元函数求导 <https://zh.wikipedia.org/wiki/%E5%81%8F%E5%AF%BC%E6%95%B0>`_ （测试题：:math:`f(x, y) = x^2 + xy + y^2, \frac{\partial f}{\partial x} = ?, \frac{\partial f}{\partial y} = ?`）；
    * `线性回归 <http://old.pep.com.cn/gzsx/jszx_1/czsxtbjxzy/qrzptgjzxjc/dzkb/dscl/>`_ ；
    * `梯度下降方法 <https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95>`_ 求函数的局部最小值。

TensorFlow 1+1
^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们可以先简单地将TensorFlow视为一个科学计算库（类似于Python下的NumPy）。

首先，我们导入TensorFlow：

.. code-block:: python

    import tensorflow as tf

.. warning:: 本手册基于TensorFlow的即时执行模式（Eager Execution）。在TensorFlow 1.X版本中， **必须** 在导入TensorFlow库后调用 ``tf.enable_eager_execution()`` 函数以启用即时执行模式。在 TensorFlow 2 中，即时执行模式将成为默认模式，无需额外调用 ``tf.enable_eager_execution()`` 函数（不过若要关闭即时执行模式，则需调用 ``tf.compat.v1.disable_eager_execution()`` 函数）。

TensorFlow使用 **张量** （Tensor）作为数据的基本单位。TensorFlow的张量在概念上等同于多维数组，我们可以使用它来描述数学中的标量（0维数组）、向量（1维数组）、矩阵（2维数组）等各种量，示例如下：

.. literalinclude:: /_static/code/zh/basic/eager/1plus1.py  
    :lines: 3-11

张量的重要属性是其形状、类型和值。可以通过张量的 ``shape`` 、 ``dtype`` 属性和 ``numpy()`` 方法获得。例如：

.. literalinclude:: /_static/code/zh/basic/eager/1plus1.py  
    :lines: 13-17

.. tip:: TensorFlow的大多数API函数会根据输入的值自动推断张量中元素的类型（一般默认为 ``tf.float32`` ）。不过你也可以通过加入 ``dtype`` 参数来自行指定类型，例如 ``zero_vector = tf.zeros(shape=(2), dtype=tf.int32)`` 将使得张量中的元素类型均为整数。张量的 ``numpy()`` 方法是将张量的值转换为一个NumPy数组。

TensorFlow里有大量的 **操作** （Operation），使得我们可以将已有的张量进行运算后得到新的张量。示例如下：

.. literalinclude:: /_static/code/zh/basic/eager/1plus1.py  
    :lines: 19-20

操作完成后， ``C`` 和 ``D`` 的值分别为::
    
    tf.Tensor(
    [[ 6.  8.]
     [10. 12.]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[19. 22.]
     [43. 50.]], shape=(2, 2), dtype=float32)

可见，我们成功使用 ``tf.add()`` 操作计算出 :math:`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}`，使用 ``tf.matmul()`` 操作计算出 :math:`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\43 & 50 \end{bmatrix}` 。

.. _automatic_derivation:

自动求导机制
^^^^^^^^^^^^^^^^^^^^^^^^^^^

在机器学习中，我们经常需要计算函数的导数。TensorFlow提供了强大的 **自动求导机制** 来计算导数。在即时执行模式下，TensorFlow引入了 ``tf.GradientTape()`` 这个“求导记录器”来实现自动求导。以下代码展示了如何使用 ``tf.GradientTape()`` 计算函数 :math:`y(x) = x^2` 在 :math:`x = 3` 时的导数：

.. literalinclude:: /_static/code/zh/basic/eager/grad.py  
    :lines: 1-7

输出::
    
    [array([9.], dtype=float32), array([6.], dtype=float32)]

这里 ``x`` 是一个初始化为3的 **变量** （Variable），使用 ``tf.Variable()`` 声明。与普通张量一样，变量同样具有形状、类型和值三种属性。使用变量需要有一个初始化过程，可以通过在 ``tf.Variable()`` 中指定 ``initial_value`` 参数来指定初始值。这里将变量 ``x`` 初始化为 ``3.`` [#f0]_。变量与普通张量的一个重要区别是其默认能够被TensorFlow的自动求导机制所求导，因此往往被用于定义机器学习模型的参数。

``tf.GradientTape()`` 是一个自动求导的记录器。只要进入了 ``with tf.GradientTape() as tape`` 的上下文环境，则在该环境中计算步骤都会被自动记录。比如在上面的示例中，计算步骤 ``y = tf.square(x)`` 即被自动记录。离开上下文环境后，记录将停止，但记录器 ``tape`` 依然可用，因此可以通过 ``y_grad = tape.gradient(y, x)`` 求张量 ``y`` 对变量 ``x`` 的导数。

在机器学习中，更加常见的是对多元函数求偏导数，以及对向量或矩阵的求导。这些对于TensorFlow也不在话下。以下代码展示了如何使用 ``tf.GradientTape()`` 计算函数 :math:`L(w, b) = \|Xw + b - y\|^2` 在 :math:`w = (1, 2)^T, b = 1` 时分别对 :math:`w, b` 的偏导数。其中 :math:`X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix},  y = \begin{bmatrix} 1 \\ 2\end{bmatrix}`。

.. literalinclude:: /_static/code/zh/basic/eager/grad.py  
    :lines: 9-16

输出::

    [62.5, array([[35.],
       [50.]], dtype=float32), array([15.], dtype=float32)]

这里， ``tf.square()`` 操作代表对输入张量的每一个元素求平方，不改变张量形状。 ``tf.reduce_sum()`` 操作代表对输入张量的所有元素求和，输出一个形状为空的纯量张量（可以通过 ``axis`` 参数来指定求和的维度，不指定则默认对所有元素求和）。TensorFlow中有大量的张量操作API，包括数学运算、张量形状操作（如 ``tf.reshape()``）、切片和连接（如 ``tf.concat()``）等多种类型，可以通过查阅TensorFlow的官方API文档 [#f3]_ 来进一步了解。

从输出可见，TensorFlow帮助我们计算出了

.. math::

    L((1, 2)^T, 1) &= 62.5
    
    \frac{\partial L(w, b)}{\partial w} |_{w = (1, 2)^T, b = 1} &= \begin{bmatrix} 35 \\ 50\end{bmatrix}
    
    \frac{\partial L(w, b)}{\partial b} |_{w = (1, 2)^T, b = 1} &= 15

..
    以上的自动求导机制结合 **优化器** ，可以计算函数的极值。这里以线性回归示例（本质是求 :math:`\min_{w, b} L = (Xw + b - y)^2` ，具体原理见 :ref:`后节 <linear-regression>` ）：

    .. literalinclude:: /_static/code/zh/basic/eager/regression.py  

.. _linear-regression:

基础示例：线性回归
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: 基础知识和原理
    
    * UFLDL教程 `Linear Regression <http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/>`_ 一节。

考虑一个实际问题，某城市在2013年-2017年的房价如下表所示：

======  =====  =====  =====  =====  =====
年份    2013   2014   2015   2016   2017
房价    12000  14000  15000  16500  17500
======  =====  =====  =====  =====  =====

现在，我们希望通过对该数据进行线性回归，即使用线性模型 :math:`y = ax + b` 来拟合上述数据，此处 ``a`` 和 ``b`` 是待求的参数。

首先，我们定义数据，进行基本的归一化操作。

.. literalinclude:: /_static/code/zh/basic/example/numpy_manual_grad.py
    :lines: 1-7

接下来，我们使用梯度下降方法来求线性模型中两个参数 ``a`` 和 ``b`` 的值 [#f1]_。

回顾机器学习的基础知识，对于多元函数 :math:`f(x)` 求局部极小值，`梯度下降 <https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95>`_ 的过程如下：

* 初始化自变量为 :math:`x_0` ， :math:`k=0` 
* 迭代进行下列步骤直到满足收敛条件：

    * 求函数 :math:`f(x)` 关于自变量的梯度 :math:`\nabla f(x_k)` 
    * 更新自变量： :math:`x_{k+1} = x_{k} - \gamma \nabla f(x_k)` 。这里 :math:`\gamma` 是学习率（也就是梯度下降一次迈出的“步子”大小）
    * :math:`k \leftarrow k+1` 

接下来，我们考虑如何使用程序来实现梯度下降方法，求得线性回归的解 :math:`\min_{a, b} L(a, b) = \sum_{i=1}^n(ax_i + b - y_i)^2` 。

NumPy下的线性回归
-------------------------------------------

机器学习模型的实现并不是TensorFlow的专利。事实上，对于简单的模型，即使使用常规的科学计算库或者工具也可以求解。在这里，我们使用NumPy这一通用的科学计算库来实现梯度下降方法。NumPy提供了多维数组支持，可以表示向量、矩阵以及更高维的张量。同时，也提供了大量支持在多维数组上进行操作的函数（比如下面的 ``np.dot()`` 是求内积， ``np.sum()`` 是求和）。在这方面，NumPy和MATLAB比较类似。在以下代码中，我们手工求损失函数关于参数 ``a`` 和 ``b`` 的偏导数 [#f2]_，并使用梯度下降法反复迭代，最终获得 ``a`` 和 ``b`` 的值。

.. literalinclude:: /_static/code/zh/basic/example/numpy_manual_grad.py
    :lines: 9-

然而，你或许已经可以注意到，使用常规的科学计算库实现机器学习模型有两个痛点：

- 经常需要手工求函数关于参数的偏导数。如果是简单的函数或许还好，但一旦函数的形式变得复杂（尤其是深度学习模型），手工求导的过程将变得非常痛苦，甚至不可行。
- 经常需要手工根据求导的结果更新参数。这里使用了最基础的梯度下降方法，因此参数的更新还较为容易。但如果使用更加复杂的参数更新方法（例如Adam或者Adagrad），这个更新过程的编写同样会非常繁杂。

而TensorFlow等深度学习框架的出现很大程度上解决了这些痛点，为机器学习模型的实现带来了很大的便利。

.. _optimizer:

TensorFlow下的线性回归
-------------------------------------------

TensorFlow的 **即时执行模式** [#f4]_ 与上述NumPy的运行方式十分类似，然而提供了更快速的运算（GPU支持）、自动求导、优化器等一系列对深度学习非常重要的功能。以下展示了如何使用TensorFlow计算线性回归。可以注意到，程序的结构和前述NumPy的实现非常类似。这里，TensorFlow帮助我们做了两件重要的工作：

* 使用 ``tape.gradient(ys, xs)`` 自动计算梯度；
* 使用 ``optimizer.apply_gradients(grads_and_vars)`` 自动更新模型参数。

.. literalinclude:: /_static/code/zh/basic/example/tensorflow_eager_autograd.py
    :lines: 10-29

在这里，我们使用了前文的方式计算了损失函数关于参数的偏导数。同时，使用 ``tf.keras.optimizers.SGD(learning_rate=1e-3)`` 声明了一个梯度下降 **优化器** （Optimizer），其学习率为1e-3。优化器可以帮助我们根据计算出的求导结果更新模型参数，从而最小化某个特定的损失函数，具体使用方式是调用其 ``apply_gradients()`` 方法。

注意到这里，更新模型参数的方法 ``optimizer.apply_gradients()`` 需要提供参数 ``grads_and_vars``，即待更新的变量（如上述代码中的 ``variables`` ）及损失函数关于这些变量的偏导数（如上述代码中的 ``grads`` ）。具体而言，这里需要传入一个Python列表（List），列表中的每个元素是一个 ``（变量的偏导数，变量）`` 对。比如上例中需要传入的参数是 ``[(grad_a, a), (grad_b, b)]`` 。我们通过 ``grads = tape.gradient(loss, variables)`` 求出tape中记录的 ``loss`` 关于 ``variables = [a, b]`` 中每个变量的偏导数，也就是 ``grads = [grad_a, grad_b]``，再使用Python的 ``zip()`` 函数将 ``grads = [grad_a, grad_b]`` 和 ``variables = [a, b]`` 拼装在一起，就可以组合出所需的参数了。

.. admonition:: Python的 ``zip()`` 函数

    ``zip()`` 函数是Python的内置函数。用自然语言描述这个函数的功能很绕口，但如果举个例子就很容易理解了：如果 ``a = [1, 3, 5]``， ``b = [2, 4, 6]``，那么 ``zip(a, b) = [(1, 2), (3, 4), ..., (5, 6)]`` 。即“将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表”，和我们日常生活中拉上拉链（zip）的操作有异曲同工之妙。在Python 3中， ``zip()`` 函数返回的是一个 zip 对象，本质上是一个生成器，需要调用 ``list()`` 来将生成器转换成列表。

    .. figure:: /_static/image/basic/zip.jpg
        :width: 60%
        :align: center

        Python的 ``zip()`` 函数图示

在实际应用中，我们编写的模型往往比这里一行就能写完的线性模型 ``y_pred = a * X + b`` （模型参数为 ``variables = [a, b]`` ）要复杂得多。所以，我们往往会编写并实例化一个模型类 ``model = Model()`` ，然后使用 ``y_pred = model(X)`` 调用模型，使用 ``model.variables`` 获取模型参数。关于模型类的编写方式可见 :doc:`"TensorFlow模型"一章 <models>`。

..
    本章小结
    ^^^^^^^^^^^^^^^^^^^^^^^


.. [#f0] Python中可以使用整数后加小数点表示将该整数定义为浮点数类型。例如 ``3.`` 代表浮点数 ``3.0``。
.. [#f3] 主要可以参考 `Tensor Transformations <https://www.tensorflow.org/versions/r1.9/api_guides/python/array_ops>`_ 和 `Math <https://www.tensorflow.org/versions/r1.9/api_guides/python/math_ops>`_ 两个页面。可以注意到，TensorFlow的张量操作API在形式上和Python下流行的科学计算库NumPy非常类似，如果对后者有所了解的话可以快速上手。
.. [#f1] 其实线性回归是有解析解的。这里使用梯度下降方法只是为了展示TensorFlow的运作方式。
.. [#f2] 此处的损失函数为均方差 :math:`L(x) = \frac{1}{2} \sum_{i=1}^5 (ax_i + b - y_i)^2`。其关于参数 ``a`` 和 ``b`` 的偏导数为 :math:`\frac{\partial L}{\partial a} = \sum_{i=1}^5 (ax_i + b - y) x_i`，:math:`\frac{\partial L}{\partial b} = \sum_{i=1}^5 (ax_i + b - y)`
.. [#f4] 与即时执行模式相对的是图执行模式（Graph Execution），即 TensorFlow 2 之前所主要使用的执行模式。本手册以面向快速迭代开发的即时执行模式为主，但会在 :doc:`附录 <../appendix/static>` 中介绍图执行模式的基本使用，供需要的读者查阅。

..  
    张量（变量、常量与占位符）
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    会话与计算图
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    自动求导与优化器
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    变量的范围（Scope）
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ..  https://tensorflow.google.cn/versions/master/api_docs/python/tf/variable_scope

    保存、恢复和持久化
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^