TensorFlow Basic
======================

.. 
    https://www.datacamp.com/community/tutorials/tensorflow-tutorial
    
    As the name suggests, TensorFlow is a procedure which flows the tensors. The tensor, resembled as multi-dimensional array, is a generalization of the vector (one dimensional) and matrix (two dimensional), while the flows of tensors are based on Dataflow Graph, also called Computation Graph. A typical TensorFlow program consists of the following parts:

    1. Define a Dataflow Graph (usually called 'model' in deep learning), which consists of large numbers of variables(called 'undetermined parameters');
    2. Repeat following steps:

    1. Convert training data into tensors and input them into Dataflow Graph for calculation (forward propagation);
    #. Evaluate the loss function and find its partial derivatives for each variable (backward propagation);
    #. Use gradient descent or other optimziers to update variables in order to reduce the value of the loss function (i.e. training parameters).
    
	After enough times (and time) for repetition in step 2, the loss function will decrease to a very small value, indicating the competition of the model training.

    Before elaborating a variety of concepts in TensorFlow such as Tensor, Dataflow Graph, Variable, Optimizer and so on, we give an example first in this handbook so as to provide readers an intuitive comprehension.

This chapter introduces basic operations in TensorFlow.

Prerequesites:

* `Basic Python operations <http://www.runoob.com/python3/python3-tutorial.html>`_ (assignment, branch & loop statement, library import)
* `WITH statement in Python <https://www.ibm.com/developerworks/cn/opensource/os-cn-pythonwith/index.html>`_ ;
* `NumPy <https://docs.scipy.org/doc/numpy/user/quickstart.html>`_ 
* `Vector <https://zh.wikipedia.org/wiki/%E5%90%91%E9%87%8F>`_ & `Matrix <https://zh.wikipedia.org/wiki/%E7%9F%A9%E9%98%B5>`_ operations (matrix addition & subtraction, matrix multiplication with vector & matrix, matrix transpose, etc., Quiz: :math:`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = ?`)
* `Derivative of function <http://old.pep.com.cn/gzsx/jszx_1/czsxtbjxzy/qrzptgjzxjc/dzkb/dscl/>`_ , `Derivative of multivarible function <https://zh.wikipedia.org/wiki/%E5%81%8F%E5%AF%BC%E6%95%B0>`_ (Quiz: :math:`f(x, y) = x^2 + xy + y^2, \frac{\partial f}{\partial x} = ?, \frac{\partial f}{\partial y} = ?`)
* `Linear regression <http://old.pep.com.cn/gzsx/jszx_1/czsxtbjxzy/qrzptgjzxjc/dzkb/dscl/>`_
* `Gradient descent <https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95>`_ Find local minimum on function

TensorFlow 1+1
^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow can be simply regarded as a library of scientific calculation (resembled as Numpy in Python). Here we calculate :math:`1+1` and :math:`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}` as our first example.

.. literalinclude:: ../_static/code/en/basic/eager/1plus1.py  

Output::
    
    tf.Tensor(2, shape=(), dtype=int32)
    tf.Tensor(
    [[19 22]
    [43 50]], shape=(2, 2), dtype=int32)

The code above declares four **tensors** named ``a``, ``b``, ``A`` and ``B``. It also invokes two **operations** ``tf.add()`` and ``tf.matmul()`` which respectively do addition and matrix multiplication with tensors. Operation results are immediately stored in tensors ``c`` and ``C``. **Shape** and **dtype** are two major attributes of tensor. Here ``a``, ``b`` and ``c`` are scalars with null shape and int32 dtype, while ``A``, ``B``, ``C`` are 2-by-2 matrices with ``(2, 2)`` shape and int32 dtype.

In machine learning, it's common to differentiate a function. TensorFlow provides powerful **Automatic Differentiation Mechanism** for differentiation. Following codes show how to utilize ``tf.GradientTape()`` to get the slope of :math:`y(x) = x^2` at :math:`x = 3`.

.. literalinclude:: ../_static/code/en/basic/eager/grad.py  
    :lines: 1-8

Output::
    
    [array([9.], dtype=float32), array([6.], dtype=float32)]

Here ``x`` a **variable** initialized to 3, declared by ``tf.get_variable()``. Like common tensors, variables also posses shape and dtype attributes, but require an initialization. We can indicate an initializer in ``tf.get_variable()`` by setting ``Initializer`` parameter. Here we use ``tf.constant_initializer(3.)`` to intialize variable ``x`` to ``3.`` with a float32 dtype. [#f0]_. An important difference between variables and common tensors is that a function can be differentiated by a variable instead of a tensor with the automatic differentiation mechanism by default. Therefore variables are usually used as parameters defined in machine learning models. ``tf.GraidentTape()`` is a recorder of automatic differentiation which records every variables and steps of calculation automatically. In the example above, variable ``x`` and steps of calucation ``y = tf.square(x)`` are recorded automatically, thus the derivative of tensor ``y`` with respect to ``x`` can be acquired through ``y_grad = tape.gradient(y, x)``.

在机器学习中，更加常见的是对多元函数求偏导数，以及对向量或矩阵的求导。这些对于TensorFlow也不在话下。以下代码展示了如何使用 ``tf.GradientTape()`` 计算函数 :math:`L(w, b) = \|Xw + b - y\|^2` 在 :math:`w = (1, 2)^T, b = 1` 时分别对 :math:`w, b` 的偏导数。其中 :math:`X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix},  y = \begin{bmatrix} 1 \\ 2\end{bmatrix}`。

.. literalinclude:: ../_static/code/zh/basic/eager/grad.py  
    :lines: 10-17

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

    .. literalinclude:: ../_static/code/zh/basic/eager/regression.py  

.. _linear-regression:

基础示例：线性回归
^^^^^^^^^^^^^^^^^^^^^^^^^^^

考虑一个实际问题，某城市在2013年-2017年的房价如下表所示：

======  =====  =====  =====  =====  =====
年份    2013   2014   2015   2016   2017
房价    12000  14000  15000  16500  17500
======  =====  =====  =====  =====  =====

现在，我们希望通过对该数据进行线性回归，即使用线性模型 :math:`y = ax + b` 来拟合上述数据，此处 ``a`` 和 ``b`` 是待求的参数。

首先，我们定义数据，进行基本的归一化操作。

.. literalinclude:: ../_static/code/zh/basic/example/numpy.py
    :lines: 1-7

接下来，我们使用梯度下降方法来求线性模型中两个参数 ``a`` 和 ``b`` 的值 [#f1]_。

回顾机器学习的基础知识，对于多元函数 :math:`f(x)` 求局部极小值，`梯度下降 <https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95>`_ 的过程如下：

* 初始化自变量为 :math:`x_0` ， :math:`k=0` 
* 迭代进行下列步骤直到满足收敛条件：

    * 求函数 :math:`f(x)` 关于自变量的梯度 :math:`\nabla f(x_k)` 
    * 更新自变量： :math:`x_{k+1} = x_{k} - \gamma \nabla f(x_k)` 。这里 :math:`\gamma` 是学习率（也就是梯度下降一次迈出的“步子”大小）
    * :math:`k \leftarrow k+1` 

接下来，我们考虑如何使用程序来实现梯度下降方法，求得线性回归的解 :math:`\min_{a, b} L(a, b) = \sum_{i=1}^n(ax_i + b - y_i)^2` 。

NumPy
-----------------------

机器学习模型的实现并不是TensorFlow的专利。事实上，对于简单的模型，即使使用常规的科学计算库或者工具也可以求解。在这里，我们使用NumPy这一通用的科学计算库来实现梯度下降方法。NumPy提供了多维数组支持，可以表示向量、矩阵以及更高维的张量。同时，也提供了大量支持在多维数组上进行操作的函数（比如下面的 ``np.dot()`` 是求内积， ``np.sum()`` 是求和）。在这方面，NumPy和MATLAB比较类似。在以下代码中，我们手工求损失函数关于参数 ``a`` 和 ``b`` 的偏导数 [#f2]_，并使用梯度下降法反复迭代，最终获得 ``a`` 和 ``b`` 的值。

.. literalinclude:: ../_static/code/zh/basic/example/numpy.py
    :lines: 9-

然而，你或许已经可以注意到，使用常规的科学计算库实现机器学习模型有两个痛点：

- 经常需要手工求函数关于参数的偏导数。如果是简单的函数或许还好，但一旦函数的形式变得复杂（尤其是深度学习模型），手工求导的过程将变得非常痛苦，甚至不可行。
- 经常需要手工根据求导的结果更新参数。这里使用了最基础的梯度下降方法，因此参数的更新还较为容易。但如果使用更加复杂的参数更新方法（例如Adam或者Adagrad），这个更新过程的编写同样会非常繁杂。

而TensorFlow等深度学习框架的出现很大程度上解决了这些痛点，为机器学习模型的实现带来了很大的便利。

TensorFlow
--------------------------------------------------------

TensorFlow的 **Eager Execution（动态图）模式** [#f4]_ 与上述NumPy的运行方式十分类似，然而提供了更快速的运算（GPU支持）、自动求导、优化器等一系列对深度学习非常重要的功能。以下展示了如何使用TensorFlow计算线性回归。可以注意到，程序的结构和前述NumPy的实现非常类似。这里，TensorFlow帮助我们做了两件重要的工作：

* 使用 ``tape.gradient(ys, xs)`` 自动计算梯度；
* 使用 ``optimizer.apply_gradients(grads_and_vars)`` 自动更新模型参数。

.. literalinclude:: ../_static/code/zh/basic/example/tensorflow_eager_autograd.py
    :lines: 12-29

在这里，我们使用了前文的方式计算了损失函数关于参数的偏导数。同时，使用 ``tf.train.GradientDescentOptimizer(learning_rate=1e-3)`` 声明了一个梯度下降 **优化器** （Optimizer），其学习率为1e-3。优化器可以帮助我们根据计算出的求导结果更新模型参数，从而最小化某个特定的损失函数，具体使用方式是调用其 ``apply_gradients()`` 方法。

注意到这里，更新模型参数的方法 ``optimizer.apply_gradients()`` 需要提供参数 ``grads_and_vars``，即待更新的变量（如上述代码中的 ``variables`` ）及损失函数关于这些变量的偏导数（如上述代码中的 ``grads`` ）。具体而言，这里需要传入一个Python列表（List），列表中的每个元素是一个（变量的偏导数，变量）对。比如这里是 ``[(grad_w, w), (grad_b, b)]`` 。我们通过 ``grads = tape.gradient(loss, variables)`` 求出tape中记录的 ``loss`` 关于 ``variables = [w, b]`` 中每个变量的偏导数，也就是 ``grads = [grad_w, grad_b]``，再使用Python的 ``zip()`` 函数将 ``grads = [grad_w, grad_b]`` 和 ``vars = [w, b]`` 拼装在一起，就可以组合出所需的参数了。

在实际应用中，我们编写的模型往往比这里一行就能写完的线性模型 ``y_pred = tf.matmul(X, w) + b`` 要复杂得多。所以，我们往往会编写一个模型类，然后在需要调用的时候使用 ``y_pred = model(X)`` 进行调用。关于模型类的编写方式可见 :doc:`下章 <models>`。

..
    本章小结
    ^^^^^^^^^^^^^^^^^^^^^^^


.. [#f0] Python中可以使用整数后加小数点表示将该整数定义为浮点数类型。例如 ``3.`` 代表浮点数 ``3.0``。
.. [#f3] 主要可以参考 `Tensor Transformations <https://www.tensorflow.org/versions/r1.9/api_guides/python/array_ops>`_ 和 `Math <https://www.tensorflow.org/versions/r1.9/api_guides/python/math_ops>`_ 两个页面。可以注意到，TensorFlow的张量操作API在形式上和Python下流行的科学计算库NumPy非常类似，如果对后者有所了解的话可以快速上手。
.. [#f1] 其实线性回归是有解析解的。这里使用梯度下降方法只是为了展示TensorFlow的运作方式。
.. [#f2] 此处的损失函数为均方差 :math:`L(x) = \frac{1}{2} \sum_{i=1}^5 (ax_i + b - y_i)^2`。其关于参数 ``a`` 和 ``b`` 的偏导数为 :math:`\frac{\partial L}{\partial a} = \sum_{i=1}^5 (ax_i + b - y) x_i`，:math:`\frac{\partial L}{\partial b} = \sum_{i=1}^5 (ax_i + b - y)`
.. [#f4] 与Eager Execution相对的是Graph Execution（静态图）模式，即TensorFlow在2018年3月的1.8版本发布之前所主要使用的模式。本手册以面向快速迭代开发的动态模式为主，但会在附录中介绍静态图模式的基本使用，供需要的读者查阅。

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