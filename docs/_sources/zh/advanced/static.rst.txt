图执行模式下的 TensorFlow 2
======================================

尽管 TensorFlow 2 建议以即时执行模式（Eager Execution）作为主要执行模式，然而，图执行模式（Graph Execution）作为 TensorFlow 2 之前的主要执行模式，依旧对于我们理解 TensorFlow 具有重要意义。尤其是当我们需要使用 :ref:`tf.function <tffunction>` 时，对图执行模式的理解更是不可或缺。

图执行模式在 TensorFlow 1.X 和 2.X 版本中的 API 不同：

- 在 TensorFlow 1.X 中，图执行模式主要通过“直接构建计算图 + ``tf.Session``” 进行操作；
- 在 TensorFlow 2 中，图执行模式主要通过 ``tf.function`` 进行操作。

在本章，我们将在 :ref:`tf.function：图执行模式 <tffunction>` 一节的基础上，进一步对图执行模式的这两种 API 进行对比说明，以帮助已熟悉 TensorFlow 1.X 的用户过渡到 TensorFlow 2。

.. hint:: TensorFlow 2 依然支持 TensorFlow 1.X 的 API。为了在 TensorFlow 2 中使用 TensorFlow 1.X 的 API ，我们可以使用 ``import tensorflow.compat.v1 as tf`` 导入 TensorFlow，并通过 ``tf.disable_eager_execution()`` 禁用默认的即时执行模式。

TensorFlow 1+1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow 的图执行模式是一个符号式的（基于计算图的）计算框架。简而言之，如果你需要进行一系列计算，则需要依次进行如下两步：

- 建立一个“计算图”，这个图描述了如何将输入数据通过一系列计算而得到输出；
- 建立一个会话，并在会话中与计算图进行交互，即向计算图传入计算所需的数据，并从计算图中获取结果。

使用计算图进行基本运算
-------------------------------------------

这里以计算 1+1 作为 Hello World 的示例。以下代码通过 TensorFlow 1.X 的图执行模式 API 计算 1+1：

.. literalinclude:: /_static/code/zh/basic/graph/1plus1.py      

输出::
    
    2

而在 TensorFlow 2 中，我们将计算图的建立步骤封装在一个函数中，并使用 ``@tf.function`` 修饰符对函数进行修饰。当需要运行此计算图时，只需调用修饰后的函数即可。由此，我们可以将以上代码改写如下：

.. literalinclude:: /_static/code/zh/basic/graph/1plus1_tffunc.py

.. admonition:: 小结

    - 在 TensorFlow 1.X 的 API 中，我们直接在主程序中建立计算图。而在 TensorFlow 2 中，计算图的建立需要被封装在一个被 ``@tf.function`` 修饰的函数中；
    - 在 TensorFlow 1.X 的 API 中，我们通过实例化一个 ``tf.Session`` ，并使用其 ``run`` 方法执行计算图的实际运算。而在 TensorFlow 2 中，我们通过直接调用被 ``@tf.function`` 修饰的函数来执行实际运算。 

计算图中的占位符与数据输入
-------------------------------------------

上面这个程序只能计算1+1，以下代码通过 TensorFlow 1.X 的图执行模式 API 中的 ``tf.placeholder()`` （占位符张量）和 ``sess.run()`` 的 ``feed_dict`` 参数，展示了如何使用TensorFlow计算任意两个数的和：

.. literalinclude:: /_static/code/zh/basic/graph/aplusb.py      

运行程序::

    >>> a = 2
    >>> b = 3
    a + b = 5

而在 TensorFlow 2 中，我们可以通过为函数指定参数来实现与占位符张量相同的功能。为了在计算图运行时送入占位符数据，只需在调用被修饰后的函数时，将数据作为参数传入即可。由此，我们可以将以上代码改写如下：

.. literalinclude:: /_static/code/zh/basic/graph/aplusb_tffunc.py   

.. admonition:: 小结

    在 TensorFlow 1.X 的 API 中，我们使用 ``tf.placeholder()`` 在计算图中声明占位符张量，并通过 ``sess.run()`` 的 ``feed_dict`` 参数向计算图中的占位符传入实际数据。而在 TensorFlow 2 中，我们使用 ``tf.function`` 的函数参数作为占位符张量，通过向被 ``@tf.function`` 修饰的函数传递参数，来为计算图中的占位符张量提供实际数据。

计算图中的变量
-----------------------------

变量的声明
+++++++++++++++++++++++++++++

**变量** （Variable）是一种特殊类型的张量，在 TensorFlow 1.X 的图执行模式 API 中使用 ``tf.get_variable()`` 建立。与编程语言中的变量很相似。使用变量前需要先初始化，变量内存储的值可以在计算图的计算过程中被修改。以下示例代码展示了如何建立一个变量，将其值初始化为0，并逐次累加1。

.. literalinclude:: /_static/code/zh/basic/graph/variable.py

输出::

    1.0
    2.0
    3.0
    4.0
    5.0

.. hint:: 为了初始化变量，也可以在声明变量时指定初始化器（initializer），并通过 ``tf.global_variables_initializer()`` 一次性初始化所有变量，在实际工程中更常用：

    .. literalinclude:: /_static/code/zh/basic/graph/variable_with_initializer.py

在 TensorFlow 2 中，我们通过实例化 ``tf.Variable`` 类来声明变量。由此，我们可以将以上代码改写如下：

.. literalinclude:: /_static/code/zh/basic/graph/variable_tffunc.py

.. admonition:: 小结

    在 TensorFlow 1.X 的 API 中，我们使用 ``tf.get_variable()`` 在计算图中声明变量节点。而在 TensorFlow 2 中，我们直接通过 ``tf.Variable`` 实例化变量对象，并在计算图中使用这一变量对象。

变量的作用域与重用
+++++++++++++++++++++++++++++

在 TensorFlow 1.X 中，我们建立模型时经常需要指定变量的作用域，以及复用变量。此时，TensorFlow 1.X 的图执行模式 API 为我们提供了 ``tf.variable_scope()`` 及 ``reuse`` 参数来实现变量作用域和复用变量的功能。以下的例子使用了 TensorFlow 1.X 的图执行模式 API 建立了一个三层的全连接神经网络，其中第三层复用了第二层的变量。

.. literalinclude:: /_static/code/zh/basic/graph/variable_scope.py

在上例中，计算图的所有变量节点为：

::

    [<tf.Variable 'dense1/weight:0' shape=(32, 10) dtype=float32>, 
     <tf.Variable 'dense1/bias:0' shape=(10,) dtype=float32>, 
     <tf.Variable 'dense2/weight:0' shape=(10, 10) dtype=float32>, 
     <tf.Variable 'dense2/bias:0' shape=(10,) dtype=float32>]

可见， ``tf.variable_scope()`` 为在其上下文中的，以 ``tf.get_variable`` 建立的变量的名称添加了“前缀”或“作用域”，使得变量在计算图中的层次结构更为清晰，不同“作用域”下的同名变量各司其职，不会冲突。同时，虽然我们在上例中调用了3次 ``dense`` 函数，即调用了6次 ``tf.get_variable`` 函数，但实际建立的变量节点只有4个。这即是 ``tf.variable_scope()`` 的 ``reuse`` 参数所起到的作用。当 ``reuse=True`` 时， ``tf.get_variable`` 遇到重名变量时将会自动获取先前建立的同名变量，而不会新建变量，从而达到了变量重用的目的。

而在 TensorFlow 2 的图执行模式 API 中，不再鼓励使用 ``tf.variable_scope()`` ，而应当使用 ``tf.keras.layers.Layer`` 和  ``tf.keras.Model`` 来封装代码和指定作用域，具体可参考 :doc:`本手册第三章 <../basic/models>`。上面的例子与下面基于 ``tf.keras`` 和 ``tf.function`` 的代码等价。

.. literalinclude:: /_static/code/zh/basic/graph/variable_scope_tffunc.py
    :lines: 1-31

我们可以注意到，在 TensorFlow 2 中，变量的作用域以及复用变量的问题自然地淡化了。基于Python类的模型建立方式自然地为变量指定了作用域，而变量的重用也可以通过简单地多次调用同一个层来实现。

为了详细了解上面的代码对变量作用域的处理方式，我们使用 ``get_concrete_function`` 导出计算图，并输出计算图中的所有变量节点：

.. code-block:: python

    graph = model.call.get_concrete_function(np.random.rand(10, 32))
    print(graph.variables)

输出如下：

::

    (<tf.Variable 'dense1/weight:0' shape=(32, 10) dtype=float32, numpy=...>,
     <tf.Variable 'dense1/bias:0' shape=(10,) dtype=float32, numpy=...>,
     <tf.Variable 'dense2/weight:0' shape=(32, 10) dtype=float32, numpy=...>,
     <tf.Variable 'dense2/bias:0' shape=(10,) dtype=float32, numpy=...)

可见，TensorFlow 2 的图执行模式在变量的作用域上与 TensorFlow 1.X 实际保持了一致。我们通过 ``name`` 参数为每个层指定的名称将成为层内变量的作用域。

.. admonition:: 小结

    在 TensorFlow 1.X 的 API 中，使用 ``tf.variable_scope()`` 及 ``reuse`` 参数来实现变量作用域和复用变量的功能。在 TensorFlow 2 中，使用 ``tf.keras.layers.Layer`` 和  ``tf.keras.Model`` 来封装代码和指定作用域，从而使变量的作用域以及复用变量的问题自然淡化。两者的实质是一样的。

自动求导机制与优化器
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在本节中，我们对 TensorFlow 1.X 和 TensorFlow 2 在图执行模式下的自动求导机制进行较深入的比较说明。

自动求导机制
-------------------------------------

我们首先回顾 TensorFlow 1.X 中的自动求导机制。在 TensorFlow 1.X 的图执行模式 API 中，可以使用 ``tf.gradients(y, x)`` 计算计算图中的张量节点 ``y`` 相对于变量 ``x`` 的导数。以下示例展示了在 TensorFlow 1.X 的图执行模式 API 中计算 :math:`y = x^2` 在 :math:`x = 3` 时的导数。 

.. literalinclude:: /_static/code/zh/basic/graph/grad.py
    :lines: 4-6

以上代码中，计算图中的节点 ``y_grad`` 即为 ``y`` 相对于 ``x`` 的导数。

而在 TensorFlow 2 的图执行模式 API 中，我们使用 ``tf.GradientTape`` 这一上下文管理器封装需要求导的计算步骤，并使用其 ``gradient`` 方法求导，代码示例如下：

.. literalinclude:: /_static/code/zh/tools/tensorboard/grad_v2.py
    :lines: 7-13

.. admonition:: 小结

    在 TensorFlow 1.X 中，我们使用 ``tf.gradients()`` 求导。而在 TensorFlow 2 中，我们使用使用 ``tf.GradientTape`` 这一上下文管理器封装需要求导的计算步骤，并使用其 ``gradient`` 方法求导。

优化器
-------------------------------------

由于机器学习中的求导往往伴随着优化，所以 TensorFlow 中更常用的是优化器（Optimizer）。在 TensorFlow 1.X 的图执行模式 API 中，我们往往使用 ``tf.train`` 中的各种优化器，将求导和调整变量值的步骤合二为一。例如，以下代码片段在计算图构建过程中，使用 ``tf.train.GradientDescentOptimizer`` 这一梯度下降优化器优化损失函数 ``loss`` ：

.. code-block:: python

    y_pred = model(data_placeholder)    # 模型构建
    loss = ...                          # 计算模型的损失函数 loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_one_step = optimizer.minimize(loss)
    # 上面一步也可拆分为
    # grad = optimizer.compute_gradients(loss)
    # train_one_step = optimizer.apply_gradients(grad)

以上代码中， ``train_one_step`` 即为一个将求导和变量值更新合二为一的计算图节点（操作），也就是训练过程中的“一步”。特别需要注意的是，对于优化器的 ``minimize`` 方法而言，只需要指定待优化的损失函数张量节点 ``loss`` 即可，求导的变量可以自动从计算图中获得（即 ``tf.trainable_variables`` ）。在计算图构建完成后，只需启动会话，使用 ``sess.run`` 方法运行 ``train_one_step`` 这一计算图节点，并通过 ``feed_dict`` 参数送入训练数据，即可完成一步训练。代码片段如下：

.. code-block:: python

    for data in dataset:
        data_dict = ... # 将训练所需数据放入字典 data 内
        sess.run(train_one_step, feed_dict=data_dict)

而在 TensorFlow 2 的 API 中，无论是图执行模式还是即时执行模式，均先使用 ``tf.GradientTape`` 进行求导操作，然后再使用优化器的 ``apply_gradients`` 方法应用已求得的导数，进行变量值的更新。也就是说，和 TensorFlow 1.X 中优化器的 ``compute_gradients`` + ``apply_gradients`` 十分类似。同时，在 TensorFlow 2 中，无论是求导还是使用导数更新变量值，都需要显式地指定变量。计算图的构建代码结构如下：

.. code-block:: python

    optimizer = tf.keras.optimizer.SGD(learning_rate=...)
    
    @tf.function
    def train_one_step(data):        
        with tf.GradientTape() as tape:
            y_pred = model(data)    # 模型构建
            loss = ...              # 计算模型的损失函数 loss
        grad = tape.gradient(loss, model.variables)  
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))  
        
在计算图构建完成后，我们直接调用 ``train_one_step`` 函数并送入训练数据即可：

.. code-block:: python

    for data in dataset:
        train_one_step(data)

.. admonition:: 小结

    在 TensorFlow 1.X 中，我们多使用优化器的 ``minimize`` 方法，将求导和变量值更新合二为一。而在 TensorFlow 2 中，我们需要先使用 ``tf.GradientTape`` 进行求导操作，然后再使用优化器的 ``apply_gradients`` 方法应用已求得的导数，进行变量值的更新。而且在这两步中，都需要显式指定待求导和待更新的变量。

.. _graph_compare:

自动求导机制的计算图对比 *
-------------------------------------

在本节，为了帮助读者更深刻地理解 TensorFlow 的自动求导机制，我们以前节的“计算 :math:`y = x^2` 在 :math:`x = 3` 时的导数”为例，展示 TensorFlow 1.X 和 TensorFlow 2 在图执行模式下，为这一求导过程所建立的计算图，并进行详细讲解。

在 TensorFlow 1.X 的图执行模式 API 中，将生成的计算图使用 TensorBoard 进行展示：

.. figure:: /_static/image/graph/grad_v1.png
    :width: 60%
    :align: center

在计算图中，灰色的块为节点的命名空间（Namespace，后文简称“块”），椭圆形代表操作节点（OpNode），圆形代表常量，灰色的箭头代表数据流。为了弄清计算图节点 ``x`` 、 ``y`` 和 ``y_grad`` 与计算图中节点的对应关系，我们将这些变量节点输出，可见：

- ``x`` : ``<tf.Variable 'x:0' shape=() dtype=float32>`` 
- ``y`` : ``Tensor("Square:0", shape=(), dtype=float32)`` 
- ``y_grad`` : ``[<tf.Tensor 'gradients/Square_grad/Mul_1:0' shape=() dtype=float32>]`` 

在 TensorBoard 中，我们也可以通过点击节点获得节点名称。通过比较我们可以得知，变量 ``x`` 对应计算图最下方的x，节点 ``y`` 对应计算图“Square”块的“ ``(Square)`` ”，节点 ``y_grad`` 对应计算图上方“Square_grad”的 ``Mul_1`` 节点。同时我们还可以通过点击节点发现，“Square_grad”块里的const节点值为2，“gradients”块里的 ``grad_ys_0`` 值为1， ``Shape`` 值为空，以及“x”块的const节点值为3。

接下来，我们开始具体分析这个计算图的结构。我们可以注意到，这个计算图的结构是比较清晰的，“x”块负责变量的读取和初始化，“Square”块负责求平方 ``y = x ^ 2`` ，而“gradients”块则负责对“Square”块的操作求导，即计算 ``y_grad = 2 * x``。由此我们可以看出， ``tf.gradients`` 是一个相对比较“庞大”的操作，并非如一般的操作一样往计算图中添加了一个或几个节点，而是建立了一个庞大的子图，以应用链式法则求计算图中特定节点的导数。

在 TensorFlow 2 的图执行模式 API 中，将生成的计算图使用 TensorBoard 进行展示：

.. figure:: /_static/image/graph/grad_v2.png
    :width: 60%
    :align: center

我们可以注意到，除了求导过程没有封装在“gradients”块内，以及变量的处理简化以外，其他的区别并不大。由此，我们可以看出，在图执行模式下， ``tf.GradientTape`` 这一上下文管理器的 ``gradient`` 方法和 TensorFlow 1.X 的 ``tf.gradients`` 是基本等价的。

.. admonition:: 小结

    TensorFlow 1.X 中的 ``tf.gradients`` 和 TensorFlow 2 图执行模式下的  ``tf.GradientTape`` 上下文管理器尽管在 API 层面的调用方法略有不同，但最终生成的计算图是基本一致的。

基础示例：线性回归
^^^^^^^^^^^^^^^^^^^^^^^^^^^

在本节，我们为 :ref:`第一章的线性回归示例 <linear-regression>` 提供一个基于 TensorFlow 1.X 的图执行模式 API 的版本，供有需要的读者对比参考。

与第一章的NumPy和即时执行模式不同，TensorFlow的图执行模式使用 **符号式编程** 来进行数值运算。首先，我们需要将待计算的过程抽象为计算图，将输入、运算和输出都用符号化的节点来表达。然后，我们将数据不断地送入输入节点，让数据沿着计算图进行计算和流动，最终到达我们需要的特定输出节点。

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
