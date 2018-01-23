TensorFlow基本概念
======================

TensorFlow，顾名思义，就是Tensor（张量）进行Flow（流动）的过程。所谓张量，即对向量（一维）和矩阵（二维）的一种推广，类似于多维数组。而张量的流动则是基于数据流图（Dataflow Graph）。一个典型的TensorFlow程序由以下几个部分组成：

1. 定义一个数据流图（在深度学习中往往称之为“模型”），其中往往包含大量的变量（深度学习中“模型的待训练参数”）；
2. 反复进行以下步骤：

 1. 将训练数据转换为张量，并送入数据流图进行计算（前向传播）；
 #. 计算损失函数的值，并对各变量求偏导数（反向传播）；
 #. 使用梯度下降或其他优化器（Optimizer）对变量进行更新以减小损失函数的值（即“对参数进行训练”）。

在步骤2重复足够多的次数（训练足够长的时间）后，损失函数达到较小的值并保持稳定，即完成了模型的训练。

在对TensorFlow的具体概念，如张量（Tensor）、数据流图（Dataflow Graph）、变量（Variable）、优化器（Optimizer）等进行具体介绍之前，本手册先举一个具体的例子，以让读者能对TensorFlow的基本运作方式有一个直观的理解。

基础示例：线性回归
^^^^^^^^^^^^^^^^^^^^^^^^^^^

考虑一个实际问题，某城市在2013年-2017年的房价如下表所示：

======  =====  =====  =====  =====  =====
年份    2013   2014   2015   2016   2017
房价    12000  14000  15000  16500  17500
======  =====  =====  =====  =====  =====

现在，我们希望通过对该数据进行线性回归，即使用线性模型 :math:`y = ax + b` 来拟合上述数据，此处 :math:`a` 和 :math:`b` 是待求的参数。

首先，我们定义数据，进行基本的归一化操作。
::

    import numpy as np

    X_raw = np.array([2013, 2014, 2015, 2016, 2017])
    y_raw = np.array([12000, 14000, 15000, 16500, 17500])

    X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

接下来，我们使用梯度下降的方法来求两个参数的值 [注]_。

NumPy
-----

在介绍TensorFlow之前，先展示如何使用NumPy这一通用的科学计算库来进行线性回归。NumPy提供了多维数组支持，可以表示向量、矩阵以及更高维的张量。同时，也提供了大量支持在多维数组上进行操作的函数（比如下面的 ``np.dot()`` 是求内积， ``np.sum()`` 是求和）。在这方面，NumPy和MATLAB比较类似。

::
    
    a, b = 0, 0
    num_epoch = 10000
    learning_rate = 1e-3
    for e in range(num_epoch):
        # 前向传播
        y_pred = a * X + b
        loss = 0.5 * (y_pred - y).dot(y_pred - y) # loss = 0.5 * np.sum(np.square(a * X + b - y))

        # 反向传播，手动计算变量（模型参数）的梯度
        grad_a, grad_b = (y_pred - y).dot(X), (y_pred - y).sum()

        # 更新参数
        a, b = a - learning_rate * grad_a, b - learning_rate * grad_b
    print(a, b)

TensorFlow
----------

::

    import tensorflow as tf

    # 定义数据流图
    learning_rate_ = tf.placeholder(dtype=tf.float32)
    X_ = tf.placeholder(dtype=tf.float32, shape=[5])
    y_ = tf.placeholder(dtype=tf.float32, shape=[5])
    a = tf.get_variable('a', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
    b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)

    y_pred = a * X_ + b
    loss = tf.constant(0.5) * tf.reduce_sum(tf.square(y_pred - y_))

    # 反向传播，手动计算变量（模型参数）的梯度
    grad_a = tf.reduce_sum((y_pred - y_) * X_)
    grad_b = tf.reduce_sum(y_pred - y_)
    
    # 手动更新参数
    new_a = a - learning_rate_ * grad_a
    new_b = b - learning_rate_ * grad_b
    update_a = tf.assign(a, new_a)
    update_b = tf.assign(b, new_b)
    # 数据流图定义到此结束
    # 注意，直到目前，我们都没有进行任何实质的数据计算，仅仅是定义了一个数据流图

    num_epoch = 10000
    learning_rate = 1e-3
    with tf.Session() as sess:
        # 初始化变量a和b
        tf.global_variables_initializer().run()
        # 循环将数据送入上面建立的数据流图中进行计算和更新变量
        for e in range(num_epoch):
            sess.run([update_a, update_b], feed_dict={X_: X, y_: y, learning_rate_: learning_rate})
        print(sess.run([a, b]))

自动求导
------------

动态图
---------

.. [注] 其实线性回归是有解析解的。这里使用梯度下降方法只是为了展示TensorFlow的运作方式。

变量、常量与占位符
^^^^^^^^^^^^^^^^^^^^^^^^^^^

会话与计算图
^^^^^^^^^^^^^^^^^^

优化器
^^^^^^^^^

保存和恢复模型
^^^^^^^^^^^^^^^^^^^^^