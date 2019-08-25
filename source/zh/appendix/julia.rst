TensorFlow in Julia（Ziyang）
==========================================================

TensorFlow.jl 简介
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow.jl 是 Tensorflow 的 Julia 版本, 经由 `malmaud <https://github.com/malmaud/>`_ 对于原版 Tensorflow 的包装实现.

作为一个对于 Tensorflow 的封装, TensorFlow.jl 和 Python 版本的 TensorFlow 具有类似的API, 并支持 GPU 加速.

为什么要使用 julia 进行 Tensorflow 开发
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

虽然 Julia 对于 Tensorflow 本身的运行并没有什么影响, 但是 TensorFlow.jl 确实具有相当的优势.

作为面向数值计算而生的现代语言，Julia 拥有一系列先进的语法特性. 优异的 JIT 能让你高速提取数据, 并处理 Tensorflow 的输出结果. 而得益于Julia的语法设计, 书写表达式也更加灵活自然.

本章我们将基于 TensorFlow.jl 0.12, 向大家简要介绍 Tensorflow 在 Julia 下的使用. 你可以参考最新的 `TensorFlow.jl 文档 <https://malmaud.github.io/TensorFlow.jl/stable/tutorial.html>`_.

在 docker 中快速体验 TensorFlow.jl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在本机已有 docker 环境的情况下, 使用预装 TensorFlow.jl 的 docker image 是非常方便的.

在命令行中执行 ``docker run -it malmaud/julia:tf`` , 然后就可以获得一个已经安装好 TensorFlow.jl 的 Julia REPL 环境. (如果你不想直接打开 Julia, 请在执行 ``docker run -it malmaud/julia:tf /bin/bash`` 来打开一个bash终端. 如需执行您需要的jl代码文件, 可以使用 docker 的目录映射.)

在 julia 包管理器中安装 TensorFlow.jl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在命令行中执行 ``julia`` 进入 Julia REPL 环境, 然后执行以下命令安装 TensorFlow.jl

.. code-block:: julia

    using pkg
    Pkg.add("TensorFlow")


基础使用
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: julia

    using TensorFlow

    # 定义一个 Session
    sess = TensorFlow.Session()

    # 定义一个常量和变量
    x = TensorFlow.constant([1])
    y = TensorFlow.Variable([2])

    # 定义一个计算
    w = x + y

    # 执行计算过程
    run(sess, TensorFlow.global_variables_initializer())
    res = run(sess, w)

    # 输出结果
    println(res)

MNIST数字分类
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

这个例子来自于 `TensorFlow.jl 文档 <https://malmaud.github.io/TensorFlow.jl/stable/tutorial.html>`_, 可以用于对比 python 版本的 API.

.. code-block:: julia

    # 使用自带例子中的 mnist_loader.jl 加载数据
    include(Pkg.dir("TensorFlow", "examples", "mnist_loader.jl"))
    loader = DataLoader()

    # 定义一个 Session
    using TensorFlow
    sess = Session()


    # 构建 softmax 回归模型
    x = placeholder(Float32)
    y_ = placeholder(Float32)
    W = Variable(zeros(Float32, 784, 10))
    b = Variable(zeros(Float32, 10))

    run(sess, global_variables_initializer())

    # 预测类和损失函数
    y = nn.softmax(x*W + b)
    cross_entropy = reduce_mean(-reduce_sum(y_ .* log(y), axis=[2]))

    # 开始训练模型
    train_step = train.minimize(train.GradientDescentOptimizer(.00001), cross_entropy)
    for i in 1:1000
        batch = next_batch(loader, 100)
        run(sess, train_step, Dict(x=>batch[1], y_=>batch[2]))
    end

    # 查看结果并评估模型
    correct_prediction = indmax(y, 2) .== indmax(y_, 2)
    accuracy=reduce_mean(cast(correct_prediction, Float32))
    testx, testy = load_test_set()

    println(run(sess, accuracy, Dict(x=>testx, y_=>testy)))
