TensorFlow in Julia
===================

Introduction to TensorFlow.jl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

虽然 Julia 是一门非常优秀的语言，但是目前 TensorFlow 并不直接支持 Julia 。如果有需要，你可以选择 TensorFlow.jl ，
这是一个由 `malmaud <https://github.com/malmaud/>`_ 封装的第三方 Julia 包。它有和 Python 版本类似的 API ，也能支持 GPU 加速。

Why using Julia to develop TensorFlow?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

先进的语法糖，让你能简明扼要的表述计算过程。而高性能的 JIT ，提供了媲美静态语言的速度（这一点是在数据预处理中非常重要，但也是 Python 难以企及的）。
所以，使用 Julia ，写的快，跑的更快。
（你可以通过 `这个视频 <https://www.youtube.com/watch?v=n2MwJ1guGVQ>`_ 了解更多）

本章我们将基于 TensorFlow.jl 0.12，向大家简要介绍 Tensorflow 在 Julia 下的使用. 你可以参考最新的 `TensorFlow.jl 文档 <https://malmaud.github.io/TensorFlow.jl/stable/tutorial.html>`_.

TensorFlow.jl environment configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow.jl express experience on docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在本机已有 docker 环境的情况下，使用预装 TensorFlow.jl 的 docker image 是非常方便的。

在命令行中执行 ``docker run -it malmaud/julia:tf`` ，然后就可以获得一个已经安装好 TensorFlow.jl 的 Julia REPL 环境。 (如果你不想直接打开 Julia，请在执行 ``docker run -it malmaud/julia:tf /bin/bash`` 来打开一个bash终端. 如需执行您需要的jl代码文件，可以使用 docker 的目录映射)

Installing TensorFlow.jl in Julia package manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在命令行中执行 ``julia`` 进入 Julia REPL 环境，然后执行以下命令安装 TensorFlow.jl

.. code-block:: julia

    using pkg
    Pkg.add("TensorFlow")


Basic usage of TensorFlow.jl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

MNIST digit catagorization
^^^^^^^^^^^^^^^^^^^^^^^^^^

这个例子来自于 `TensorFlow.jl 文档 <https://malmaud.github.io/TensorFlow.jl/stable/tutorial.html>`_ ，可以用于对比 python 版本的 API.

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
