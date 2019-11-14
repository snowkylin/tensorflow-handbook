Swift for TensorFlow (S4TF) (Huan）
==========================================================

.. figure:: /_static/image/swift/swift-tensorflow.png
    :width: 60%
    :align: center
     
    “Swift for TensorFlow is an attempt to change the default tools used by the entire machine learning and data science ecosystem.”
     
     -- Jameson Toole,  Co-founder & CTO of Fritz.ai

S4TF 简介
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Google 推出的 Swift for TensorFlow （简称S4TF）是专门针对 TensorFlow 优化过的 Swift 版本。（目前处在 Pre-Alpha 阶段）

为了能够在程序语言级支持 TensorFlow 所需的所有功能特性，S4TF 做为了 Swift 语言本身的一个分支，为 Swift 语言添加了机器学习所需要的所有功能扩展。它不仅仅是一个用 Swift 写成的 TensorFlow API 封装，Google 还为 Swift 增加了编译器和语言增强功能，提供了一种新的编程模型，结合了图的性能、Eager Execution 的灵活性和表达能力。

.. admonition:: Swift 语言创始人 Chris Lattner

    Swift 语言是 Chris Lattner 在苹果公司工作时创建的。 现在 Chris Lattner 在 Google Brain 工作，专门从事深度学习的研究，并为 Swift 重写了编译器，为 TensorFlow 做定制优化。

本章我们将向大家简要介绍 Swift for TensorFlow 的使用。你可以参考最新的 `Swift for TensorFlow 文档 <https://www.tensorflow.org/swift>`_.

为什么要使用 Swift 进行 TensorFlow 开发
---------------------------------------------

相对于 TensorFlow 的其他版本（如 Python，C++ 等），S4TF 拥有其独有的优势，比如：

#. 开发效率高：强类型语言，能够静态检查变量类型
#. 迁移成本低：与 Python，C，C++ 能够无缝结合
#. 执行性能高：能够直接编译为底层硬件代码
#. 专门为机器学习打造：语言原生支持自动微分系统

与其他语言相比，S4TF 还有更多优势。谷歌正在大力投资，使 Swift 成为其 TensorFlow ML 基础设施的一个关键组件，而且很有可能 Swift 将成为深度学习的专属语言。

.. admonition:: 更多使用 Swift 的理由

    有兴趣的读者可以参考官方文档：`Why Swift for TensorFlow <https://github.com/tensorflow/swift/blob/master/docs/WhySwiftForTensorFlow.md>`_

S4TF 环境配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本地安装 Swift for TensorFlow
---------------------------------------------------------------

目前 S4TF 支持 Mac 和 Linux 两个运行环境。安装需要下载预先编译好的软件包，同时按照对应的操作系统的说明进行操作。安装后，即可以使用全套 Swift 工具，包括 Swift（Swift REPL / Interpreter）和 Swiftc（Swift编译器）。官方文档（含下载地址）可见 `这里 <https://github.com/tensorflow/swift/blob/master/Installation.md>`_ 。

在 Colaboratory 中快速体验 Swift for TensorFlow
---------------------------------------------------------------

Google 的 Colaboratory 可以直接支持 Swift 语言的运行环境。可以 `点此 <https://colab.research.google.com/github/huan/tensorflow-handbook-swift/blob/master/tensorflow-handbook-swift-blank.ipynb>`_ 直接打开一个空白的，具备 Swift 运行环境的 Colab Notebook ，这是立即体验 Swift for TensorFlow 的最方便的办法。

在 Docker 中快速体验 Swift for TensorFlow
---------------------------------------------------------------

在本机已有 docker 环境的情况下, 使用预装 Swift for TensorFlow 的 Docker Image 是非常方便的。

- 获得一个 S4TS 的 Jupyter Notebook

    在命令行中执行 ``nvidia-docker run -ti --rm -p 8888:8888 --cap-add SYS_PTRACE -v "$(pwd)":/notebooks zixia/swift`` 来启动 Jupyter ，然后根据提示的 URL ，打开浏览器访问即可。

- 获得一个已经安装好 S4TF 的 Swift REPL 环境
    
    在命令行中执行 ``docker run -it --privileged --userns=host zixia/swift swift``

.. admonition:: 使用 Docker 执行 Swift 代码文件

    通过使用 Docker 的目录映射，可以启动 Docker 之后执行本地代码文件。详细使用方法可以参考 Docker Image ``zixia/swift`` 开源项目的地址：https://github.com/huan/docker-swift-tensorflow

S4TF 基础使用
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Swift 是动态强类型语言，也就是说 Swift 支持通过编译器自动检测类型，同时要求变量的使用要严格符合定义，所有变量都必须先定义后使用。

下面的代码，因为最初声明的 ``n`` 是整数类型 ``42`` ，所以如果将 ``'string'`` 赋值给 ``n`` 时，会出现类型不匹配的问题，Swift 将会报错：

.. code-block:: swift

    var n = 42
    n = "string"

报错输出：

::

    Cannot assign value of type 'String' to type 'Int'

下面是一个使用 TensorFlow 计算的基础示例：

.. code-block:: swift

    import TensorFlow

    // 声明两个Tensor
    let x = Tensor<Float>([1])
    let y = Tensor<Float>([2])

    // 对两个 Tensor 做加法运算
    let w = x + y

    // 输出结果
    print(w)

.. admonition::  ``Tensor<Float>`` 中的 ``<Float>``

    在这里的 ``Float`` 是用来指定 Tensor 这个类所相关的内部数据类型。可以根据需要替换为其他合理数据类型，比如 “Double”。

在 Swift 中使用标准的 TensorFlow API
---------------------------------------------

在 ``import TensorFlow`` 之后，既可以在 Swift 语言中，使用核心的 TensorFlow API。

1. 处理数字和矩阵的代码，API 与 TensorFlow 高度保持了一致：

.. code-block:: swift

    let x = Tensor<BFloat16>(zeros: [32, 128])
    let h1 = sigmoid(matmul(x, w1) + b1)
    let h2 = tanh(matmul(h1, w1) + b1)
    let h3 = softmax(matmul(h2, w1) + b1)

2. 处理 Dataset 的代码，基本上将 Python API 中的 ``tf.data.Dataset`` 同名函数直接改写为 Swift 语法即可直接使用：

.. code-block:: swift

    let imageBatch = Dataset(elements: images)
    let labelBatch = Dataset(elements: labels)
    let zipped = zip(imageBatch, labelBatch).batched(8)

    let imageBatch = Dataset(elements: images)
    let labelBatch = Dataset(elements: labels)
    for (image, label) in zip(imageBatch, labelBatch) {
        let y = matmul(image, w) + b
        let loss = (y - label).squared().mean()
        print(loss)
    }

.. admonition:: ``matmul()`` 的别名： ``•``

    为了代码更加简洁，``matmul(a, b)`` 可以简写为 ``a • b``。``•`` 符号在 Mac 上，可以通过键盘按键 `Option + 8` 输入。

在 Swift 中直接加载 Python 语言库
---------------------------------------------

Swift 语言支持直接加载 Python 函数库（比如 NumPy），也支持直接加载系统动态链接库，很方便的做到即导入即用。

借助 S4TF 强大的集成能力，从 Python 迁移到 Swift 非常简单。您可以逐步迁移 Python 代码（或继续使用 Python 代码库），因为 S4TF 支持直接在代码中加载 Python 原生代码库，使得开发者可以继续使用熟悉的语法在 Swift 中调用 Python 中已经完成的功能。

下面我们以 NumPy 为例，看一下如何在 Swift 语言中，直接加载 Python 的 NumPy 代码库，并且直接进行调用：

.. code-block:: swift

    import Python

    let np = Python.import("numpy")
    let x = np.array([[1, 2], [3, 4]])
    let y = np.array([11, 12])
    print(x.dot(y))

输出：

::

    [35 81]

除了能够直接调用 Python 之外，Swift 也快成直接调用系统函数库。比如下面的代码例子展示了我们可以在 Swift 中直接加载 Glibc 的动态库，然后调用系统底层的 malloc 和 memcpy 函数，对变量直接进行操作。

.. code-block:: swift

    import Glibc
    let x = malloc(18)
    memcpy(x, "memcpy from Glibc", 18)
    free(x)

通过 Swift 强大的集成能力，针对 C/C++ 语言库的加载和调用，处理起来也将会是非常简单高效。

语言原生支持自动微分
---------------------------------------------

我们可以通过 ``@differentiable`` 参数，非常容易地定义一个可被微分的函数：

.. code-block:: swift

    @differentiable
    func frac(x: Double) -> Double {
        return 1/x
    }

    gradient(of: frac)(0.5)

输出：

::

    -4.0

在上面的代码例子中，我们通过将函数 ``frac()`` 标记为 ``@differentiable`` ，然后就可以通过 ``gradient()`` 函数，将其转换为求解微分的新函数 ``gradient(of: trac)``，接下来就可以根据任意 x 值求解函数 frac 所在 x 点的梯度了。

.. admonition:: Swift 函数声明中的参数名称和类型

    Swift 使用 ``func`` 声明一个函数。在函数的参数中，变量名的冒号后面代表的是“参数类型”；在函数参数和函数体（``{}``） 之前，还可以通过瘦箭头（``->``）来指定函数的``返回值类型``。

    比如在上面的代码中，参数变量名为 “x”；参数类型为 “Double”；函数返回类型为 “Double”。

MNIST数字分类
---------------------------------------------

下面我们以最简单的 MNIST 数字分类为例子，给大家介绍一下基础的 S4TF 编程代码实现。

1. 首先，引入S4TF模块 ``TensorFlow``、Python桥接模块 ``Python``，基础模块 ``Foundation`` 和 MNIST 数据集模块 ``MNIST``：

.. literalinclude:: /_static/code/zh/appendix/swift/mnist.swift
    :lines: 1-5

.. admonition:: Swift MNIST Dataset 模块

    Swift MNIST Dataset 模块是一个简单易用的 MNIST 数据集加载模块，基于 Swift 语言，提供了完整的数据集加载 API。项目 Github：https://github.com/huan/swift-MNIST

2. 其次，声明一个最简单的 MLP 神经网络架构，将输入的 784 个图像数据，转换为 10 个神经元的输出：

.. admonition:: 使用 ``Layer`` 协议定义神经网络模型

    为了定义一个 Swift 神经网络模型，我们需要建立一个遵循 ``Layer`` 协议，来声明一个定义神经网络结构的 ``Struct``。
    
    其中，最为核心的部分是声明 ``callAsFunction(_:)`` 方法，来定义输入和输出 Tensor 的映射关系。
    
    ``callAsFunction(_:)`` 中可以通过类似 Keras 的 Sequential 的方法进行定义：``input.sequences(through: layer1, layer2, ...)`` 将输入和所有的后续处理层 ``layer1``, ``layer2``, ... 等衔接起来。

import TensorFlow

.. literalinclude:: /_static/code/zh/appendix/swift/mnist.swift
    :lines: 7-22

.. admonition:: Swift 参数标签

    在代码中，我们会看到形如 ``callAsFunction(_ input: Input)`` 这样的函数声明。其中，``_`` 代表忽略参数标签。

    Swift 中，每个函数参数都有一个 `参数标签` (Argument Label) 以及一个 `参数名称` (Parameter Name)。 `参数标签` 主要应用在调用函数的情况，使得函数的实参与真实命名相关联，更加容易理解实参的意义。同时因为有 `参数标签` 的存在，实在的顺序是可以随意改变的。
    
    如果你不希望为参数添加标签，可以使用一个下划线(_)来代替一个明确的 `参数标签`。

3. 接下来，我们实例化这个 MLP 神经网络模型，实例化 MNIST 数据集，并将其存入 ``imageBatch`` 和 ``labelBatch`` 变量：

.. literalinclude:: /_static/code/zh/appendix/swift/mnist.swift
    :lines: 24-31

4. 然后，我们通过对数据集的循环，计算模型的梯度 ``grads`` 并通过 ``optimizer.update()`` 来反向传播更新模型的参数，进行训练：

.. literalinclude:: /_static/code/zh/appendix/swift/mnist.swift
    :lines: 33-42

.. admonition:: Swift 闭包函数（Closure）

    Swift 的闭包函数声明为：``{ (parameters) -> return type in statements }``，其中：``parameters`` 为闭包接受的参数，``return type`` 为闭包运行完毕的返回值类型，``statements`` 为闭包内的运行代码。
    
    比如上述代码中的  ``{ model -> Tensor<Float> in`` 这一段，就声明了一个传入参数为 ``model``，返回类型为 ``Tensor<Float>`` 的闭包函数。

.. admonition:: Swift 尾随闭包语法 (Trailing Closure Syntax)

    如果函数需要一个闭包作为参数，且这个参数是最后一个参数，那么我们可以将闭包函数放在函数参数列表外（也就是括号外），这种格式称为尾随闭包。

.. admonition:: Swift 输入输出参数 (In-Out Parameters)

    在 Swift 语言中，函数缺省是不可以修改参数的值的。为了让函数能够修改传入的参数变量，需要将传入的参数作为输入输出参数（In-Out Parmeters）。具体表现为需要在参数前加 ``&`` 符号，表示这个值可以被函数修改。

5. 最后，我们使用训练好的模型，在测试数据集上进行检查，得到模型的准度：

.. literalinclude:: /_static/code/zh/appendix/swift/mnist.swift
    :lines: 44-

以上程序运行输出为：

::

    Downloading train-images-idx3-ubyte ...
    Downloading train-labels-idx1-ubyte ...
    Reading data.
    Constructing data tensors.
    Test Accuracy: 0.9116667

本小节的源代码可以在 https://github.com/huan/tensorflow-handbook-swift 找到。加载 `MNIST` 数据集使用了作者封装的 Swift Module： `swift-MNIST <https://github.com/huan/swift-MNIST>`_。更方便的是在 Google Colab 上直接打开 `本例子的 Jupyter Notebook <https://colab.research.google.com/github/huan/tensorflow-handbook-swift/blob/master/tensorflow-handbook-swift-example.ipynb>`_ 直接运行。
