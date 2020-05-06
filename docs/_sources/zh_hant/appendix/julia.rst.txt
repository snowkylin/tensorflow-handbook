TensorFlow in Julia（Ziyang）
==========================================================

TensorFlow.jl 簡介
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

雖然 Julia 是一門非常優秀的語言，但是目前 TensorFlow 並不直接支持 Julia 。如果有需要，你可以選擇 TensorFlow.jl ，
這是一個由 `malmaud <https://github.com/malmaud/>`_ 封裝的第三方 Julia 包。它有和 Python 版本類似的 API ，也能支持 GPU 加速。

爲什麼要使用 Julia ？
---------------------------------------------

先進的語法糖，讓你能簡明扼要的表述計算過程。而高性能的 JIT ，提供了媲美靜態語言的速度（這一點是在數據預處理中非常重要，但也是 Python 難以企及的）。
所以，使用 Julia ，寫的快，跑的更快。
（你可以通過 `這個視頻 <https://www.youtube.com/watch?v=n2MwJ1guGVQ>`_ 了解更多）

本章我們將基於 TensorFlow.jl 0.12，向大家簡要介紹 Tensorflow 在 Julia 下的使用. 你可以參考最新的 `TensorFlow.jl 文檔 <https://malmaud.github.io/TensorFlow.jl/stable/tutorial.html>`_.

TensorFlow.jl 環境配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 docker 中快速體驗 TensorFlow.jl
--------------------------------------------

在本機已有 docker 環境的情況下，使用預裝 TensorFlow.jl 的 docker image 是非常方便的。

在命令行中執行 ``docker run -it malmaud/julia:tf`` ，然後就可以獲得一個已經安裝好 TensorFlow.jl 的 Julia REPL 環境。 (如果你不想直接打開 Julia，請在執行 ``docker run -it malmaud/julia:tf /bin/bash`` 來打開一個bash終端. 如需執行您需要的jl代碼文件，可以使用 docker 的目錄映射)

在 julia 包管理器中安裝 TensorFlow.jl
--------------------------------------------

在命令行中執行 ``julia`` 進入 Julia REPL 環境，然後執行以下命令安裝 TensorFlow.jl

.. code-block:: julia

    using pkg
    Pkg.add("TensorFlow")


TensorFlow.jl 基礎使用
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: julia

    using TensorFlow

    # 定義一個 Session
    sess = TensorFlow.Session()

    # 定義一個常量和變量
    x = TensorFlow.constant([1])
    y = TensorFlow.Variable([2])

    # 定義一個計算
    w = x + y

    # 執行計算過程
    run(sess, TensorFlow.global_variables_initializer())
    res = run(sess, w)

    # 輸出結果
    println(res)

MNIST數字分類
---------------------------------------------

這個例子來自於 `TensorFlow.jl 文檔 <https://malmaud.github.io/TensorFlow.jl/stable/tutorial.html>`_ ，可以用於對比 python 版本的 API.

.. code-block:: julia

    # 使用自帶例子中的 mnist_loader.jl 加載數據
    include(Pkg.dir("TensorFlow", "examples", "mnist_loader.jl"))
    loader = DataLoader()

    # 定義一個 Session
    using TensorFlow
    sess = Session()


    # 構建 softmax 回歸模型
    x = placeholder(Float32)
    y_ = placeholder(Float32)
    W = Variable(zeros(Float32, 784, 10))
    b = Variable(zeros(Float32, 10))

    run(sess, global_variables_initializer())

    # 預測類和損失函數
    y = nn.softmax(x*W + b)
    cross_entropy = reduce_mean(-reduce_sum(y_ .* log(y), axis=[2]))

    # 開始訓練模型
    train_step = train.minimize(train.GradientDescentOptimizer(.00001), cross_entropy)
    for i in 1:1000
        batch = next_batch(loader, 100)
        run(sess, train_step, Dict(x=>batch[1], y_=>batch[2]))
    end

    # 查看結果並評估模型
    correct_prediction = indmax(y, 2) .== indmax(y_, 2)
    accuracy=reduce_mean(cast(correct_prediction, Float32))
    testx, testy = load_test_set()

    println(run(sess, accuracy, Dict(x=>testx, y_=>testy)))
