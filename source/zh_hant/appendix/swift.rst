Swift for TensorFlow (S4TF) (Huan）
==========================================================

.. figure:: /_static/image/swift/swift-tensorflow.png
    :width: 60%
    :align: center
     
    「Swift for TensorFlow is an attempt to change the default tools used by the entire machine learning and data science ecosystem.」
     
     -- Jameson Toole,  Co-founder & CTO of Fritz.ai

S4TF 簡介
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Google 推出的 Swift for TensorFlow （簡稱 S4TF）是專門針對 TensorFlow 優化過的 Swift 版本。（目前處在 Pre-Alpha 階段）

為了能夠在程式語言級支持 TensorFlow 所需的所有功能特性，S4TF 做為了 Swift 語言本身的一個分支，為 Swift 語言添加了機器學習所需要的所有功能擴展。它不僅僅是一個用 Swift 寫成的 TensorFlow API 封裝，Google 還為 Swift 增加了編譯器和語言增強功能，提供了一種新的編程模型，結合了圖的性能、Eager Execution 的靈活性和表達能力。

.. admonition:: Swift 語言創始人 Chris Lattner

    Swift 語言是 Chris Lattner 在蘋果公司工作時創建的。 現在 Chris Lattner 在 Google Brain 工作，專門從事深度學習的研究，並為 Swift 重寫了編譯器，為 TensorFlow 做定製優化。

本章我們將向大家簡要介紹 Swift for TensorFlow 的使用。你可以參考最新的 `Swift for TensorFlow 文件 <https://www.tensorflow.org/swift>`_.

為什麼要使用 Swift 進行 TensorFlow 開發
---------------------------------------------

相對於 TensorFlow 的其他版本（如 Python，C++ 等），S4TF 擁有其獨有的優勢，比如：

#. 開發效率高：強類型語言，能夠靜態檢查變數類型
#. 遷移成本低：與 Python，C，C++ 能夠無縫結合
#. 執行性能高：能夠直接編譯為底層硬體程式碼
#. 專門為機器學習打造：語言原生支持自動微分系統

與其他語言相比，S4TF 還有更多優勢。谷歌正在大力投資，使 Swift 成為其 TensorFlow ML 基礎設施的一個關鍵組件，而且很有可能 Swift 將成為深度學習的專屬語言。

.. admonition:: 更多使用 Swift 的理由

    有興趣的讀者可以參考官方文件：`Why Swift for TensorFlow <https://github.com/tensorflow/swift/blob/master/docs/WhySwiftForTensorFlow.md>`_

S4TF 環境配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本機安裝 Swift for TensorFlow
---------------------------------------------------------------

目前 S4TF 支持 Mac 和 Linux 兩個運行環境。安裝需要下載預先編譯好的軟體包，同時按照對應的操作，進行操作說明。安裝後，即可以使用全套 Swift 工具，包括 Swift（Swift REPL / Interpreter）和 Swiftc（Swift 編譯器）。官方文件（包含下載網址）可見 `這裡 <https://github.com/tensorflow/swift/blob/master/Installation.md>`_ 。

在 Colaboratory 中快速體驗 Swift for TensorFlow
---------------------------------------------------------------

Google 的 Colaboratory 可以直接支援 Swift 語言的執行環境。可以 `點此 <https://colab.research.google.com/github/huan/tensorflow-handbook-swift/blob/master/tensorflow-handbook-swift-blank.ipynb>`_ 直接打開一個空白的，具備 Swift 執行環境的 Colab Notebook ，這是立即體驗 Swift for TensorFlow 的最方便的辦法。


在 Docker 中快速體驗 Swift for TensorFlow
---------------------------------------------------------------

在本機已有 docker 環境的情況下，使用預先裝好的 Swift for TensorFlow 的 Docker Image 是非常方便的。

1. 開啟一個 S4TS 的 Jupyter Notebook

    在終端機中執行 ``nvidia-docker run -ti --rm -p 8888:8888 --cap-add SYS_PTRACE -v "$(pwd)":/notebooks zixia/swift`` 來啟動 Jupyter ，然後根據提示的 URL ，打開瀏覽器執行即可。


2. 執行一個本機的 Swift 程式碼文件
    
    為了執行本機的 ``s4tf.swift`` 文件，我們可以用如下 docker 指令：

    ::

        nvidia-docker run -ti --rm --privileged --userns=host \
            -v "$(pwd)":/notebooks \  
            zixia/swift \
            swift ./s4tf.swift

S4TF 基礎使用
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Swift 是動態強型態語言，也就是說 Swift 支持通過編譯器自動檢測類型，同時要求變數的使用要嚴格符合定義，並且變數都必須先定義後才能使用。

下面的程式碼，因為最初宣告的 ``n`` 是整數類型 ``42`` ，如果將 ``'string'`` 賦值給 ``n`` 時，會出現類型不匹配的問題，Swift 將會報錯：

.. code-block:: swift

    var n = 42
    n = "string"

報錯輸出：

::

    Cannot assign value of type 'String' to type 'Int'

下面是一個使用 TensorFlow 計算的基礎範例：

.. code-block:: swift

    import TensorFlow

    // 宣告兩個Tensor
    let x = Tensor<Float>([1])
    let y = Tensor<Float>([2])

    // 對兩個 Tensor 做加法運算
    let w = x + y

    // 輸出結果
    print(w)

.. admonition::  ``Tensor<Float>`` 中的 ``<Float>``

    在這裡的 ``Float`` 是用來指定 Tensor 這個類型所相關的內部資料類型。可以根據需求替換為其他合理資料類型，例如 “Double”。

在 Swift 中使用標準的 TensorFlow API
---------------------------------------------

在 ``import TensorFlow`` 之後，就可以在 Swift 語言中，使用核心的 TensorFlow API。

1. 處理數字和矩陣的程式碼，API 與 TensorFlow 高度保持了一致：

.. code-block:: swift

    let x = Tensor<BFloat16>(zeros: [32, 128])
    let h1 = sigmoid(matmul(x, w1) + b1)
    let h2 = tanh(matmul(h1, w1) + b1)
    let h3 = softmax(matmul(h2, w1) + b1)

2. 處理 Dataset 的程式碼，基本上，將 Python API 中的 ``tf.data.Dataset`` 同名函數直接改寫為 Swift 語法即可直接使用：

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

.. admonition:: ``matmul()`` 的別名： ``•``

    為了程式碼更加簡潔，``matmul(a, b)`` 可以簡寫爲 ``a • b``。``•`` 符號在 Mac 上，可以通過鍵盤按鍵 `Option + 8` 輸入。

在 Swift 中直接載入 Python 語言函式庫
---------------------------------------------

Swift 語言支持直接載入 Python 函數函式庫（比如 NumPy），也支持直接載入系統動態鏈接函式庫，很方便的做到即導入使用。

借助 S4TF 強大的集成能力，從 Python 遷移到 Swift 非常簡單。您可以逐步遷移 Python 程式碼（或繼續使用 Python 程式碼函式庫），因為 S4TF 支持直接在程式碼中載入 Python 原生程式碼函式庫，使得開發者可以繼續使用熟悉的語法在 Swift 中調用 Python 中已經完成的功能。

下面我們以 NumPy 為例，看一下如何在 Swift 語言中，直接載入 Python 的 NumPy 程式碼函式庫，並且直接進行調用：
.. code-block:: swift

    import Python

    let np = Python.import("numpy")
    let x = np.array([[1, 2], [3, 4]])
    let y = np.array([11, 12])
    print(x.dot(y))

輸出：

::

    [35 81]

除了能夠直接調用 Python 之外，Swift 也快成直接調用系統函數函式庫。比如下面的程式碼例子展示了我們可以在 Swift 中直接載入 Glibc 的動態函式庫，然後調用系統底層的 malloc 和 memcpy 函數，對變數直接進行操作。

.. code-block:: swift

    import Glibc
    let x = malloc(18)
    memcpy(x, "memcpy from Glibc", 18)
    free(x)

通過 Swift 強大的集成能力，針對 C/C++ 語言函式庫的載入和調用，處理起來也將會是非常簡單高效。

語言原生支持自動微分 
---------------------------------------------

我們可以通過 ``@differentiable`` 參數，非常容易的定義一個可被微分的函數：

.. code-block:: swift

    @differentiable
    func frac(x: Double) -> Double {
        return 1/x
    }

    gradient(of: frac)(0.5)

輸出：

::

    -4.0

在上面的程式碼例子中，我們通過將函數 ``frac()`` 標記爲 ``@differentiable`` ，然後就可以通過 ``gradient()`` 函數，將其轉換爲求解微分的新函數 ``gradient(of: trac)``，接下來就可以根據任意 x 值求解函數 frac 所在 x 點的梯度了。

.. admonition:: Swift 函數宣告中的參數名稱和類型

    Swift 使用 ``func`` 宣告一個函數。在函數的參數中，變數名的冒號後面代表的是 “參數類型”；在函數的參數和函數體（``{}``） 之前，還可以通過箭頭（``->``）來指定函數 ``返回值的類型``。

    比如在上面的程式碼中，參數變數名為 “x”；參數類型為 “Double”；函數返回類型為 “Double”。

MNIST數字分類
---------------------------------------------

下面我們以最簡單的 MNIST 數字分類為例子，給大家介紹一下基礎的 S4TF 範例程式碼。

1. 首先，導入 S4TF 模組  ``TensorFlow``、Python 橋接模組 ``Python``，基礎模組 ``Foundation``  和 MNIST 資料集模組 ``MNIST``：

.. literalinclude:: /_static/code/zh-hant/appendix/swift/mnist.swift
    :lines: 1-5

.. admonition:: Swift MNIST Dataset 模組

    Swift MNIST Dataset 模組是一個簡單易用的 MNIST 資料集載入模組，基於 Swift 語言，提供了完整的資料集載入 API。項目 Github：https://github.com/huan/swift-MNIST

2.  宣告一個最簡單的 MLP 神經網路架構，將輸入的 784 個圖像資料，轉換為 10 個神經元的輸出：

.. admonition:: 使用 ``Layer`` 協議定義神經網路模型

    為了在 Swift 中定義一個神經網路模型，我們需要建立一個 ``Struct`` 來實現模型結構，並確保符合 ``Layer`` 協議。
    
    其中，最為關鍵的部分是宣告 ``callAsFunction(_:)`` 方法，來定義輸入和輸出 Tensor 的對應關係。
    
.. literalinclude:: /_static/code/zh-hant/appendix/swift/mnist.swift
    :lines: 7-24

.. admonition:: Swift 參數標籤

    在程式碼中，我們會看到像是 ``callAsFunction(_ input: Input)`` 這樣的函數宣告。其中，``_`` 代表忽略參數標籤。

    Swift 中，每個函數參數都有一個 參數標籤 (Argument Label) 以及一個 參數名稱 (Parameter Name)。 參數標籤 主要應用在調用函數的情況，使得函數的實參與真實命名相關聯，更加容易理解實際參數的意義。同時因為有 參數標籤 的存在，確切的順序是可以隨意改變的。
    
    如果你不希望為參數添加標籤，可以使用一個下劃線 (_) 來代替一個明確的 參數標籤。

3. 接下來，我們實例化這個 MLP 神經網路模型，實例化 MNIST 資料集，並將其存入 ``imageBatch`` 和 ``labelBatch`` 變量：

.. literalinclude:: /_static/code/zh-hant/appendix/swift/mnist.swift
    :lines: 26-33

4. 然後，我們通過對資料集的循環，計算模型的梯度 ``grads`` 並通過 ``optimizer.update()`` 來反向傳播更新模型的參數，進行訓練：

.. literalinclude:: /_static/code/zh-hant/appendix/swift/mnist.swift
    :lines: 35-44

.. admonition:: Swift 閉包函數（Closure）

    Swift 的閉包函數宣告為：``{ (parameters) -> return type in statements }``，其中：``parameters`` 為閉包接受的參數，``return type`` 為閉包運行完畢的返回值類型，``statements`` 為閉包內的運行程式碼。
    
    比如上述程式碼中的   ``{ model -> Tensor<Float> in`` 這一段，就宣告了一個傳入參數為 ``model``，返回類型爲 ``Tensor<Float>`` 的閉包函數。

.. admonition:: Swift 尾隨閉包語法 (Trailing Closure Syntax)

    如果函數需要一個閉包作為參數，這個參數是最後一個參數，那我們可以將閉包函數放在函數參數列表外（也就是括號外），這種格式稱為尾隨閉包。

.. admonition:: Swift 輸入輸出參數 (In-Out Parameters)

    在 Swift 語言中，預設參數是不可以修改參數值的。為了讓函數能夠修改傳入參數變數，將需要將傳入的參數作為輸入輸出參數（In-Out Parmeters），在參數前加 ``&`` 符號，表示這個值可以被函數修改。

.. admonition:: 優化器的參數

    優化器更新模型參數的方法是 ``update(variables, along: direction)`` 。其中，``variables`` 是需要更新模型（內部包含的參數），因為需要被更新，所以我們通過添加 ``&`` 在參數變數前，通過引用的方式傳入。``direction`` 是模型參數所對應的梯度，需要通過參數標籤  ``along`` 來指定輸入。

5. 最後，我們使用訓練好的模型，在測試資料集上進行檢查，得到模型的準確度：

.. literalinclude:: /_static/code/zh-hant/appendix/swift/mnist.swift
    :lines: 46-

以上程式運行輸出為：

::

    Downloading train-images-idx3-ubyte ...
    Downloading train-labels-idx1-ubyte ...
    Reading data.
    Constructing data tensors.
    Test Accuracy: 0.9116667

本節的原始碼可以在 https://github.com/huan/tensorflow-handbook-swift 找到。載入 MNIST 資料集使用了作者封裝的 Swift Module： `swift-MNIST <https://github.com/huan/swift-MNIST>`_。更方便的是在 Google Colab 上直接打開 `本例子的 Jupyter Notebook <https://colab.research.google.com/github/huan/tensorflow-handbook-swift/blob/master/tensorflow-handbook-swift-example.ipynb>`_ 直接運行。

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 200 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>
