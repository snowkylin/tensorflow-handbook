TensorFlow 模型建立與訓練
========================================

.. _linear:

本章介紹如何使用 TensorFlow 快速建立動態模型。

- 模型的建構： ``tf.keras.Model`` 和 ``tf.keras.layers``
- 模型的損失函數： ``tf.keras.losses``
- 模型的優化器： ``tf.keras.optimizer``
- 模型的評估： ``tf.keras.metrics``

.. admonition:: 前置知識

    * `Python  物件導向程式語言 <https://openhome.cc/Gossip/CodeData/PythonTutorial/FunctionModuleClassPackagePy3.html>`_ （在 Python 內定義類別的方法、類別的繼承、建構和解構函式，`使用 super() 函數呼叫父類方法 <https://medium.com/@dboyliao/python-%E7%B9%BC%E6%89%BF-543-bc3d8ef51d6ds>`_ ，`使用__call__() 方法對實例進行呼叫 <https://kknews.cc/zh-tw/code/z9p8rvg.html>`_ 等）；
    * 多層感知器(Multilayer Perceptron, MLP)、卷積神經網路、循環神經網路和強化學習(Reinforcement Learning, RL)（每節之前給出參考資料）。
    *  `Python 的裝飾器 <https://medium.com/citycoddee/python%E9%80%B2%E9%9A%8E%E6%8A%80%E5%B7%A7-3-%E7%A5%9E%E5%A5%87%E5%8F%88%E7%BE%8E%E5%A5%BD%E7%9A%84-decorator-%E5%97%B7%E5%97%9A-6559edc87bc0>`_ （非必須）

模型（Model）與層（Layer）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
..  https://www.tensorflow.org/programmers_guide/eager

在 TensorFlow 中，推薦使用 Keras（ ``tf.keras`` ）建立模型。Keras 是一個廣為流行的高級神經網路 API，簡單、快速而不失靈活性，現已得到 TensorFlow 的官方內建和全面支援。

Keras 有兩個重要的概念： **模型（Model）** 和 **層（Layer）** 。層將各種計算流程和變數進行了封裝（例如基本的全連接層(Fully Connected Layer)，CNN 的卷積層(Convolution Layer)、池化層(Pooling Layer)...等），而模型則將各種層進行組織和連接，並封裝成一個整體，描述了如何將輸入資料通過各種層以及運算而得到輸出。在需要模型呼叫的時候，使用 ``y_pred = model(X)`` 的形式即可。Keras 在 ``tf.keras.layers`` 下內建了深度學習中大量常用的的預定義層，同時也允許我們自定義層。

Keras 模型以類別的形式呈現，我們可以通過繼承 ``tf.keras.Model`` 這個 Python 類別來定義自己的模型。在繼承類別中，我們需要重寫 ``__init__()`` （建構函數，初始化）和 ``call(input)`` （模型呼叫）兩個方法，同時也可以根據需要增加自定義的方法。

.. code-block:: python

    class MyModel(tf.keras.Model):
        def __init__(self):
            super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
            # 此處添加初始化程式碼（包含 call 方法中會用到的層），例如
            # layer1 = tf.keras.layers.BuiltInLayer(...)
            # layer2 = MyCustomLayer(...)

        def call(self, input):
            # 此處添加模型呼叫的程式碼（處理輸入並返回輸出），例如
            # x = layer1(input)
            # output = layer2(x)
            return output

        # 還可以添加自定義的方法

.. figure:: /_static/image/model/model_cn.png
    :width: 50%
    :align: center

    Keras 模型類別定義示意圖

繼承 ``tf.keras.Model`` 後，我們同時可以使用父類的若干方法和屬性，例如在實例化類 ``model = Model()`` 後，可以通過 ``model.variables`` 這一屬性直接獲得模型中的所有變數，免去我們一個個顯示指定變數的麻煩。

上一章中簡單的線性模型 ``y_pred = a * X + b`` ，我們可以通過Model類別的方式編寫如下：

.. literalinclude:: /_static/code/zh-hant/model/linear/linear.py

這裡，我們沒有顯式宣告 ``a`` 和 ``b`` 兩個變數並寫出 ``y_pred = a * X + b`` 這一線性變換，而是建立了一個繼承了 ``tf.keras.Model`` 的模型類 ``Linear`` 。這個類別在初始化部分實例了一個 **全連接層** （ ``tf.keras.layers.Dense`` ），並在 call 方法中對這個層進行呼叫，實現了線性變換的計算。如果需要顯式宣告自己的變數並使用變數進行自定義運算，或者希望了解 Keras 層的內部原理，請參考 :ref:`自定義層 <custom_layer>`。

.. admonition:: Keras 的全連接層：線性變換 + 激活函數

    `全連接層 <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`_ （Fully-connected Layer，``tf.keras.layers.Dense`` ）是 Keras 中最基礎和常用的層之一，對輸入矩陣 :math:`A` 進行 :math:`f(AW + b)` 的線性變換 + 激活函數操作。如果不指定激活函數，即是純粹的線性變換 :math:`AW + b`。具體而言，給定輸入張量 ``input = [batch_size, input_dim]`` ，該層對輸入張量首先進行 ``tf.matmul(input, kernel) + bias`` 的線性變換（ ``kernel`` 和 ``bias`` 是層中可訓練的變量），然後對線性變換後張量的每個元素通過激活函數 ``activation`` ，從而輸出形狀為 ``[batch_size, units]`` 的二維張量。

    .. figure:: /_static/image/model/dense.png
        :width: 60%
        :align: center

    其包含的主要參數如下：

    * ``units`` ：輸出張量的維度；
    * ``activation`` ：激活函數，對應於 :math:`f(AW + b)` 中的 :math:`f` ，預設為無激活函數（ ``a(x) = x`` ）。常用的激活函數包括 ``tf.nn.relu`` 、 ``tf.nn.tanh`` 和 ``tf.nn.sigmoid`` ；
    * ``use_bias`` ：是否加入偏移量 ``bias`` ，即 :math:`f(AW + b)` 中的 :math:`b`。預設為 ``True`` ；
    * ``kernel_initializer`` 、 ``bias_initializer`` ：權重矩陣 ``kernel`` 和偏移量 ``bias`` 兩個變數的初始化器。預設為 ``tf.glorot_uniform_initializer`` [#glorot]_ 。設置為 ``tf.zeros_initializer`` 表示將兩個變量均初始化為全 0；

    該層包含權重矩陣 ``kernel = [input_dim, units]`` 和偏移量 ``bias = [units]`` [#broadcast]_ 兩個可訓練變數，對應於 :math:`f(AW + b)` 中的 :math:`W` 和 :math:`b`。

    這裡著重從數學矩陣運算和線性變換的角度描述了全連接層。基於神經元建模的描述可參考 :ref:`後文介紹 <neuron>` 。

    .. [#glorot] Keras 中的很多層都預設使用 ``tf.glorot_uniform_initializer`` 初始化變數，關於該初始化器可參考 https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform。
    .. [#broadcast] 你可能會注意到， ``tf.matmul(input, kernel)`` 的結果是一個形狀為 ``[batch_size, units]`` 的二維矩陣，這個二維矩陣要如何與形狀為 ``[units]`` 的一維偏移量 ``bias`` 相加呢？事實上，這裡是 TensorFlow 的 Broadcasting 機制在起作用，該加法運算相當於將二維矩陣的每一行加上了 ``Bias`` 。Broadcasting 機制的具體介紹可見 https://www.tensorflow.org/xla/broadcasting 。

.. admonition:: 為什麼模型類是重新呼叫 ``call()`` 方法而不是  ``__call__()`` 方法？

    在 Python 中，對類別的實例 ``myClass`` 進行形如 ``myClass()`` 的呼叫等價於 ``myClass.__call__()`` （具體請見本章初 “前置知識” 的 ``__call__()`` 部分）。那麼看起來，為了使用 ``y_pred = model(X)`` 的形式呼叫模型類別，應該重寫 ``__call__()`` 方法才對呀？原因是 Keras 在模型呼叫的前後還需要有一些自己的內部操作，所以暴露出一個專門用於重新呼叫的 ``call()`` 方法。 ``tf.keras.Model`` 這一父類已經包含 ``__call__()`` 的定義。 ``__call__()`` 中主要呼叫了 ``call()`` 方法，同時還需要在進行一些 keras 的內部操作。這裡，我們通過繼承 ``tf.keras.Model`` 並重新呼叫 ``call()`` 方法，即可在保持 keras 結構的同時加入模型呼叫的程式碼。

.. _mlp:

基礎範例：多層感知器（MLP）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我們從編寫一個最簡單的 `多層感知器 <https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E5%99%A8>`_ （Multilayer Perceptron, MLP），或者說 “多層全連接神經網路” 開始，介紹 TensorFlow 的模型編寫方式。在這一部分，我們依次進行以下步驟：

- 使用 ``tf.keras.datasets`` 獲得資料集並預處理
- 使用 ``tf.keras.Model`` 和 ``tf.keras.layers`` 建構模型
- 建構模型訓練流程，使用 ``tf.keras.losses`` 計算損失函數，並使用 ``tf.keras.optimizer`` 優化模型
- 構建模型評估流程，使用 ``tf.keras.metrics`` 計算評量指標

.. admonition:: 基礎知識和原理

    * UFLDL 教程 `Multi-Layer Neural Network <http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/>`_ 一節；
    * 史丹佛大學課程 `CS231n: Convolutional Neural Networks for Visual Recognition <https://cs231n.github.io/>`_ 中的 “Neural Networks Part 1 ~ 3” 部分。

這裡，我們使用多層感知器完成 MNIST 手寫體數字圖片資料集 [LeCun1998]_ 的分類任務。

.. figure:: /_static/image/model/mnist_0-9.png
    :align: center

    MNIST 手寫體數字圖片範例

資料獲取及預處理： ``tf.keras.datasets``
-------------------------------------------

先進行預備工作，實現一個簡單的 ``MNISTLoader`` 類來讀取 MNIST 資料集資料。這裡使用了 ``tf.keras.datasets`` 快速載入 MNIST 資料集。

.. literalinclude:: /_static/code/zh-hant/model/utils.py
    :lines: 5-19

.. hint:: ``mnist = tf.keras.datasets.mnist`` 將從網路上自動下載 MNIST 資料集並加載。如果運行時出現網路連接錯誤，可以從 https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz 或 https://s3.amazonaws.com/img-datasets/mnist.npz 下載 MNIST 資料集資料。這裡使用了 ``mnist.npz`` 文件，並放置於使用者目錄的 ``.keras/dataset`` 目錄下（Windows 下使用者目錄為 ``C:\Users\使用者名`` ，Linux 下使用者目錄為 ``/home/使用者名`` ）。

.. admonition:: TensorFlow 的圖像資料表示

    在 TensorFlow 中，圖片資料集的一種典型表示是 ``[圖片數目，長，寬，色彩通道數]`` 的四維張量。在上面的 ``DataLoader`` 類中， ``self.train_data`` 和 ``self.test_data`` 分別載入了 60,000 和 10,000 張大小為 ``28*28`` 的手寫體數字圖片。由於這裡讀入的是灰階圖片，色彩通道數為 1（彩色 RGB 圖像色彩通道數為 3），所以我們使用 ``np.expand_dims()`` 函數為圖片資料手動在最後添加一維通道。

.. _mlp_model:

模型的建構： ``tf.keras.Model`` 和 ``tf.keras.layers``
-------------------------------------------------------------------------------

多層感知器的模型類別實現與上面的線性模型類似，使用 ``tf.keras.Model`` 和 ``tf.keras.layers`` 建構，所不同的地方在於層數增加了（顧名思義，“多層” 感知器），以及引入了非線性激活函數（這裡使用了 `ReLU 函數 <https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E6%95%B4%E6%B5%81%E5%87%BD%E6%95%B0>`_ ， 即下方的 ``activation=tf.nn.relu`` ）。該模型輸入一個向量（比如這裡是拉直的 ``1×784`` 手寫體數字圖片），輸出 10 維的向量，分別代表這張圖片屬於 0 到 9 的機率。

.. literalinclude:: /_static/code/zh-hant/model/mnist/mlp.py
    :lines: 4-

.. admonition:: softmax 函數

    這裡，因為我們希望輸出 “輸入圖片分別屬於 0 到 1 的機率”，也就是一個 10 維的離散機率分佈，所以我們希望這個 10 維向量至少滿足兩個條件：

    * 該向量中的每個元素均在 :math:`[0, 1]` 之間；
    * 該向量的所有元素之和為 1。

    為了使得模型的輸出能始終滿足這兩個條件，我們使用 `Softmax 函數 <https://zh.wikipedia.org/wiki/Softmax%E5%87%BD%E6%95%B0>`_ （正規化指數函數， ``tf.nn.softmax`` ）對模型的原始輸出進行正規化。其形式為 :math:`\sigma(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}` 。不僅如此，softmax 函數能夠凸顯原始向量中最大的值，並抑制低於最大值的其他分量，這也是該函數被稱作 softmax 函數的原因（即平滑化的 argmax 函數）。

.. figure:: /_static/image/model/mlp.png
    :width: 80%
    :align: center

    MLP 模型示意圖

模型的訓練： ``tf.keras.losses`` 和 ``tf.keras.optimizer``
-------------------------------------------------------------------------------

定義一些模型超參數(Hyperparameters)：

.. literalinclude:: /_static/code/zh-hant/model/mnist/main.py
    :lines: 8-10

實例化模型和資料讀取類，並實例化一個 ``tf.keras.optimizer`` 的優化器（這裡使用常用的 Adam 優化器）：

.. code-block:: python

    model = MLP()
    data_loader = MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

然後疊代進行以下步驟：

- 從 DataLoader 中隨機取一批訓練資料；
- 將這批資料送入模型，計算出模型的預測值；
- 將模型預測值與真實值進行比較，計算損失函數（loss）。這里使用 ``tf.keras.losses`` 中的交叉熵函數作為損失函數；
- 計算損失函數關於模型變數的導數；
- 將求出的導數值傳入優化器，使用優化器的 ``apply_gradients`` 方法更新模型參數以最小化損失函數（優化器的詳細使用方法見 :ref:`前章 <optimizer>`  ）。

具體代碼實現如下：

.. literalinclude:: /_static/code/zh-hant/model/mnist/main.py
    :lines: 93-102

.. admonition:: 交叉熵（cross entropy）與 ``tf.keras.losses``

    你或許注意到了，在這裡，我們沒有明顯的寫出一個損失函數，而是使用了 ``tf.keras.losses`` 中的 ``sparse_categorical_crossentropy`` （交叉熵）函數，將模型的預測值 ``y_pred`` 與真實的標籤值 ``y`` 作為函數參數傳入，由 Keras 幫助我們計算損失函數的值。

    交叉熵作為損失函數，在分類問題中被廣泛應用。其離散形式為 :math:`H(y, \hat{y}) = -\sum_{i=1}^{n}y_i \log(\hat{y_i})` ，其中 :math:`y` 為真實機率分佈， :math:`\hat{y}` 為預測機率分佈， :math:`n` 為分類任務的類別個數。預測機率分佈與真實分佈越接近，則交叉熵的值越小，反之則越大。更具體的介紹及其在機器學習中的應用可參考 `這篇文章 <https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E4%BB%8B%E7%B4%B9-%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8-loss-function-2dcac5ebb6cb>`_ 。

    在 ``tf.keras`` 中，有兩個交叉熵相關的損失函數 ``tf.keras.losses.categorical_crossentropy`` 和 ``tf.keras.losses.sparse_categorical_crossentropy`` 。其中 sparse 的含義是，真實的標籤值 ``y_true`` 可以直接傳入 int 類型的標籤類別。具體而言：

    .. code-block:: python

        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)

    與

    .. code-block:: python

        loss = tf.keras.losses.categorical_crossentropy(
            y_true=tf.one_hot(y, depth=tf.shape(y_pred)[-1]),
            y_pred=y_pred
        )

    的結果相同。

模型的評估： ``tf.keras.metrics``
-------------------------------------------------------------------------------

最後，我們使用測試集評估模型的性能。這裡，我們使用 ``tf.keras.metrics`` 中的 ``SparseCategoricalAccuracy`` 評量器來評估模型在測試集上的性能，該評量器能夠對模型預測的結果與真實結果進行比較，並輸出預測正確的樣本數占總樣本數的比例。我們疊代測試資料集，每次通過 ``update_state()`` 方法向評量器輸入兩個參數： ``y_pred`` 和 ``y_true`` ，即模型預測出的結果和真實結果。評量器具有內部變數來保存當前評估指標相關的參數數值（例如當前已傳入的累計樣本數和當前預測正確的樣本數）。疊代結束後，我們使用 ``result()`` 方法輸出最終的評量指標值（預測正確的樣本數占總樣本數的比例）。

在以下評量器程式碼中，我們提出了一個實例 ``tf.keras.metrics.SparseCategoricalAccuracy``，並使用 For 循環疊代分批次傳入了測試集資料的預測結果與真實結果，並輸出訓練後的模型在測試資料集上的準確率。

.. literalinclude:: /_static/code/zh-hant/model/mnist/main.py
    :lines: 104-110

輸出結果::

    test accuracy: 0.947900

可以注意到，使用這樣簡單的模型，已經可以達到 95% 左右的準確率。

.. _neuron:

.. admonition:: 神經網路的基本單位：神經元 [#order]_

    如果我們將上面的神經網路放大來看，詳細研究計算過程，比如取第二層的第 k 個計算單元，可以得到示意圖如下：

    .. figure:: /_static/image/model/neuron.png
        :width: 80%
        :align: center

    該計算單元 :math:`Q_k` 有 100 個權重值參數 :math:`w_{0k}, w_{1k}, ..., w_{99k}` 和 1 個偏移參數 :math:`b_k` 。將第 1 層中所有的 100 個計算單元 :math:`P_0, P_1, ..., P_{99}` 的值作為輸入，分別按權重值 :math:`w_{ik}` 加和（即 :math:`\sum_{i=0}^{99} w_{ik} P_i` ），並加上偏置值 :math:`b_k` ，然後送入激活函數 :math:`f` 進行計算，即得到輸出結果。

    事實上，這種結構和真實的神經細胞（神經元）類似。神經元由樹突、細胞體和軸突構成。樹突接受其他神經元傳來的信號作為輸入（一個神經元可以有數千甚至上萬樹突），細胞體對電位信號進行整合，而產生的信號則通過軸突傳到神經末梢的突觸，傳播到下一個（或多個）神經元。

    .. figure:: /_static/image/model/real_neuron_cn.png
        :width: 80%
        :align: center

        神經細胞模式圖（修改自 Quasar Jarosz at English Wikipedia [CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0)]）

    上面的計算單元，可以被視作對神經元結構的數學建模。在上面的例子裡，第二層的每一個計算單元（神經元）有 100 個權重值參數和 1 個偏移參數，而第二層計算單元的數目是 10 個，因此這一個全連接層的總參數量為 100*10 個權重值參數和 10 個偏移參數。事實上，這正是該全連接層中的兩個變數 ``kernel`` 和 ``bias`` 的形狀。仔細研究一下，你會發現，這裡基於神經元建模的介紹與上文基於矩陣計算的介紹是等價的。

    .. [#order] 事實上，應當是先有神經元建模的概念，再有基於神經元和層結構的神經網路。但由於本手冊著重介紹 TensorFlow 的使用方法，所以調整了介紹順序。

卷積神經網路（CNN）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`卷積神經網路 <https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C>`_ （Convolutional Neural Network, CNN）是一種結構類似於人類或動物的 `視覺系統 <https://zh.wikipedia.org/wiki/%E8%A7%86%E8%A7%89%E7%B3%BB%E7%BB%9F>`_ 的神經網路，包含一個或多個卷積層（Convolutional Layer）、池化層（Pooling Layer）和全連接層（Fully-connected Layer）。

.. admonition:: 基礎知識和原理

    * 台灣大學李宏毅教授的《機器學習》課程的 `Convolutional Neural Network <https://www.youtube.com/watch?v=FrKWiRv254g>`_ 一章；
    * UFLDL 教程 `Convolutional Neural Network <http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/>`_ 一節；
    * 史丹佛課程 `CS231n: Convolutional Neural Networks for Visual Recognition <https://cs231n.github.io/>`_ 中的 “Module 2: Convolutional Neural Networks” 部分。

使用Keras實現卷積神經網路
-------------------------------------------------------

卷積神經網路的一個範例實現如下所示，和 :ref:`上節中的多層感知器 <mlp_model>` 在程式碼結構上很類似，只是新加入了一些卷積層和池化層。這裡的網路結構並不是唯一的，可以增加、刪除或調整 CNN 的網路結構和參數，以達到更好的性能。

.. literalinclude:: /_static/code/zh-hant/model/mnist/cnn.py
    :lines: 4-

.. figure:: /_static/image/model/cnn.png
    :align: center

    範例程式碼中的 CNN 結構圖示

將前節的 ``model = MLP()`` 更換成 ``model = CNN()`` ，輸出如下::

    test accuracy: 0.988100

可以發現準確率相較於前節的多層感知器有非常顯著的提高。事實上，通過改變模型的網路結構（比如加入 Dropout 層防止過擬合），準確率還有進一步提升的空間。

使用Keras中預定義的典型卷積神經網路結構
-------------------------------------------------------

``tf.keras.applications`` 中有一些預定義好的典型卷積神經網路結構，如 ``VGG16`` 、 ``VGG19`` 、 ``ResNet`` 、 ``MobileNet`` 等。我們可以直接呼叫這些典型的卷積神經網路結構（甚至載入預訓練的參數），而無需手動定義網路結構。

例如，我們可以使用以下代碼來實例化一個 ``MobileNetV2`` 網路結構：

.. code-block:: python

    model = tf.keras.applications.MobileNetV2()

當執行以上程式碼時，TensorFlow會自動從網路上下載 ``MobileNetV2`` 網路結構，因此在第一次執行程式碼時需要具備網路連接。每個網路結構具有自己特定的詳細參數設置，一些共通的常用參數如下：

- ``input_shape`` ：輸入張量的形狀（不含第一維的Batch），大多預設為 ``224 × 224 × 3`` 。一般而言，模型對輸入張量的大小有下限，長和寬至少為 ``32 × 32`` 或 ``75 × 75`` ；
- ``include_top`` ：在網路的最後是否包含全連接層，默認為 ``True`` ；
- ``weights`` ：預訓練權重值，預設為 ``'imagenet'`` ，即為當前模型載入在ImageNet資料集上預訓練的權重值。如需隨機初始化變數可設為 ``None`` ；
- ``classes`` ：分類數，預設為1000。修改該參數需要 ``include_top`` 參數為 ``True`` 且 ``weights`` 參數為 ``None`` 。

各網路模型參數的詳細介紹可參考 `Keras文件 <https://keras.io/applications/>`_ 。

以下展示一個例子，使用 ``MobileNetV2`` 網路在 ``tf_flowers`` 五種分類數據集上進行訓練（為了程式碼的簡短高效，在該範例中我們使用了 :doc:`TensorFlow Datasets <../appendix/tfds>` 和 :ref:`tf.data <tfdata>` 載入和預處理資料）。通過將 ``weights`` 設置為 ``None`` ，我們隨機初始化變數而不使用預訓練權重值。同時將 ``classes`` 設置為5，對應於5種分類的資料集。

.. literalinclude:: /_static/code/zh-hant/model/cnn/mobilenet.py
    :emphasize-lines: 10    

後文的部分章節（如 :doc:`分散式訓練 <../appendix/distributed>` ）中，我們也會直接呼叫這些經典的網路結構來進行訓練。

.. admonition:: 卷積層和池化層的工作原理

    卷積層（Convolutional Layer，以 ``tf.keras.layers.Conv2D`` 為代表）是 CNN 的核心組件，其結構與大腦的視覺皮層有相似之處。

    回憶我們之前建立的 :ref:`神經細胞的計算模型 <neuron>` 以及全連接層，我們預設每個神經元與上一層的所有神經元相連。不過，在視覺皮層的神經元中，情況並不是這樣。你或許在生物課上學習過 **接受區** （Receptive Field）這一概念，即視覺皮層中的神經元並非與前一層的所有神經元相連，而只是感受一片區域內的視覺信號，並只對局部區域的視覺刺激進行反應。CNN 中的卷積層正體現了這一特性。

    例如，下圖是一個 7×7 的單通道圖片信號輸入：

    .. figure:: /_static/image/model/conv_image.png
        :align: center

    如果使用之前基於全連接層的模型，我們需要讓每個輸入信號對應一個權重值，即建模一個神經元需要 7×7=49 個權重值（加上偏置項是50個），並得到一個輸出信號。如果一層有 N 個神經元，我們就需要 49N 個權重值，並得到 N 個輸出信號。

    而在 CNN 的卷積層中，我們這樣建立一個卷積層的神經元：

    .. figure:: /_static/image/model/conv_field.png
        :align: center

    圖中 3×3 的紅框代表該神經元的接受區。由此，我們只需 3×3=9 個權重值 :math:`W = \begin{bmatrix}w_{1, 1} & w_{1, 2} & w_{1, 3} \\w_{2, 1} & w_{2, 2} & w_{2, 3} \\w_{3, 1} & w_{3, 2} & w_{3, 3}\end{bmatrix}`  ，外加1個偏移項 :math:`b`  ，即可得到一個輸出信號。例如，對於紅框所示的位置，輸出信號即為對矩陣 :math:`\begin{bmatrix}0 \times w_{1, 1} & 0 \times w_{1, 2} & 0 \times w_{1, 3} \\0 \times w_{2, 1} & 1 \times w_{2, 2} & 0 \times w_{2, 3} \\0 \times w_{3, 1} & 0 \times w_{3, 2} & 2 \times w_{3, 3}\end{bmatrix}` 的所有元素求和並加上偏移項 :math:`b`，記作 :math:`a_{1, 1}`  。

    不過，3×3 的範圍顯然不足以處理整個圖片，因此我們使用滑動視窗的方法。使用相同的參數 :math:`W` ，但將紅框在圖片中從左到右滑動，進行逐行掃描，每滑動到一個位置就計算一個值。例如，當紅框向右移動一個單位時，我們計算矩陣 :math:`\begin{bmatrix}0 \times w_{1, 1} & 0 \times w_{1, 2} & 0 \times w_{1, 3} \\1 \times w_{2, 1} & 0 \times w_{2, 2} & 1 \times w_{2, 3} \\0 \times w_{3, 1} & 2 \times w_{3, 2} & 1 \times w_{3, 3}\end{bmatrix}` 的所有元素的和加上偏移項 :math:`b`，記作 :math:`a_{1, 2}` 。由此，和一般的神經元只能輸出 1 個值不同，這里的卷積層神經元可以輸出一個 5×5 的矩陣 :math:`A = \begin{bmatrix}a_{1, 1} & \cdots & a_{1, 5} \\ \vdots & & \vdots \\ a_{5, 1} & \cdots & a_{5, 5}\end{bmatrix}`  。

    .. figure:: /_static/image/model/conv_procedure_cn.png
        :align: center

        卷積示意圖。一個單通道的 7×7 圖片在通過一個接受區為 3×3 ，參數為10個的卷積層神經元後，得到 5×5 的矩陣作為卷積結果。

    下面，我們使用TensorFlow來驗證一下上圖的計算結果。

    將上圖中的輸入圖片、權重值矩陣 :math:`W` 和偏移項 :math:`b` 表示為NumPy陣列 ``image`` , ``W`` , ``b`` 如下：

    .. literalinclude:: /_static/code/zh-hant/model/cnn/cnn_example.py
        :lines: 4-21

    然後建立一個僅有一個卷積層的模型，用 ``W`` 和 ``b`` 初始化 [#sequential]_ ：

    .. literalinclude:: /_static/code/zh-hant/model/cnn/cnn_example.py
        :lines: 23-30

    最後將圖片資料 ``image`` 輸入模型，列印輸出：

    .. literalinclude:: /_static/code/zh-hant/model/cnn/cnn_example.py
        :lines: 32-33

    程式運行結果為：

    ::

        tf.Tensor(
        [[ 6.  5. -2.  1.  2.]
         [ 3.  0.  3.  2. -2.]
         [ 4.  2. -1.  0.  0.]
         [ 2.  1.  2. -1. -3.]
         [ 1.  1.  1.  3.  1.]], shape=(5, 5), dtype=float32)

    可見與上圖中矩陣 :math:`A`  的值一致。
    
    還有一個問題，以上假設圖片都只有一個通道（例如灰階圖片），但如果圖片是彩色的（例如有 RGB 三個通道）該怎麼辦呢？此時，我們可以為每個通道準備一個 3×3 的權重值矩陣，即一共有 3×3×3=27 個權重值。對於每個通道，均使用自己的權重值矩陣進行處理，輸出時將多個通道所輸出的值進行加和即可。

    可能有讀者會注意到，按照上述介紹的方法，每次卷積後的結果相比於原始圖片而言，四周都會“少一圈”。比如上面 7×7 的圖片，卷積後變成了 5×5 ，這有時會為後面的工作帶來麻煩。因此，我們可以設定padding策略。在 ``tf.keras.layers.Conv2D`` 中，當我們將 ``padding`` 參數設為 ``same`` 時，會將周圍缺少的部分使用0補齊，使得輸出的矩陣大小和輸入一致。

    最後，既然我們可以使用滑動窗口的方法進行卷積，那麼每次滑動的步長是不是可以設置呢？答案是肯定的。通過 ``tf.keras.layers.Conv2D`` 的 ``strides`` 參數即可設置步長（預設為1）。比如，在上面的例子中，如果我們將步長設定為2，輸出的卷積結果即會是一個3×3的矩陣。

    ..
        一個動態演示如下圖所示。其中紅色的矩陣為多通道的圖像（這里展示為 2 個通道），綠色的矩陣為圖像的每個通道所對應的權值矩陣 :math:`W` ，藍色的矩陣為輸出矩陣 :math:`A`  。

        .. figure:: /_static/image/model/conv_sliding_window.gif
            :align: center

            捲積示意圖（來源： https://blog.csdn.net/huachao1001/article/details/79120521 ）

    事實上，卷積的形式多種多樣，以上的介紹只是其中最簡單和基礎的一種。更多卷積方式的範例可見 `Convolution arithmetic <https://github.com/vdumoulin/conv_arithmetic>`_ 。

    池化層（Pooling Layer）的理解則簡單得多，其可以理解為對圖片進行下降取樣的過程，對於每一次滑動窗口中的所有值，輸出其中的最大值（MaxPooling）、均值或其他方法產生的值。例如，對於一個三通道的 16×16 圖像（即一個 ``16*16*3`` 的張量），經過感受野為 2×2，滑動步長為 2 的池化層，則得到一個 ``8*8*3`` 的張量。

    .. [#sequential] 這里使用了較為簡易的Sequential模式建立模型，具體介紹見 :ref:`後文 <sequential_functional>`  。

循環神經網路（RNN）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

循環神經網路（Recurrent Neural Network, RNN）是一種適宜於處理序列資料的神經網路，被廣泛用於語言模型、文本生成、機器翻譯等。

.. admonition:: 基礎知識和原理

    - `Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs <http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/>`_
    - 台灣大學李宏毅教授的《機器學習》課程的 `Recurrent Neural Network (part 1) <https://www.youtube.com/watch?v=xCGidAeyS4M>`_ `Recurrent Neural Network (part 2) <https://www.youtube.com/watch?v=rTqmWlnwz_0>`_ 兩部分。
    - LSTM 原理：`Understanding LSTM Networks <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_
    - RNN 序列生成：[Graves2013]_

這裡，我們使用 RNN 來進行尼采風格文本的自動生成。 [#rnn_reference]_

這個任務的本質其實預測一段英文文本的接續字母的機率分佈。比如，我們有以下句子::

    I am a studen

這個句子（序列）一共有 13 個字符（包含空格）。當我們閱讀到這個由 13 個字符組成的序列後，根據我們的經驗，我們可以預測出下一個字符很大機率是 “t”。我們希望建立這樣一個模型，逐個輸入一段長為 ``seq_length`` 的序列，輸出這些序列接續的下一個字元的機率分佈。我們從下一個字符的機率分佈中取樣作為預測值，然後滾雪球式的生成下兩個字符，下三個字符等等，即可完成文本的生成任務。

首先，還是實現一個簡單的 ``DataLoader`` 類別來讀取文本，並以字符為單位進行編碼。設字符種類數為 ``num_chars`` ，則每種字符賦予一個 0 到 ``num_chars - 1`` 之間的唯一整數編號 i。

.. literalinclude:: /_static/code/zh-hant/model/text_generation/rnn.py
    :lines: 35-53

接下來進行模型的實現。在 ``__init__`` 方法中我們實例化一個常用的 ``LSTMCell`` 單元，以及一個線性變換用的全連接層，我們首先對序列進行“One Hot”操作，即將序列中的每個字元的編碼 i 均變換為一個 ``num_char`` 維向量，其第 i 位為 1，其餘均為 0。變換後的序列張量形狀為 ``[seq_length, num_chars]`` 。然後，我們初始化 RNN 單元的狀態，存入變量 ``state`` 中。接下來，將序列從頭到尾依次送入 RNN 單元，即在 t 時間，將上一個時間 t-1 的 RNN 單元狀態 ``state`` 和序列的第 t 個元素 ``inputs[t, :]`` 送入 RNN 單元，得到當前時間的輸出 ``output`` 和 RNN 單元狀態。取 RNN 單元最後一次的輸出，通過全連接層變換到 ``num_chars`` 維，即作為模型的輸出。

.. figure:: /_static/image/model/rnn_single.jpg
    :width: 50%
    :align: center

    ``output, state = self.cell(inputs[:, t, :], state)`` 圖示

.. figure:: /_static/image/model/rnn.jpg
    :width: 100%
    :align: center

    RNN 流程圖示

具體實現如下：

.. literalinclude:: /_static/code/zh-hant/model/text_generation/rnn.py
    :lines: 7-25

定義一些模型超參數：

.. literalinclude:: /_static/code/zh-hant/model/text_generation/rnn.py
    :lines: 57-60

訓練過程與前節基本一致，在此不再贅述：

- 從 ``DataLoader`` 中隨機取一批訓練資料；
- 將這批資料送入模型，計算出模型的預測值；
- 將模型預測值與真實值進行比較，計算損失函數（loss）；
- 計算損失函數關於模型變數的導數；
- 使用優化器更新模型參數以最小化損失函數。

.. literalinclude:: /_static/code/zh-hant/model/text_generation/rnn.py
    :lines: 62-73

關於文本生成的過程有一點需要特別注意。之前，我們一直使用 ``tf.argmax()`` 函數，將對應機率最大的值作為預測值。然而對於文本生成而言，這樣的預測方式過於絕對，會使得生成的文本失去豐富性。於是，我們使用 ``np.random.choice()`` 函數按照生成的機率分佈取樣。這樣，即使是對應機率較小的字元，也有機會被取樣到。同時，我們加入一個 ``temperature`` 參數控制分佈的形狀，參數值越大則分佈越平緩（最大值和最小值的差值越小），生成文本的豐富度越高；參數值越小則分佈越陡峭，生成文本的豐富度越低。

.. literalinclude:: /_static/code/zh-hant/model/text_generation/rnn.py
    :lines: 27-32

通過這種方式進行 “滾雪球” 式的連續預測，即可得到生成文本。

.. literalinclude:: /_static/code/zh-hant/model/text_generation/rnn.py
    :lines: 75-83

生成的文本如下::

    diversity 0.200000:
    conserted and conseive to the conterned to it is a self--and seast and the selfes as a seast the expecience and and and the self--and the sered is a the enderself and the sersed and as a the concertion of the series of the self in the self--and the serse and and the seried enes and seast and the sense and the eadure to the self and the present and as a to the self--and the seligious and the enders

    diversity 0.500000:
    can is reast to as a seligut and the complesed
    has fool which the self as it is a the beasing and us immery and seese for entoured underself of the seless and the sired a mears and everyther to out every sone thes and reapres and seralise as a streed liees of the serse to pease the cersess of the selung the elie one of the were as we and man one were perser has persines and conceity of all self-el

    diversity 1.000000:
    entoles by
    their lisevers de weltaale, arh pesylmered, and so jejurted count have foursies as is
    descinty iamo; to semplization refold, we dancey or theicks-welf--atolitious on his
    such which
    here
    oth idey of pire master, ie gerw their endwit in ids, is an trees constenved mase commars is leed mad decemshime to the mor the elige. the fedies (byun their ope wopperfitious--antile and the it as the f

    diversity 1.200000:
    cain, elvotidue, madehoublesily
    inselfy!--ie the rads incults of to prusely le]enfes patuateded:.--a coud--theiritibaior "nrallysengleswout peessparify oonsgoscess teemind thenry ansken suprerial mus, cigitioum: 4reas. whouph: who
    eved
    arn inneves to sya" natorne. hag open reals whicame oderedte,[fingo is
    zisternethta simalfule dereeg hesls lang-lyes thas quiin turjentimy; periaspedey tomm--whach

.. [#rnn_reference] 此處的任務及實現參考了 https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

.. admonition:: 循環神經網路的工作過程

    循環神經網路是一個處理時間序列資料的神經網路結構，也就是說，我們需要在腦海裡有一根時間軸，循環神經網路具有初始狀態 :math:`s_0` ，在每個時間點 :math:`t` 迭代對當前時間的輸入 :math:`x_t` 進行處理，修改自身的狀態 :math:`s_t` ，並進行輸出 :math:`o_t` 。

    循環神經網路的核心是狀態 :math:`s` ，是一個特定維數的向量，類似於神經網路的 “記憶”。在 :math:`t=0` 的初始時刻，:math:`s_0` 被賦予一個初始值（常用的為全 0 向量）。然後，我們用類似於遞歸的方法來描述循環神經網路的工作過程。即在 :math:`t` 時間，我們假設 :math:`s_{t-1}` 已經求出，注意如何在此基礎上求出 :math:`s_{t}` ：

    - 對輸入向量 :math:`x_t` 通過矩陣 :math:`U` 進行線性變換，:math:`U x_t` 與狀態 s 具有相同的維度；
    - 對 :math:`s_{t-1}` 通過矩陣 :math:`W` 進行線性變換，:math:`W s_{t-1}` 與狀態 s 具有相同的維度；
    - 將上述得到的兩個向量相加並通過激活函數，作為當前狀態 :math:`s_t` 的值，即 :math:`s_t = f(U x_t + W s_{t-1})`。也就是說，當前狀態的值是上一個狀態的值和當前輸入進行某種資訊整合而產生的；
    - 對當前狀態 :math:`s_t` 通過矩陣 :math:`V` 進行線性變換，得到當前時間的輸出 :math:`o_t`。

    .. figure:: /_static/image/model/rnn_cell.jpg
        :align: center

        RNN 工作過程圖示（來自 http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/）

    我們假設輸入向量 :math:`x_t` 、狀態 :math:`s` 和輸出向量 :math:`o_t` 的維度分別為 :math:`m`、:math:`n`、:math:`p`，則 :math:`U \in \mathbb{R}^{m \times n}`、:math:`W \in \mathbb{R}^{n \times n}`、:math:`V \in \mathbb{R}^{n \times p}`。

    上述為最基礎的 RNN 原理介紹。在實際使用時往往使用一些常見的改進型，如LSTM（長短期記憶神經網路，解決了長序列的梯度消失問題，適用於較長的序列）、GRU等。

.. _rl:

深度強化學習（DRL）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`強化學習 <https://zh.wikipedia.org/wiki/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0>`_ （Reinforcement learning，RL）強調如何基於環境而行動，以取得最大化的預期利益。結合了深度學習技術後的強化學習（Deep Reinforcement learning，DRL）更是如虎添翼。近年廣為人知的 AlphaGo 即是深度強化學習的典型應用。

.. note:: 可參考本手冊附錄的 :強化學習簡介 :doc:`../appendix/rl` 一章以獲得強化學習的基礎知識。

這裡，我們使用深度強化學習玩 CartPole（平衡桿）遊戲。平衡桿是控制論中的經典問題，在這個遊戲中，一根桿子的底部與一個小車通過軸相連，而桿的重心在軸之上，因此是一個不穩定的系統。在重力的作用下，桿很容易倒下。而我們則需要控制小車在水平的軌道上進行左右運動，以使得桿一直保持豎直平衡狀態。

.. only:: html

    .. figure:: /_static/image/model/cartpole.gif
        :width: 500
        :align: center

        CartPole 遊戲

.. only:: latex

    .. figure:: /_static/image/model/cartpole.png
        :width: 500
        :align: center

        CartPole 遊戲

我們使用 `OpenAI 推出的 Gym 環境套件 <https://gym.openai.com/>`_ 中的 CartPole 遊戲環境，可使用 ``pip install gym`` 進行安裝，具體安裝步驟和教程可參考 `官方文件 <https://gym.openai.com/docs/>`_ 和 `這裡 <https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-4-gym/>`_ 。和Gym的交互過程很像是一個回合制遊戲，我們首先獲得遊戲的初始狀態（比如桿子的初始角度和小車位置），然後在每個回合t，我們都需要在當前可行的動作中選擇一個並交由Gym執行（比如向左或者向右推動小車，每個回合中二者只能擇一），Gym在執行動作後，會返回動作執行後的下一個狀態和當前回合所獲得的獎勵值（比如我們選擇向左推動小車並執行後，小車位置更加偏左，而桿的角度更加偏右，Gym將新的角度和位置返回給我們。而如果桿在這一回合仍沒有倒下，Gym同時返回給我們一個小的正獎勵）。這個過程可以一直疊代下去，直到遊戲結束（比如桿子倒下）。在 Python 中，Gym 的基本呼叫方法如下：

.. code-block:: python

    import gym

    env = gym.make('CartPole-v1')       # 實例化一個遊戲環境，參數為遊戲名稱
    state = env.reset()                 # 初始化環境，獲得初始狀態
    while True:
        env.render()                    # 對當前影像進行渲染，繪圖到螢幕
        action = model.predict(state)   # 假設我們有一個訓練好的模型，能夠通過當前狀態預測出這時應該進行的動作
        next_state, reward, done, info = env.step(action)   # 讓環境執行動作，獲得執行完動作的下一個狀態，動作的獎勵，遊戲是否已結束以及額外資訊
        if done:                        # 如果遊戲結束則退出循環
            break

那麼，我們的任務就是訓練出一個模型，能夠根據當前的狀態預測出應該進行的一個好的動作。大致上來說，一個好的動作應當能夠最大化整個遊戲過程中獲得的獎勵之和，這也是強化學習的目標。以CartPole遊戲為例，我們的目標是希望做出合適的動作使得桿一直不倒，即遊戲交互的回合數盡可能地多。而回合每進行一次，我們都會獲得一個小的正獎勵，回合數越多則累積的獎勵值也越高。因此，我們最大化遊戲過程中的獎勵之和與我們的最終目標是一致的。

以下程式碼展示了如何使用深度強化學習中的 Deep Q-Learning 方法 [Mnih2013]_ 來訓練模型。首先，我們引入TensorFlow、Gym和一些常用套件，並定義一些模型超參數：

.. literalinclude:: /_static/code/zh-hant/model/rl/rl.py
    :lines: 1-14

然後，我們使用 ``tf.keras.Model`` 建立一個Q函數網路（Q-network），用於擬合Q Learning中的Q函數。這裡我們使用較簡單的多層全連接神經網路進行擬合。該網路輸入當前狀態，輸出各個動作下的Q-value（CartPole下為2維，即向左和向右推動小車）。

.. literalinclude:: /_static/code/zh-hant/model/rl/rl.py
    :lines: 16-31

最後，我們在主程式中實現Q Learning演算法。

.. literalinclude:: /_static/code/zh-hant/model/rl/rl.py
    :lines: 34-82

對於不同的任務（或者說環境），我們需要根據任務的特點，設計不同的狀態以及採取合適的網路來擬合 Q 函數。例如，如果我們考慮經典的打磚塊游戲（Gym 環境套件中的  `Breakout-v0 <https://gym.openai.com/envs/Breakout-v0/>`_ ），每一次執行動作（擋板向左、向右或不動），都會返回一個 ``210 * 160 * 3`` 的 RGB 圖片，表示當前影像畫面。為了讓打磚塊遊戲這個任務設計合適的狀態表示，我們有以下分析：

* 磚塊的顏色資訊並不是很重要，畫面轉換成灰階也不影響操作，因此可以去除狀態中的顏色資訊（即將圖片轉為灰階表示）；
* 小球移動的資訊很重要，如果只知道單一張影像畫面而不知道小球往哪邊移動，即使是人也很難判斷擋板應當移動的方向。因此，必須在狀態中加入小球運動方向的資訊。一個簡單的方式是將當前影像與前面幾張影像畫面進行疊加，得到一個 ``210 * 160 * X`` （X 為疊加影格數）的狀態表示；
* 每個影格數的分辨率不需要特別高，只要能大致瞭解方塊、小球和擋板的位置以做出決策即可，因此對於每張影像的長寬可做適當壓縮。

而考慮到我們需要從圖片資訊中提取特徵，使用 CNN 作為擬合 Q 函數的網路將更為適合。由此，將上面的 ``QNetwork`` 更換為 CNN 網路，並對狀態做一些修改，即可用於玩一些簡單的影像遊戲。

..
    .. admonition:: 深度強化學習原理初探

        與前面所介紹的捲積神經網路和循環神經網路不同，強化學習（Reinforcement Learning）是一種學習演算法的類型。

        TODO

Keras Pipeline *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://medium.com/tensorflow/what-are-symbolic-and-imperative-apis-in-tensorflow-2-0-dfccecb01021
    https://www.tensorflow.org/beta/guide/keras/overview
    https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models

以上範例均使用了 Keras 的 Subclassing API 建立模型，即對 ``tf.keras.Model`` 類進行擴展以定義自己的新模型，同時人工編寫了訓練和評估模型的流程。這種方式靈活度高，且與其他流行的深度學習框架（如 PyTorch、Chainer）共通，是本手冊所推薦的方法。不過在很多時候，我們只需要建立一個結構相對簡單和典型的神經網路（比如上文中的 MLP 和 CNN），並使用常見的手法進行訓練。這時，Keras 也給我們提供了另一套更為簡單高效的內建方法來建立、訓練和評估模型。

.. _sequential_functional:

Keras Sequential/Functional API 模式建立模型
-------------------------------------------

最典型和常用的神經網路結構是將一堆層按特定順序疊加起來，那麼，我們是不是只需要提供一個層的列表，就能由 Keras 將它們自動首尾相連，形成模型呢？Keras 的 Sequential API 正是如此。通過向 ``tf.keras.models.Sequential()`` 提供一個層的列表，就能快速地建立一個 ``tf.keras.Model`` 模型並返回：

.. literalinclude:: /_static/code/zh-hant/model/mnist/main.py
    :lines: 18-23

不過，這種層疊結構並不能表示任意的神經網路結構。為此，Keras 提供了 Functional API，幫助我們建立更為複雜的模型，例如多輸入 / 輸出或存在參數共用的模型。其使用方法是將層作為可調用的對象並返回張量（這點與之前章節的使用方法一致），並將輸入向量和輸出向量提供給 ``tf.keras.Model`` 的 ``inputs`` 和 ``outputs`` 參數，範例如下：

.. literalinclude:: /_static/code/zh-hant/model/mnist/main.py
    :lines: 25-30
..
    https://www.tensorflow.org/alpha/guide/keras/functional

使用 Keras Model 的 ``compile`` 、 ``fit`` 和 ``evaluate`` 方法訓練和評估模型
--------------------------------------------------------------------------------------

當模型建立完成後，通過 ``tf.keras.Model`` 的 ``compile`` 方法配置訓練過程：

.. literalinclude:: /_static/code/zh-hant/model/mnist/main.py
    :lines: 84-88

``tf.keras.Model.compile`` 接受 3 個重要的參數：

 - ``oplimizer`` ：優化器，可從 ``tf.keras.optimizers`` 中選擇；
 - ``loss`` ：損失函數，可從 ``tf.keras.losses`` 中選擇；
 - ``metrics`` ：評量指標，可從 ``tf.keras.metrics`` 中選擇。

接下來，可以使用 ``tf.keras.Model`` 的 ``fit`` 方法訓練模型：

.. literalinclude:: /_static/code/zh-hant/model/mnist/main.py
    :lines: 89

``tf.keras.Model.fit`` 接受 5 個重要的參數：

 - ``x`` ：訓練資料；
 - ``y`` ：目標資料（資料標籤）；
 - ``epochs`` ：將訓練資料疊代多少遍；
 - ``batch_size`` ：批次的大小；
 - ``validation_data`` ：驗證資料，可用於在訓練過程中監控模型的性能。

Keras 支援使用 ``tf.data.Dataset`` 進行訓練，詳見 :ref:`tf.data <tfdata>` 。

最後，使用 ``tf.keras.Model.evaluate`` 評估訓練效果，提供測試資料及標籤即可：

.. literalinclude:: /_static/code/zh-hant/model/mnist/main.py
    :lines: 90

..
    https://www.tensorflow.org/beta/guide/keras/training_and_evaluation

自定義層、損失函數和評量指標 *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可能你還會問，如果現有的這些層無法滿足我的要求，我需要定義自己的層怎麼辦？事實上，我們不僅可以繼承 ``tf.keras.Model`` 編寫自己的模型類，也可以繼承 ``tf.keras.layers.Layer`` 編寫自己的層。

.. _custom_layer:

自定義層
-------------------------------------------

自定義層需要繼承 ``tf.keras.layers.Layer`` 類，並覆寫 ``__init__`` 、 ``build`` 和 ``call`` 三個方法，如下所示：

.. code-block:: python

    class MyLayer(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            # 初始化程式碼

        def build(self, input_shape):     # input_shape 是一個 TensorShape 類型對象，提供輸入的形狀
            # 在第一次使用該層的時候呼叫該部分程式碼，在這裡創建變數可以使得變數的形狀自適應輸入
            # 而不需要使用者額外指定變數形狀。
            # 如果已經可以完全確定變數的形狀，也可以在__init__部分創建變數
            self.variable_0 = self.add_weight(...)
            self.variable_1 = self.add_weight(...)

        def call(self, inputs):
            # 模型呼叫的程式碼（處理輸入並返回輸出）
            return output

例如，如果我們要自己實現一個 :ref:`本章第一節 <linear>` 中的全連接層（ ``tf.keras.layers.Dense`` ），可以按以下方式編寫。此程式碼在 ``build`` 方法中創建兩個變數，並在 ``call`` 方法中使用創建的變數進行運算：

.. literalinclude:: /_static/code/zh-hant/model/custom/linear.py
    :lines: 9-22

在定義模型的時候，我們便可以如同 Keras 中的其他層一樣，呼叫我們自定義的層 ``LinearLayer``：

.. literalinclude:: /_static/code/zh-hant/model/custom/linear.py
    :lines: 25-32

自定義損失函數和評量指標
-------------------------------------------

..
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/losses/Loss

自定義損失函數需要繼承 ``tf.keras.losses.Loss`` 類別，重寫 ``call`` 方法即可，輸入真實值 ``y_true`` 和模型預測值 ``y_pred`` ，輸出模型預測值和真實值之間通過自定義的損失函數計算出的損失值。下面的範例為均方差(Mean-Square Error, MSE）損失函數：

.. code-block:: python

    class MeanSquaredError(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            return tf.reduce_mean(tf.square(y_pred - y_true))

..
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/metrics/Metric

自定義評量指標需要繼承 ``tf.keras.metrics.Metric`` 類，並重寫 ``__init__`` 、 ``update_state`` 和 ``result`` 三個方法。下面的範例對前面用到的 ``SparseCategoricalAccuracy`` 評量指標類別做了一個簡單的重實現：

.. literalinclude:: /_static/code/zh-hant/model/utils.py
    :lines: 22-34

.. [LeCun1998] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998. http://yann.lecun.com/exdb/mnist/
.. [Graves2013] Graves, Alex. “Generating Sequences With Recurrent Neural Networks.” ArXiv:1308.0850 [Cs], August 4, 2013. http://arxiv.org/abs/1308.0850.
.. [Mnih2013] Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. “Playing Atari with Deep Reinforcement Learning.” ArXiv:1312.5602 [Cs], December 19, 2013. http://arxiv.org/abs/1312.5602.

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 190 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>