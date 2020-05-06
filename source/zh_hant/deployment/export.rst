TensorFlow模型導出
====================================================

.. _savedmodel:

使用SavedModel完整導出模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/beta/guide/saved_model

在部署模型時，我們的第一步往往是將訓練好的整個模型完整導出爲一系列標準格式的文件，然後即可在不同的平台上部署模型文件。這時，TensorFlow爲我們提供了SavedModel這一格式。與前面介紹的Checkpoint不同，SavedModel包含了一個TensorFlow程序的完整信息： **不僅包含參數的權值，還包含計算的流程（即計算圖）** 。當模型導出爲SavedModel文件時，無需建立模型的原始碼即可再次運行模型，這使得SavedModel尤其適用於模型的分享和部署。後文的TensorFlow Serving（伺服器端部署模型）、TensorFlow Lite（移動端部署模型）以及TensorFlow.js都會用到這一格式。

Keras模型均可方便地導出爲SavedModel格式。不過需要注意的是，因爲SavedModel基於計算圖，所以對於使用繼承 ``tf.keras.Model`` 類建立的Keras模型，其需要導出到SavedModel格式的方法（比如 ``call`` ）都需要使用 ``@tf.function`` 修飾（ ``@tf.function`` 的使用方式見 :ref:`前文 <tffunction>` ）。然後，假設我們有一個名爲 ``model`` 的Keras模型，使用下面的代碼即可將模型導出爲SavedModel：

.. code-block:: python

    tf.saved_model.save(model, "保存的目標文件夾名稱")

在需要載入SavedModel文件時，使用

.. code-block:: python

    model = tf.saved_model.load("保存的目標文件夾名稱")

即可。

.. hint:: 對於使用繼承 ``tf.keras.Model`` 類建立的Keras模型 ``model`` ，使用SavedModel載入後將無法使用 ``model()`` 直接進行推斷，而需要使用 ``model.call()`` 。

以下是一個簡單的示例，將 :ref:`前文MNIST手寫體識別的模型 <mlp>` 進行導出和導入。

導出模型到 ``saved/1`` 文件夾：

.. literalinclude:: /_static/code/zh/savedmodel/keras/train_and_export.py
    :emphasize-lines: 22

將 ``saved/1`` 中的模型導入並測試性能：

.. literalinclude:: /_static/code/zh/savedmodel/keras/load_savedmodel.py
    :emphasize-lines: 6, 12

輸出::

    test accuracy: 0.952000

使用繼承 ``tf.keras.Model`` 類建立的Keras模型同樣可以以相同方法導出，唯須注意 ``call`` 方法需要以 ``@tf.function`` 修飾，以轉化爲SavedModel支持的計算圖，代碼如下：

.. code-block:: python
    :emphasize-lines: 8

    class MLP(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
            self.dense2 = tf.keras.layers.Dense(units=10)

        @tf.function
        def call(self, inputs):         # [batch_size, 28, 28, 1]
            x = self.flatten(inputs)    # [batch_size, 784]
            x = self.dense1(x)          # [batch_size, 100]
            x = self.dense2(x)          # [batch_size, 10]
            output = tf.nn.softmax(x)
            return output

    model = MLP()
    ...

模型導入並測試性能的過程也相同，唯須注意模型推斷時需要顯式調用 ``call`` 方法，即使用：

.. code-block:: python
    :emphasize-lines: 2

        ...
        y_pred = model.call(data_loader.test_data[start_index: end_index])
        ...

Keras Sequential save方法（Jinpeng）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我們以keras模型訓練和保存爲例進行講解，如下是keras官方的mnist模型訓練樣例。

源碼地址::

    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

以上代碼，是基於keras的Sequential構建了多層的卷積神經網絡，並進行訓練。

爲了方便起見可使用如下命令拷貝到本地::

    curl -LO https://raw.githubusercontent.com/keras-team/keras/master/examples/mnist_cnn.py

然後，在最後加上如下一行代碼（主要是對keras訓練完畢的模型進行保存）::

    model.save('mnist_cnn.h5')


在終端中執行mnist_cnn.py文件，如下::

    python mnist_cnn.py

.. warning:: 該過程需要連接網絡獲取 ``mnist.npz`` 文件（https://s3.amazonaws.com/img-datasets/mnist.npz），會被保存到 ``$HOME/.keras/datasets/`` 。如果網絡連接存在問題，可以通過其他方式獲取 ``mnist.npz`` 後，直接保存到該目錄即可。

執行過程會比較久，執行結束後，會在當前目錄產生 ``mnist_cnn.h5`` 文件（HDF5格式），就是keras訓練後的模型，其中已經包含了訓練後的模型結構和權重等信息。

在伺服器端，可以直接通過 ``keras.models.load_model("mnist_cnn.h5")`` 加載，然後進行推理；在行動裝置需要將HDF5模型文件轉換爲TensorFlow Lite的格式，然後通過相應平台的Interpreter加載，然後進行推理。

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 192 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>