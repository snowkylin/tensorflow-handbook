TensorFlow Serving
==========================

當我們將模型訓練完畢後，往往需要將模型在生產環境中部署。最常見的方式，是在服務器上提供一個API，即客戶機向服務器的某個API發送特定格式的請求，服務器收到請求數據後通過模型進行計算，並返回結果。如果僅僅是做一個Demo，不考慮高並發和性能問題，其實配合 `Flask <https://palletsprojects.com/p/flask/>`_ 等Python下的Web框架就能非常輕鬆地實現服務器API。不過，如果是在真的實際生產環境中部署，這樣的方式就顯得力不從心了。這時，TensorFlow爲我們提供了TensorFlow Serving這一組件，能夠幫助我們在實際生產環境中靈活且高性能地部署機器學習模型。

TensorFlow Serving安裝
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow Serving可以使用apt-get或Docker安裝。在生產環境中，推薦 `使用Docker部署TensorFlow Serving <https://www.tensorflow.org/tfx/serving/docker>`_ 。不過此處出於教學目的，介紹依賴環境較少的 `apt-get安裝 <https://www.tensorflow.org/tfx/serving/setup#installing_using_apt>`_ 。

.. warning:: 軟件的安裝方法往往具有時效性，本節的更新日期爲2019年8月。若遇到問題，建議參考 `TensorFlow網站上的最新安裝說明 <https://www.tensorflow.org/tfx/serving/setup>`_ 進行操作。

首先設置安裝源：

::

    # 添加Google的TensorFlow Serving源
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
    # 添加gpg key
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

更新源後，即可使用apt-get安裝TensorFlow Serving

::

    sudo apt-get update
    sudo apt-get install tensorflow-model-server

.. hint:: 在運行curl和apt-get命令時，可能需要設置代理。

    curl設置代理的方式爲 ``-x`` 選項或設置 ``http_proxy`` 環境變量，即

    ::

        export http_proxy=http://代理服務器IP:端口

    或

    ::

        curl -x http://代理服務器IP:端口 URL

    apt-get設置代理的方式爲 ``-o`` 選項，即

    ::

        sudo apt-get -o Acquire::http::proxy="http://代理服務器IP:端口" ...

    Windows 10下，可以在 `Linux子系統（WSL） <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ 內使用相同的方式安裝TensorFlow Serving。

TensorFlow Serving模型部署
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow Serving可以直接讀取SavedModel格式的模型進行部署（導出模型到SavedModel文件的方法見 :ref:`前文 <savedmodel>` ）。使用以下命令即可：

::

    tensorflow_model_server \
        --rest_api_port=端口號（如8501） \
        --model_name=模型名 \
        --model_base_path="SavedModel格式模型的文件夾絕對地址（不含版本號）"

.. note:: TensorFlow Serving支持熱更新模型，其典型的模型文件夾結構如下：

    ::
        
        /saved_model_files
            /1      # 版本號爲1的模型文件
                /assets
                /variables
                saved_model.pb
            ...
            /N      # 版本號爲N的模型文件
                /assets
                /variables
                saved_model.pb

    
    上面1~N的子文件夾代表不同版本號的模型。當指定 ``--model_base_path`` 時，只需要指定根目錄的 **絕對地址** （不是相對地址）即可。例如，如果上述文件夾結構存放在 ``home/snowkylin`` 文件夾內，則 ``--model_base_path`` 應當設置爲 ``home/snowkylin/saved_model_files`` （不附帶模型版本號）。TensorFlow Serving會自動選擇版本號最大的模型進行載入。 

Keras Sequential模式模型的部署
---------------------------------------------------

由於Sequential模式的輸入和輸出都很固定，因此這種類型的模型很容易部署，無需其他額外操作。例如，要將 :ref:`前文使用SavedModel導出的MNIST手寫體識別模型 <savedmodel>` （使用Keras Sequential模式建立）以 ``MLP`` 的模型名在 ``8501`` 端口進行部署，可以直接使用以下命令：

::

    tensorflow_model_server \
        --rest_api_port=8501 \
        --model_name=MLP \
        --model_base_path="/home/.../.../saved"  # 文件夾絕對地址根據自身情況填寫，無需加入版本號

然後就可以按照 :ref:`後文的介紹 <call_serving_api>` ，使用gRPC或者RESTful API在客戶端調用模型了。

自定義Keras模型的部署
---------------------------------------------------

使用繼承 ``tf.keras.Model`` 類建立的自定義Keras模型的自由度相對更高。因此當使用TensorFlow Serving部署模型時，對導出的SavedModel文件也有更多的要求：

- 需要導出到SavedModel格式的方法（比如 ``call`` ）不僅需要使用 ``@tf.function`` 修飾，還要在修飾時指定 ``input_signature`` 參數，以顯式說明輸入的形狀。該參數傳入一個由 ``tf.TensorSpec`` 組成的列表，指定每個輸入張量的形狀和類型。例如，對於MNIST手寫體數字識別，我們的輸入是一個 ``[None, 28, 28, 1]`` 的四維張量（ ``None`` 表示第一維即Batch Size的大小不固定），此時我們可以將模型的 ``call`` 方法做以下修飾：

.. code-block:: python
    :emphasize-lines: 4

    class MLP(tf.keras.Model):
        ...

        @tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)])
        def call(self, inputs):
            ...

- 在將模型使用 ``tf.saved_model.save`` 導出時，需要通過 ``signature`` 參數提供待導出的函數的簽名（Signature）。簡單說來，由於自定義的模型類里可能有多個方法都需要導出，因此，需要告訴TensorFlow Serving每個方法在被客戶端調用時分別叫做什麼名字。例如，如果我們希望客戶端在調用模型時使用 ``call`` 這一簽名來調用 ``model.call`` 方法時，我們可以在導出時傳入 ``signature`` 參數，以 ``dict`` 的鍵值對形式告知導出的方法對應的簽名，代碼如下：

.. code-block:: python
    :emphasize-lines: 3

    model = MLP()
    ...
    tf.saved_model.save(model, "saved_with_signature/1", signatures={"call": model.call})

以上兩步均完成後，即可使用以下命令部署：

::

    tensorflow_model_server \
        --rest_api_port=8501 \
        --model_name=MLP \
        --model_base_path="/home/.../.../saved_with_signature"  # 修改爲自己模型的絕對地址

.. _call_serving_api:

在客戶端調用以TensorFlow Serving部署的模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/tfx/serving/api_rest
    http://www.ruanyifeng.com/blog/2014/05/restful_api.html

TensorFlow Serving支持以gRPC和RESTful API調用以TensorFlow Serving部署的模型。本手冊主要介紹較爲通用的RESTful API方法。

RESTful API以標準的HTTP POST方法進行交互，請求和回復均爲JSON對象。爲了調用服務器端的模型，我們在客戶端向服務器發送以下格式的請求：

服務器URI： ``http://服務器地址:端口號/v1/models/模型名:predict`` 

請求內容：

::

    {
        "signature_name": "需要調用的函數簽名（Sequential模式不需要）",
        "instances": 輸入數據
    }

回復爲：

::

    {
        "predictions": 返回值
    }

Python客戶端示例
------------------------------------------------------

以下示例使用 `Python的Requests庫 <https://2.python-requests.org//zh_CN/latest/user/quickstart.html>`_ （你可能需要使用 ``pip install requests`` 安裝該庫）向本機的TensorFlow Serving服務器發送MNIST測試集的前10幅圖像並返回預測結果，同時與測試集的真實標籤進行比較。

.. literalinclude:: /_static/code/zh/savedmodel/keras/client.py

輸出：

::

    [7 2 1 0 4 1 4 9 6 9]
    [7 2 1 0 4 1 4 9 5 9]

可見預測結果與真實標籤值非常接近。

對於自定義的Keras模型，在發送的數據中加入 ``signature_name`` 鍵值即可，即將上面代碼的 ``data`` 建立過程改爲

.. literalinclude:: /_static/code/zh/savedmodel/custom/client.py
    :lines: 8-11

Node.js客戶端示例（Ziyang）
------------------------------------------------------

以下示例使用 `Node.js <https://nodejs.org/zh-cn/>`_ 將下圖轉換爲28*28的灰度圖，發送給本機的TensorFlow Serving服務器，並輸出返回的預測值和概率。（其中使用了 `圖像處理庫jimp <https://github.com/oliver-moran/jimp>`_ 和 `HTTP庫superagent <https://visionmedia.github.io/superagent/>`_ ，可使用 ``npm install jimp`` 和 ``npm install superagent`` 安裝）

.. figure:: /_static/image/serving/test_pic_tag_5.png
    :align: center

    ``test_pic_tag_5.png`` ：一個由作者手寫的數字5。（運行下面的代碼時可下載該圖片並放在與代碼同一目錄下）

.. literalinclude:: /_static/code/zh/savedmodel/keras/client.js
    :language: javascript

運行結果爲：

::

    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1               1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1                 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1       1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1       1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1     1                 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1                         1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1         1 1 1 1 1 1     1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1       1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1         1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1         1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1     1 1 1         1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1                 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1         1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    我們猜這個數字是5，概率是0.846008837

可見輸出結果符合預期。

.. note:: 如果你不熟悉HTTP POST，可以參考 `這裡 <https://www.runoob.com/tags/html-httpmethods.html>`_ 。事實上，當你在用瀏覽器填寫表單（比方說性格測試）並點擊「提交」按鈕，然後獲得返回結果（比如說「你的性格是ISTJ」）時，就很有可能是在向服務器發送一個HTTP POST請求並獲得了服務器的回覆。

    RESTful API是一個流行的API設計理論，可以參考 `這裡 <http://www.ruanyifeng.com/blog/2014/05/restful_api.html>`_ 獲得簡要介紹。

    關於TensorFlow Serving的RESTful API的完整使用方式可參考 `文檔 <https://www.tensorflow.org/tfx/serving/api_rest>`_ 。

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 193 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>

