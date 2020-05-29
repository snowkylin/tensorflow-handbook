TensorFlow Serving
==========================

當我們將模型訓練完畢後，往往需要將模型在開發環境中部署。最常見的方式是在伺服器上提供一個 API，即客戶端向伺服器的某個 API 發送特定格式的請求，伺服器收到請求資料後透過模型進行計算，並返回結果。如果只是做一個 Demo，不考慮並發性和性能問題，其實配合 `Flask <https://palletsprojects.com/p/flask/>`_ 等 Python 下的 Web 框架就能非常輕鬆地實現伺服器 API。不過，如果是在真的實際開發環境中部署，這樣的方式就顯得力不從心了。這時，TensorFlow 為我們提供了 TensorFlow Serving 這一組件，能夠幫助我們在實際開發環境中靈活且高性能地部署機器學習模型。

TensorFlow Serving 安裝
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow Serving 可以使用 apt-get 或 Docker安裝。在開發環境中，推薦 `使用Docker部署TensorFlow Serving <https://www.tensorflow.org/tfx/serving/docker>`_ 。不過此處出於教學目的，介紹環境依賴較少的 `apt-get安裝 <https://www.tensorflow.org/tfx/serving/setup#installing_using_apt>`_ 。

.. warning:: 軟體的安裝方法往往具有時效性，本節的更新日期為2019年8月。若遇到問題，建議參考 `TensorFlow網站上的最新安裝說明 <https://www.tensorflow.org/tfx/serving/setup>`_ 進行操作。

首先設置安裝來源：

::

    # 添加Google的TensorFlow Serving安裝來源
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
    # 添加gpg key
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

更新安裝來源後，即可使用 apt-get 安裝 TensorFlow Serving

::

    sudo apt-get update
    sudo apt-get install tensorflow-model-server

.. hint:: 在執行 curl 和 apt-get 命令時，可能需要設置代理伺服器。

    curl 設置代理伺服器的方式為 ``-x`` 選項或設置 ``http_proxy`` 環境變量，即

    ::

        export http_proxy=http://代理伺服器IP:埠號

    或

    ::

        curl -x http://代理伺服器IP:埠號 URL

    apt-get設置代理的方式為 ``-o`` 選項，即

    ::

        sudo apt-get -o Acquire::http::proxy="http://代理伺服器IP:埠號" ...

    Windows 10 下，可以在 `Linux子系統（WSL） <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ 使用相同的方式安裝 TensorFlow Serving。

TensorFlow Serving 模型部署
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow Serving 可以直接讀取 SavedModel 格式的模型進行部署（匯出模型到 SavedModel 文件的方法見 :ref:`前文 <savedmodel>` ）。使用以下指令即可：

::

    tensorflow_model_server \
        --rest_api_port=埠號（如8501） \
        --model_name=模型名 \
        --model_base_path="SavedModel格式模型的資料夾絕對地址（不含版本號碼）"

.. note:: TensorFlow Serving 支援即時更新模型，其典型的模型資料夾結構如下：

    ::
        
        /saved_model_files
            /1      # 版本號碼為1的模型文件
                /assets
                /variables
                saved_model.pb
            ...
            /N      # 版本號碼為N的模型文件
                /assets
                /variables
                saved_model.pb

    
    上面 1~N 的子資料夾代表不同版本號的模型。當指定 ``--model_base_path`` 時，只需要指定根目錄的 **絕對路徑** （不是相對路徑）即可。例如，如果上述資料夾結構存放在 ``home/snowkylin`` 資料夾內，則 ``--model_base_path`` 應該設置為 ``home/snowkylin/saved_model_files`` （不附帶模型版本號碼）。TensorFlow Serving 會自動選擇版本號碼最大的模型進行載入。 

Keras Sequential 模式模型的部署
---------------------------------------------------

由於 Sequential 模式的輸入和輸出都很固定，因此這種類型的模型很容易部署，不需要其他額外操作。例如，要將 :ref:`前文使用 SavedModel 匯出的 MNIST 手寫體識別模型 <savedmodel>` （使用Keras Sequential模式建立）以 ``MLP`` 的模型名在埠號 ``8501`` 進行部署，可以直接使用以下指令：

::

    tensorflow_model_server \
        --rest_api_port=8501 \
        --model_name=MLP \
        --model_base_path="/home/.../.../saved"  # 資料夾絕對路徑根據自身情況填寫，無需加入版本號碼

然後就可以按照 :ref:`後文的介紹 <call_serving_api>` ，使用 gRPC 或者 RESTful API 在客戶端呼叫模型了。

自定義 Keras 模型的部署
---------------------------------------------------

使用繼承 ``tf.keras.Model`` 類建立的自定義 Keras 模型的自由度相對更高。因此使用 TensorFlow Serving 部署模型時，對導出的 SavedModel 文件也有更多的要求：

- 匯出 SavedModel 格式的方法（比如 ``call`` ）不僅需要使用 ``@tf.function`` 修飾，還要在修飾時指定 ``input_signature`` 參數，以說明輸入的形狀。該參數傳入一個由 ``tf.TensorSpec`` 組成的列表，指定每個輸入張量的形狀和類型。例如，對於 MNIST 手寫體數字辨識，我們的輸入是一個 ``[None, 28, 28, 1]`` 的四維張量（ ``None`` 表示第一維即 Batch Size 的大小不固定），此時我們可以將模型的 ``call`` 方法做以下修飾：

.. code-block:: python
    :emphasize-lines: 4

    class MLP(tf.keras.Model):
        ...

        @tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)])
        def call(self, inputs):
            ...

- 將模型使用 ``tf.saved_model.save`` 匯出時，需要通過 ``signature`` 參數提供待匯出的函數的簽名（Signature）。簡單說來，由於自定義的模型類別裡可能有多個方法都需要匯出，因此，需要告訴 TensorFlow Serving 每個方法在被客戶端呼叫時分別叫做什麼名字。例如，如果我們希望客戶端在呼叫模型時使用 ``call`` 這一簽名來呼叫 ``model.call`` 方法時，我們可以在匯出時傳入 ``signature`` 參數，以 ``dict`` 的形式告知匯出的方法對應的名稱，程式碼如下：

.. code-block:: python
    :emphasize-lines: 3

    model = MLP()
    ...
    tf.saved_model.save(model, "saved_with_signature/1", signatures={"call": model.call})

以上兩步均完成後，即可使用以下指令部署：

::

    tensorflow_model_server \
        --rest_api_port=8501 \
        --model_name=MLP \
        --model_base_path="/home/.../.../saved_with_signature"  # 修改為自己模型的絕對地址

.. _call_serving_api:

在客戶端呼叫以 TensorFlow Serving 部署的模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/tfx/serving/api_rest
    http://www.ruanyifeng.com/blog/2014/05/restful_api.html

TensorFlow Serving 支援以 gRPC和RESTful API 調用以 TensorFlow Serving 部署的模型。本手冊主要介紹較為通用的 RESTful API 方法。

RESTful API 以標準的 HTTP POST 方法進行通信，請求和回覆均為 JSON 對象。為了呼叫伺服器端的模型，我們在客戶端向伺服器發送以下格式的請求：

伺服器 URI： ``http://伺服器地址:埠號/v1/models/模型名:predict`` 

請求內容：

::

    {
        "signature_name": "需要呼叫的函數簽名（Sequential模式不需要）",
        "instances": 輸入資料
    }

回覆為：

::

    {
        "predictions": 返回值
    }

Python 客戶端範例
------------------------------------------------------

以下範例使用 `Python的Requests庫 <https://requests.readthedocs.io/en/master/>`_ （你可能需要使用 ``pip install requests`` 安裝該函式庫）向本機的TensorFlow Serving 伺服器發送 MNIST 測試集的前 10 幅圖像並返回預測結果，同時與測試集的真實標籤進行比較。

.. literalinclude:: /_static/code/zh-hant/savedmodel/keras/client.py

輸出：

::

    [7 2 1 0 4 1 4 9 6 9]
    [7 2 1 0 4 1 4 9 5 9]

可以發現，預測結果與真實標籤值非常接近。

對於自定義的 Keras 模型，在發送的資料中加入 ``signature_name`` 鍵值即可，將上面程式碼 ``data`` 建立過程改為

.. literalinclude:: /_static/code/zh-hant/savedmodel/custom/client.py
    :lines: 8-11

Node.js客戶端範例（Ziyang）
------------------------------------------------------

以下範例使用 `Node.js <https://nodejs.org/zh-tw/>`_ 將下圖轉換為 28*28 的灰階圖，發送給本機的 TensorFlow Serving 伺服器，並輸出返回的預測結果和機率。（其中使用了 `圖像處理庫jimp <https://github.com/oliver-moran/jimp>`_ 和 `HTTP庫superagent <https://visionmedia.github.io/superagent/>`_ ，可使用 ``npm install jimp`` 和 ``npm install superagent`` 安裝）

.. figure:: /_static/image/serving/test_pic_tag_5.png
    :align: center

    ``test_pic_tag_5.png`` ：一個由作者手寫的數字5。（運行下麵的代碼時可下載該圖片並放在與代碼同一目錄下）

.. literalinclude:: /_static/code/zh-hant/savedmodel/keras/client.js
    :language: javascript

執行結果為：

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
    我們猜這個數字是5，機率是0.846008837

可以發現、輸出結果符合預期。

.. note:: 如果你不熟悉 HTTP POST，可以參考 `這裡 <https://blog.toright.com/posts/1203/%E6%B7%BA%E8%AB%87-http-method%EF%BC%9A%E8%A1%A8%E5%96%AE%E4%B8%AD%E7%9A%84-get-%E8%88%87-post-%E6%9C%89%E4%BB%80%E9%BA%BC%E5%B7%AE%E5%88%A5%EF%BC%9F.html>`_ 。事實上，當你在用瀏覽器填寫表單（比方說性格測試）並點擊“提交”按鈕，然後獲得返回結果（比如說“你的性格是 ISTJ”）時，就很有可能是在向伺服器發送一個 HTTP POST 請求並獲得了伺服器的回覆。

    RESTful API 是一個流行的 API 設計理論，可以參考 `這裡 <https://medium.com/itsems-frontend/api-%E6%98%AF%E4%BB%80%E9%BA%BC-restful-api-%E5%8F%88%E6%98%AF%E4%BB%80%E9%BA%BC-a001a85ab638>`_ 查看簡介。

    關於 TensorFlow Serving 的 RESTful API 的完整使用方式可參考 `文件 <https://www.tensorflow.org/tfx/serving/api_rest>`_ 。

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

