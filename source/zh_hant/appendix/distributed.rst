TensorFlow分布式訓練
============================================

..
    https://www.tensorflow.org/beta/guide/distribute_strategy

當我們擁有大量計算資源時，透過使用合適的分散式策略，我們可以充分利用這些計算資源，大幅壓縮模型訓練的時間。針對不同的使用場景，TensorFlow 在 ``tf.distribute.Strategy`` 中為我們提供了許多種分散式策略，使得我們能夠更高效的訓練模型。

.. _zh_hant_multi_gpu:

單機多卡訓練： ``MirroredStrategy`` 
-------------------------------------------

..
    https://www.tensorflow.org/beta/tutorials/distribute/keras
    https://juejin.im/post/5ba9d72ff265da0ac849384b
    https://www.codercto.com/a/86644.html

``tf.distribute.MirroredStrategy`` 是一種簡單且高性能的，資料並行的同步式分散式策略，主要支援多個 GPU 在同一台主機上訓練。使用這種策略時，我們只需實例化一個 ``MirroredStrategy`` 策略::


    strategy = tf.distribute.MirroredStrategy()

並將模型建構的程式碼放入 ``strategy.scope()`` 的上下文環境中::

    with strategy.scope():
        # 模型建構程式碼

.. tip:: 可以在參數中指定設備，如::

        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    
    即指定只使用第0、1號GPU參與分散式策略。
    
以下程式碼展示了使用 ``MirroredStrategy`` 策略，在 :doc:`TensorFlow Datasets <../appendix/tfds>` 中的部分圖像資料集上使用 Keras 訓練 MobileNetV2 的過程：

.. literalinclude:: /_static/code/zh-hant/distributed/multi_gpu.py
    :emphasize-lines: 8-10, 21

在以下的測試中，我們使用同一台主機上的 4 塊 NVIDIA GeForce GTX 1080 Ti 顯卡進行單機多卡的模型訓練。所有測試的 epoch 數均為 5。使用單機無分散式配置時，雖然機器依然具有 4 塊顯卡，但程式不使用分散式的設置，直接進行訓練，Batch Size 設置為 64。使用單機四卡時，測試總 Batch Size 為 64（分發到單台機器的 Batch Size 為 16）和總 Batch Size 為 256（分發到單台機器的 Batch Size 為 64）兩種情況。

============  ==============================  ==============================  =============================
資料集        單機無分散式（Batch Size爲64）  單機四卡（總Batch Size爲64）    單機四卡（總Batch Size爲256）
============  ==============================  ==============================  =============================
cats_vs_dogs  146s/epoch                      39s/epoch                       29s/epoch
tf_flowers    22s/epoch                       7s/epoch                        5s/epoch
============  ==============================  ==============================  =============================

可見，使用 MirroredStrategy 後，模型訓練的速度有了大幅度的提高。在所有顯卡性能接近的情況下，訓練時間與顯卡的數目接近於反比關係。

.. admonition:: ``MirroredStrategy`` 過程簡介

    MirroredStrategy的步驟如下：

    - 訓練開始前，該策略在所有 N 個計算設備上均各複製一份完整的模型；
    - 每次訓練傳入一個批量的資料時，將資料分成 N 份，分別傳入 N 個計算設備（即資料平行處理）；
    - N 個計算設備使用區域變數（鏡像變數）分別計算自己所獲得部分的資料梯度；
    - 使用分散式計算的 All-reduce 操作，在計算設備間高效的，交換梯度資料並進行求和，使得最終每個設備都有了所有設備的梯度和；
    - 使用梯度求和的結果更新區域變數（鏡像變數）；
    - 當所有設備均更新區域變數後，進行下一輪訓練（該並行策略是同步的）。

    默認情況下，TensorFlow 中的 ``MirroredStrategy`` 策略使用 NVIDIA NCCL 進行 All-reduce 操作。

    ..
        https://www.tensorflow.org/beta/tutorials/distribute/training_loops

        爲了進一步理解MirroredStrategy的過程，以下展示一個手工構建訓練流程的示例，相對而言要複雜不少：

        # TODO

多機訓練： ``MultiWorkerMirroredStrategy`` 
-------------------------------------------

..
    https://www.tensorflow.org/beta/tutorials/distribute/multi_worker_with_keras

多機訓練的方法和單機多卡類似，將 ``MirroredStrategy`` 更換為適合多機訓練的 ``MultiWorkerMirroredStrategy`` 即可。不過，由於涉及到多台電腦之間的通訊，還需要進行一些額外的設置。具體而言，需要設置環境變數 ``TF_CONFIG`` ，範例如下::

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ["localhost:20000", "localhost:20001"]
        },
        'task': {'type': 'worker', 'index': 0}
    })

``TF_CONFIG`` 由 ``cluster`` 和 ``task`` 兩部分組成：

- ``cluster`` 說明了整個多機集群的結構和每台機器的網路位置（IP + 埠號）。對於每一台機器，``cluster`` 的值都是相同的；
- ``task`` 說明了目前機器的角色。例如， ``{'type': 'worker', 'index': 0}`` 說明目前機器是 ``cluster`` 中的第 0 個 worker（即 ``localhost:20000`` ）。每一台機器的 ``task`` 值都需要針對目前主機進行分別的設置。

以上內容設置完成後，在所有的機器上逐一執行訓練程式碼即可。先執行的程式碼在尚未與其他主機連接時會進入監聽狀態，待整個集群的連接建立完畢後，所有的機器即會同時開始訓練。

.. hint:: 請在各台機器上均注意防火牆的設置，尤其是需要開放與其他主機通信的埠號。如上例的 0 號 worker 需要開放 20000 埠號，1 號 worker 需要開放 20001 埠號。

以下範例的訓練任務與之前章節相同，只不過遷移到了多機訓練環境。假設我們有兩台機器，即首先在兩台機器上均部署下面的程式，唯一的區別是 ``task`` 部分，第一台機器設置為 ``{'type': 'worker', 'index': 0}`` ，第二台機器設置為 ``{'type': 'worker', 'index': 1}`` 。接下來，在兩台機器上依序執行程式，待通訊成功後，即會自動開始訓練流程。

.. literalinclude:: /_static/code/zh-hant/distributed/multi_worker.py
    :emphasize-lines: 10-18, 27

在以下測試中，我們在 Google Cloud Platform 分別建立兩台具有單張 NVIDIA Tesla K80 的虛擬機實例（具體建立方式參見 :ref:`後文介紹 <zh_hant_GCP>` ），並分別測試在使用一個 GPU 時的訓練時間和使用兩台虛擬機實例進行分散式訓練的訓練時間。所有測試的 epoch 數均為 5。使用單機單卡時，Batch Size 設置為 64。使用雙機單卡時，測試總 Batch Size 為 64（分發到單台機器的 Batch Size 為 32）和總 Batch Size 為 128（分發到單台機器的 Batch Size 為 64）兩種情況。

============  ==========================  ==============================  =============================
資料集        單機單卡（Batch Size爲64）   雙機單卡（總Batch Size爲64）    雙機單卡（總Batch Size爲128）
============  ==========================  ==============================  =============================
cats_vs_dogs  1622s                       858s                            755s
tf_flowers    301s                        152s                            144s                               
============  ==========================  ==============================  =============================

可見模型訓練的速度同樣有大幅度的提高。在所有機器性能接近的情況下，訓練時間與機器的數目接近於反比關係。

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 196 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>


