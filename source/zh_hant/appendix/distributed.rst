TensorFlow分布式訓練
============================================

..
    https://www.tensorflow.org/beta/guide/distribute_strategy

當我們擁有大量計算資源時，通過使用合適的分布式策略，我們可以充分利用這些計算資源，從而大幅壓縮模型訓練的時間。針對不同的使用場景，TensorFlow在 ``tf.distribute.Strategy`` 中爲我們提供了若干種分布式策略，使得我們能夠更高效地訓練模型。

.. _multi_gpu:

單機多卡訓練： ``MirroredStrategy`` 
-------------------------------------------

..
    https://www.tensorflow.org/beta/tutorials/distribute/keras
    https://juejin.im/post/5ba9d72ff265da0ac849384b
    https://www.codercto.com/a/86644.html

``tf.distribute.MirroredStrategy`` 是一種簡單且高性能的，數據並行的同步式分布式策略，主要支持多個GPU在同一台主機上訓練。使用這種策略時，我們只需實例化一個 ``MirroredStrategy`` 策略::

    strategy = tf.distribute.MirroredStrategy()

並將模型構建的代碼放入 ``strategy.scope()`` 的上下文環境中::

    with strategy.scope():
        # 模型構建代碼

.. tip:: 可以在參數中指定設備，如::

        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    
    即指定只使用第0、1號GPU參與分布式策略。
    
以下代碼展示了使用 ``MirroredStrategy`` 策略，在 :doc:`TensorFlow Datasets <../appendix/tfds>` 中的部分圖像數據集上使用Keras訓練MobileNetV2的過程：

.. literalinclude:: /_static/code/zh/distributed/multi_gpu.py
    :emphasize-lines: 8-10, 21

在以下的測試中，我們使用同一台主機上的4塊NVIDIA GeForce GTX 1080 Ti顯卡進行單機多卡的模型訓練。所有測試的epoch數均爲5。使用單機無分布式配置時，雖然機器依然具有4塊顯卡，但程序不使用分布式的設置，直接進行訓練，Batch Size設置爲64。使用單機四卡時，測試總Batch Size爲64（分發到單台機器的Batch Size爲16）和總Batch Size爲256（分發到單台機器的Batch Size爲64）兩種情況。

============  ==============================  ==============================  =============================
數據集        單機無分布式（Batch Size爲64）  單機四卡（總Batch Size爲64）    單機四卡（總Batch Size爲256）
============  ==============================  ==============================  =============================
cats_vs_dogs  146s/epoch                      39s/epoch                       29s/epoch
tf_flowers    22s/epoch                       7s/epoch                        5s/epoch
============  ==============================  ==============================  =============================

可見，使用MirroredStrategy後，模型訓練的速度有了大幅度的提高。在所有顯卡性能接近的情況下，訓練時長與顯卡的數目接近於反比關係。

.. admonition:: ``MirroredStrategy`` 過程簡介

    MirroredStrategy的步驟如下：

    - 訓練開始前，該策略在所有N個計算設備上均各複製一份完整的模型；
    - 每次訓練傳入一個批次的數據時，將數據分成N份，分別傳入N個計算設備（即數據並行）；
    - N個計算設備使用本地變量（鏡像變量）分別計算自己所獲得的部分數據的梯度；
    - 使用分布式計算的All-reduce操作，在計算設備間高效交換梯度數據並進行求和，使得最終每個設備都有了所有設備的梯度之和；
    - 使用梯度求和的結果更新本地變量（鏡像變量）；
    - 當所有設備均更新本地變量後，進行下一輪訓練（即該並行策略是同步的）。

    默認情況下，TensorFlow中的 ``MirroredStrategy`` 策略使用NVIDIA NCCL進行All-reduce操作。

    ..
        https://www.tensorflow.org/beta/tutorials/distribute/training_loops

        爲了進一步理解MirroredStrategy的過程，以下展示一個手工構建訓練流程的示例，相對而言要複雜不少：

        # TODO

多機訓練： ``MultiWorkerMirroredStrategy`` 
-------------------------------------------

..
    https://www.tensorflow.org/beta/tutorials/distribute/multi_worker_with_keras

多機訓練的方法和單機多卡類似，將 ``MirroredStrategy`` 更換爲適合多機訓練的 ``MultiWorkerMirroredStrategy`` 即可。不過，由於涉及到多台計算機之間的通訊，還需要進行一些額外的設置。具體而言，需要設置環境變量 ``TF_CONFIG`` ，示例如下::

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ["localhost:20000", "localhost:20001"]
        },
        'task': {'type': 'worker', 'index': 0}
    })

``TF_CONFIG`` 由 ``cluster`` 和 ``task`` 兩部分組成：

- ``cluster`` 說明了整個多機集羣的結構和每台機器的網絡地址（IP+端口號）。對於每一台機器，``cluster`` 的值都是相同的；
- ``task`` 說明了當前機器的角色。例如， ``{'type': 'worker', 'index': 0}`` 說明當前機器是 ``cluster`` 中的第0個worker（即 ``localhost:20000`` ）。每一台機器的 ``task`` 值都需要針對當前主機進行分別的設置。

以上內容設置完成後，在所有的機器上逐個運行訓練代碼即可。先運行的代碼在尚未與其他主機連接時會進入監聽狀態，待整個集羣的連接建立完畢後，所有的機器即會同時開始訓練。

.. hint:: 請在各台機器上均注意防火牆的設置，尤其是需要開放與其他主機通信的端口。如上例的0號worker需要開放20000端口，1號worker需要開放20001端口。

以下示例的訓練任務與前節相同，只不過遷移到了多機訓練環境。假設我們有兩台機器，即首先在兩台機器上均部署下面的程序，唯一的區別是 ``task`` 部分，第一台機器設置爲 ``{'type': 'worker', 'index': 0}`` ，第二台機器設置爲 ``{'type': 'worker', 'index': 1}`` 。接下來，在兩台機器上依次運行程序，待通訊成功後，即會自動開始訓練流程。

.. literalinclude:: /_static/code/zh/distributed/multi_worker.py
    :emphasize-lines: 10-18, 27

在以下測試中，我們在Google Cloud Platform分別建立兩台具有單張NVIDIA Tesla K80的虛擬機實例（具體建立方式參見 :ref:`後文介紹 <GCP>` ），並分別測試在使用一個GPU時的訓練時長和使用兩台虛擬機實例進行分布式訓練的訓練時長。所有測試的epoch數均爲5。使用單機單卡時，Batch Size設置爲64。使用雙機單卡時，測試總Batch Size爲64（分發到單台機器的Batch Size爲32）和總Batch Size爲128（分發到單台機器的Batch Size爲64）兩種情況。

============  ==========================  ==============================  =============================
數據集        單機單卡（Batch Size爲64）   雙機單卡（總Batch Size爲64）    雙機單卡（總Batch Size爲128）
============  ==========================  ==============================  =============================
cats_vs_dogs  1622s                       858s                            755s
tf_flowers    301s                        152s                            144s                               
============  ==========================  ==============================  =============================

可見模型訓練的速度同樣有大幅度的提高。在所有機器性能接近的情況下，訓練時長與機器的數目接近於反比關係。

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


