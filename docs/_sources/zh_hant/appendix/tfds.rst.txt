TensorFlow Datasets 資料集載入
============================================

`TensorFlow Datasets <https://www.tensorflow.org/datasets/>`_ 是一個可以馬上使用的資料集集合，包含數十種常用的機器學習資料集。通過簡單的幾行程式碼即可將資料以 ``tf.data.Dataset`` 的格式載入。關於 ``tf.data.Dataset`` 的使用可參考:ref:`tf.data <zh_hant_tfdata>`。

該工具是一個獨立的 Python 套件，可以通過::

    pip install tensorflow-datasets

安裝。

在使用時，首先使用 import 載入該套件

.. code-block:: python

    import tensorflow as tf
    import tensorflow_datasets as tfds

然後，最基礎的用法是使用 ``tfds.load`` 方法，載入所需的資料集。例如，以下三行程式碼分別載入了 MNIST、貓狗分類和 ``tf_flowers`` 三個圖像分類資料集：


.. code-block:: python

    dataset = tfds.load("mnist", split=tfds.Split.TRAIN, as_supervised=True)
    dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
    dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)

當第一次載入特定資料集時，TensorFlow Datasets 會自動從雲端下載資料集到本機，並顯示下載進度。例如，載入 MNIST 資料集時，終端機輸出如下：

::

    Downloading and preparing dataset mnist (11.06 MiB) to C:\Users\snowkylin\tensorflow_datasets\mnist\3.0.0...
    WARNING:absl:Dataset mnist is hosted on GCS. It will automatically be downloaded to your
    local data directory. If you'd instead prefer to read directly from our public
    GCS bucket (recommended if you're running on GCP), you can instead set
    data_dir=gs://tfds-data/datasets.

    Dl Completed...: 100%|██████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:10<00:00,  2.93s/ file] 
    Dl Completed...: 100%|██████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:10<00:00,  2.73s/ file] 
    Dataset mnist downloaded and prepared to C:\Users\snowkylin\tensorflow_datasets\mnist\3.0.0. Subsequent calls will reuse this data.

.. hint:: 在使用 TensorFlow Datasets 時，可能需要設置代理伺服器。較為簡易的方式是設置 ``TFDS_HTTPS_PROXY`` 環境變數（ `參考這裡 <https://github.com/tensorflow/datasets/blob/dd51a2d510bdcbf4498e9dcd2ee1ef33d44a13f3/tensorflow_datasets/core/download/downloader.py#L147>`_ ），即

    ::

        export HTTPS_PROXY=http://代理伺服器IP:埠號

``tfds.load`` 方法返回一個 ``tf.data.Dataset`` 對象。部分重要的參數如下：

..
    https://www.tensorflow.org/datasets/api_docs/python/tfds/load

- ``as_supervised`` ：若為 True，則根據資料集的特性，將資料集中的每行元素整理為監督式的二元組 ``(input, label)`` （即 “資料 + 標籤”）形式，否則資料集中，每行元素為包含所有特徵的字典。
- ``split``：指定返回資料集的特定部分。若不指定，則返回整個資料集。一般有 ``tfds.Split.TRAIN`` （訓練集）和 ``tfds.Split.TEST`` （測試集）選項。

TensorFlow Datasets 當前支持的資料集可在 `官方文檔 <https://www.tensorflow.org/datasets/datasets>`_ 查看，或者也可以使用 ``tfds.list_builders()`` 查看。

當得到了 ``tf.data.Dataset`` 類型的資料集後，我們即可使用 ``tf.data`` 對資料集進行各種預處理以及讀取資料。例如：

.. code-block:: python
    
    # 使用 TessorFlow Datasets 載入「tf_flowers」資料集
    dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
    # 對 dataset 進行大小調整、打散和分批次操作
    dataset = dataset.map(lambda img, label: (tf.image.resize(img, [224, 224]) / 255.0, label)) \
        .shuffle(1024) \
        .batch(32)
    # 疊代資料
    for images, labels in dataset:
        # 對images和labels進行操作

詳細操作說明可見 :ref:`本手冊的 tf.data 一節 <zh_hant_tfdata>` 。同時，本手冊的 :doc:`分散式訓練 <../appendix/distributed>` 一章也使用了 TensorFlow Datasets 載入資料集。可以參考這些章節的範例程式碼以進一步了解 TensorFlow Datasets 的使用方法。

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 199 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>
