TensorFlow Hub 模型復用（Jinpeng）
============================================

在軟體開發中，我們經常復用開源軟體或者庫，避免了相同功能的代碼重複開發，減少了大量的重複勞動，也有效縮短了軟體開發周期。代碼復用，對軟體產業的蓬勃發展，有著極大的助推作用。

相應的，TF Hub目的是爲了更好的復用已訓練好且經過充分驗證的模型，可節省海量的訓練時間和計算資源。這些預訓練好的模型，可以進行直接部署，也可以進行遷移學習（Transfer Learning）。對個人開發者來說，TF Hub是非常有意義的，他們可以快速復用像谷歌這樣的大公司使用海量計算資源訓練的模型，而他們個人去獲取這些資源是很不現實的。

TF Hub 網站
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /_static/image/appendix/tfhub_main.png
    :width: 90%
    :align: center

打開主頁 ``https://tfhub.dev/`` ，在左側有Text、Image、Video和Publishers等選項，可以選取關注的類別，然後在頂部的搜索框輸入關鍵字可以搜索模型。

以 ``stylization`` 爲例，我們搜索到如下模型：

.. figure:: /_static/image/appendix/tfhub_example.png
    :width: 90%
    :align: center

該模型的地址如下：

https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2

其中，末尾的 ``2`` 爲該模型的版本號。

.. hint::
    
    #. 注意目前還有很多模型是基於TF1.0的，選擇的過程中請注意甄別，有些模型會明確寫出來是試用哪個版本，或者，檢查使用是否是tfhub 0.5.0或以上版本的API ``hub.load(url)`` ，在之前版本使用的是 ``hub.Module(url)`` 。
    #. 如果不能訪問tfhub.dev，請大家轉換域名到國內鏡像 ``https://hub.tensorflow.google.cn/`` ，模型下載地址也需要相應轉換。

TF Hub 安裝
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TF Hub是單獨的一個庫，需要單獨安裝，安裝命令如下：


.. code-block:: bash

    pip install tensorflow-hub


.. hint::
    
    在TF2.0上，必須使用0.5.0或以上版本，因爲接口有變動。


TF Hub 模型使用樣例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TF Hub模型的復用非常簡單，代碼模式如下：

.. code-block:: python

    import tensorflow_hub as hub
    
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_model = hub.load(hub_handle)
    outputs = hub_model(inputs)

根據 ``stylization`` 模型的參考代碼和notebook，進行了精簡和修改，實現了圖像的風格轉換功能。

.. literalinclude:: /_static/code/zh/tfhub/stylization.py
    :lines: 19-

其中， ``hub.load(url)`` 就是把TF Hub的模型從網絡下載和加載進來， ``hub_module`` 就是運行模型， ``outputs`` 即爲輸出。

上面的代碼，輸入的圖像是一張筆者拍的風景照片，風格圖片是故宮館藏的《王希孟千里江山圖卷》部分截屏。

輸入圖片：

.. figure:: /_static/image/appendix/contentimg.jpeg
    :width: 90%
    :align: center

風格圖片：

.. figure:: /_static/image/appendix/styleimg.jpeg
    :width: 90%
    :align: center

輸出圖片：

.. figure:: /_static/image/appendix/stylized_img.png
    :width: 90%
    :align: center

大家可以在如下路徑獲取notebook和代碼體驗：

https://github.com/snowkylin/tensorflow-handbook/tree/master/source/_static/code/zh/tfhub

也可在谷歌提供的如下notebook體驗：

https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_arbitrary_image_stylization.ipynb

TF Hub 模型retrain樣例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

相信預預訓練的模型不一定滿足開發者的實際訴求，還需要進行二次訓練。針對這種情況，TF Hub提供了很方便的Keras接口 ``hub.KerasLayer(url)`` ，其可以封裝在Keras的 ``Sequential`` 層狀結構中，進而可以針對開發者的需求和數據進行再訓練。

我們以 ``inception_v3`` 的模型爲例，簡單介紹 ``hub.KerasLayer(url)`` 使用的方法：

.. code-block:: python
    
    import tensorflow as tf
    import tensorflow_hub as hub
    
    num_classes = 10
    
    # 使用 hub.KerasLayer 組件待訓練模型
    new_model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4", output_shape=[2048], trainable=False),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    new_model.build([None, 299, 299, 3])
    
    # 輸出模型結構
    new_model.summary()


執行以上代碼輸出結果如下，其中 ``keras_layer (KerasLayer)`` 就是從TF Hub上獲取的模型。

.. code-block:: bash

   Model: "sequential"
   _________________________________________________________________
   Layer (type)                 Output Shape              Param #   
   =================================================================
   keras_layer (KerasLayer)     multiple                  21802784  
   _________________________________________________________________
   dense (Dense)                multiple                  20490     
   =================================================================
   Total params: 21,823,274
   Trainable params: 20,490
   Non-trainable params: 21,802,784
   _________________________________________________________________ 

剩下的訓練和模型保存跟正常的Keras的 ``Sequential`` 模型完全一樣。

可在谷歌提供的如下notebook體驗：

https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 198 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>
