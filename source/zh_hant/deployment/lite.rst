TensorFlow Lite（Jinpeng）
====================================================

TensorFlow Lite 是 TensorFlow 在可攜式和 IoT 等邊緣設備端的解決方案，提供了 Java、Python 和 C++ API 庫，可以執行在 Android、iOS 和 Raspberry Pi 等設備上。2019 年是 5G 元年，萬物互聯的時代已經來臨，作為 TensorFlow 在邊緣設備上的基礎設施，TFLite 將會是越來越重要的角色。

目前 TFLite 只提供了推論功能，在伺服器端進行訓練後，經過如下簡單處理即可部署到邊緣設備上。

* 模型轉換：由於邊緣設備計算等資源有限，使用 TensorFlow 訓練好的模型，模型太大、執行效率比較低，不能直接在可攜式裝置部署，需要透過相應工具進行轉換成適合邊緣設備的格式。

* 邊緣設備部署：本章節以 Android 為例，簡單介紹如何在 Android 應用中部署轉化後的模型，完成 Mnist 圖片的辨識。

模型轉換
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
轉換工具有兩種：終端機工具和 Python API

TF2.0 對模型轉換工具發生了非常大的變化，推薦大家使用 Python API 進行轉換，終端機工具只提供了基本的轉化功能。轉換後的原模型為 ``FlatBuffers`` 格式。 ``FlatBuffers`` 原來主要應用於遊戲場景，是Google為了高性能場景創建的序列化函式庫，相比 Protocol Buffer 有更高的性能和更小的檔案等優勢，更適合於邊緣設備部署。

轉換方式有兩種：Float 格式和 Quantized 格式

為了熟悉兩種方式我們都會使用，針對 Float 格式的，先使用終端機工具 ``tflite_convert`` ，跟著 TensorFlow 一起安裝（見 `一般安裝步驟 <https://tf.wiki/zh/basic/installation.html#id1>`_ ）。 

在終端機執行下列指令：

.. code-block::

    tflite_convert -h

輸出結果如下，該指令的使用方法：

.. code-block::

    usage: tflite_convert [-h] --output_file OUTPUT_FILE
                          (--saved_model_dir SAVED_MODEL_DIR | --keras_model_file KERAS_MODEL_FILE)
      --output_file OUTPUT_FILE
                            Full filepath of the output file.
      --saved_model_dir SAVED_MODEL_DIR
                            Full path of the directory containing the SavedModel.
      --keras_model_file KERAS_MODEL_FILE
                            Full filepath of HDF5 file containing tf.Keras model.

在 `TensorFlow 模型匯出 <https://tf.wiki/zh-hant/deployment/export.html>`_ 中，我們知道 TF2.0 支援兩種模型匯出方法和格式 SavedModel 和 Keras Sequential。

SavedModel 匯出模型轉換：

.. code-block:: bash

    tflite_convert --saved_model_dir=saved/1 --output_file=mnist_savedmodel.tflite

Keras Sequential 匯出模型轉換：

.. code-block:: bash

    tflite_convert --keras_model_file=mnist_cnn.h5 --output_file=mnist_sequential.tflite

到此，已經得到兩個 TensorFlow Lite 模型。因為兩者後續操作基本一致，我們只處理 SavedModel 格式的，Keras Sequential 的轉換可以按照相同的方法處理。

Android部署
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

現在開始在 Android 環境部署，需要先給 Android Studio 配置 proxy 的鏡像網址。

**配置build.gradle**

將 ``build.gradle`` 中的 maven 來源 ``google()`` 和 ``jcenter()`` 分別替換為鏡像網址，如下：

.. code-block::

    buildscript {
    
        repositories {
            maven { url 'https://maven.aliyun.com/nexus/content/repositories/google' }
            maven { url 'https://maven.aliyun.com/nexus/content/repositories/jcenter' }
        }
        dependencies {
            classpath 'com.android.tools.build:gradle:3.5.1'
        }
    }
    
    allprojects {
        repositories {
            maven { url 'https://maven.aliyun.com/nexus/content/repositories/google' }
            maven { url 'https://maven.aliyun.com/nexus/content/repositories/jcenter' }
        }
    }

**配置app/build.gradle**

新建一個 Android Project，打開 ``app/build.gradle`` 添加如下資訊：

.. code-block::

    android {
        aaptOptions {
            noCompress "tflite" // 編譯apk時，不壓縮tflite文件
        }
    }

    dependencies {
        implementation 'org.tensorflow:tensorflow-lite:1.14.0'
    }

其中，

#. ``aaptOptions`` 設置 tflite 文件不壓縮，確保後面 tflite 文件可以被 Interpreter 正確載入。
#. ``org.tensorflow:tensorflow-lite`` 的最新版本號碼可以在這裡查詢 https://bintray.com/google/tensorflow/tensorflow-lite

設置好之後，sync 和 build 整個程式包，如果 build 成功說明，配置成功。

**添加 tflite 文件到 assets 資料夾**

在 app 目錄先新建 assets 目錄，並將 ``mnist_savedmodel.tflite`` 文件保存到assets目錄。重新編譯apk，檢查新編譯出來的 apk 的 assets 資料夾是否有 ``mnist_cnn.tflite`` 文件。

點擊選單 Build->Build APK(s) 觸發 apk 編譯，apk 編譯成功點擊右下角的 EventLog。點擊最後一條資訊中的 ``analyze`` 連結，會觸發 apk analyzer 查看新編譯出來的 apk，若在 assets 目錄下存在 ``mnist_savedmodel.tflite`` ，則編譯打包成功，如下：

.. code-block::

    assets
         |__mnist_savedmodel.tflite

**載入模型**

使用如下指令將 ``mnist_savedmodel.tflite`` 文件載入到 memory-map 中，作為 Interpreter 實例化的輸入

.. code-block:: java

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(mModelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

.. hint::

    memory-map 可以把整個文件映射到虛擬記憶體中，用於提升 tflite 模型的讀取性能。更多請參考： `JDK API介紹 <https://docs.oracle.com/javase/8/docs/api/java/nio/channels/FileChannel.html#map-java.nio.channels.FileChannel.MapMode-long-long->`_

實例化 Interpreter，其中 acitivity 是為了從 assets 中獲取模型，因為我們把模型編譯到 assets 中，只能透過 ``getAssets()`` 打開。

.. code-block:: java

    mTFLite = new Interpreter(loadModelFile(activity));

memory-map後的 ``MappedByteBuffer`` 直接作為 ``Interpreter`` 的輸入， ``mTFLite`` （ ``Interpreter`` ）就是轉換後模型的執行載體。

**執行輸入**

我們使用 MNIST test 測試集中的圖片作為輸入，mnist 圖像大小 28*28，單像素，因為我們輸入的資料需要設置成如下格式

.. code-block:: java

    //Float模型相關參數
    // com/dpthinker/mnistclassifier/model/FloatSavedModelConfig.java
    protected void setConfigs() {
        setModelName("mnist_savedmodel.tflite");

        setNumBytesPerChannel(4);

        setDimBatchSize(1);
        setDimPixelSize(1);

        setDimImgWeight(28);
        setDimImgHeight(28);

        setImageMean(0);
        setImageSTD(255.0f);
    }

    // 初始化
    // com/dpthinker/mnistclassifier/classifier/BaseClassifier.java
    private void initConfig(BaseModelConfig config) {
        this.mModelConfig = config;
        this.mNumBytesPerChannel = config.getNumBytesPerChannel();
        this.mDimBatchSize = config.getDimBatchSize();
        this.mDimPixelSize = config.getDimPixelSize();
        this.mDimImgWidth = config.getDimImgWeight();
        this.mDimImgHeight = config.getDimImgHeight();
        this.mModelPath = config.getModelName();
    }

將 MNIST 圖片轉化成 ``ByteBuffer`` ，並保持到 ``imgData`` （  ``ByteBuffer`` ）中

.. code-block:: java

    // 將輸入的Bitmap轉化為Interpreter可以辨識的ByteBuffer
    // com/dpthinker/mnistclassifier/classifier/BaseClassifier.java
    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        int[] intValues = new int[mDimImgWidth * mDimImgHeight];
        scaleBitmap(bitmap).getPixels(intValues,
                0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        ByteBuffer imgData = ByteBuffer.allocateDirect(
                mNumBytesPerChannel * mDimBatchSize * mDimImgWidth * mDimImgHeight * mDimPixelSize);
        imgData.order(ByteOrder.nativeOrder());
        imgData.rewind();

        // Convert the image toFloating point.
        int pixel = 0;
        for (int i = 0; i < mDimImgWidth; ++i) {
            for (int j = 0; j < mDimImgHeight; ++j) {
                //final int val = intValues[pixel++];
                int val = intValues[pixel++];
                mModelConfig.addImgValue(imgData, val); //添加把Pixel數值轉化並添加到ByteBuffer
            }
        }
        return imgData;
    }

    // mModelConfig.addImgValue定義
    // com/dpthinker/mnistclassifier/model/FloatSavedModelConfig.java
    public void addImgValue(ByteBuffer imgData, int val) {
        imgData.putFloat(((val & 0xFF) - getImageMean()) / getImageSTD());
    }


``convertBitmapToByteBuffer`` 的輸出即為模型執行的輸入。

**執行輸出**

定義一個 1*10 的多維陣列，因為我們只有 10 個 label，具體程式碼如下

.. code-block:: java

    privateFloat[][] mLabelProbArray = newFloat[1][10];

執行結束後，每個二級元素都是一個label的機率。

**執行及結果處理**

開始執行模型，具體程式碼如下

.. code-block:: java

    mTFLite.run(imgData, mLabelProbArray);

針對某個圖片，執行後 ``mLabelProbArray`` 的內容就是各個 label 辨識的機率。對他們進行排序，找出機率最高的 label 並顯示辨識結果給用戶.

在Android應用中，筆者使用了 ``View.OnClickListener()`` 觸發 ``"image/*"`` 類型的 ``Intent.ACTION_GET_CONTENT`` ，進而獲取設備上的圖片（只支援 MNIST 標準圖片）。然後，透過 ``RadioButtion`` 的選擇情況，確認載入哪種轉換後的模型，並觸發真正分類操作。這部分比較簡單，請讀者自行閱讀程式碼即可，不再重複介紹。

選取一張 MNIST 測試集中的圖片進行測試，得到結果如下：

.. figure:: /_static/image/deployment/mnist_float.png
    :width: 40%
    :align: center

.. hint::
    
    注意我們這裡直接用 ``mLabelProbArray`` 數值中的 index作為label了，因為 MNIST 的 label 完全跟 index 從 0 到 9 匹配。如果是其他的分類問題，需要根據實際情況進行轉換。

Quantization 模型轉換
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::
    Quantized 模型是對原模型進行轉換過程中，將 float 參數轉化為 uint8 類型，進而產生的模型會更小、執行更快，但是解析度會有所下降。

前面我們介紹了 Float 模型的轉換方法，接下來我們要展示 Quantized 模型，在 TF1.0 上，可以使用終端機工具轉換 Quantized 模型。在筆者嘗試的情況看在 TF2.0 上，終端機工具目前只能轉換為 Float 模型，Python API 只能轉換為 Quantized 模型。

Python API 轉換方法如下：

.. code-block:: bash

    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model('saved/1')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    open("mnist_savedmodel_quantized.tflite", "wb").write(tflite_quant_model)

最終轉換後的 Quantized 模型即為同個目錄下的 ``mnist_savedmodel_quantized.tflite`` 。

相對 TF1.0，上面的方法簡化了很多，不需要考慮各種各樣的參數，谷歌一直在優化開發者的使用體驗。

在TF1.0上，我們可以使用 ``tflite_convert`` 獲得模型具體結構，然後通過 graphviz 轉換為 pdf 或 png 等方便查看。
在TF2.0上，提供了新的一步到位的工具 ``visualize.py`` ，直接轉換為 html 文件，除了模型結構，還有更清晰的關鍵資訊。

.. hint::
    
    ``visualize.py`` 目前應該還是開發階段，使用前需要先從 github 下載最新的 ``TensorFlow`` 和 ``FlatBuffers`` 原始碼，並且兩者要在同一目錄，因為 ``visualize.py`` 原始碼中是按照兩者在同一目錄寫的呼叫路徑。

    下載 TensorFlow：

    .. code-block:: bash    
        
        git clone git@github.com:tensorflow/tensorflow.git
    
    下載 FlatBuffers：
    
    .. code-block:: bash    
    
        git clone git@github.com:google/flatbuffers.git
    
    編譯 FlatBuffers：（筆者使用的 Mac，其他平台請大家自行配置，應該不麻煩）
    
    #. 下載cmake：執行 ``brew install cmake``
    #. 設置編譯環境：在 ``FlatBuffers`` 的根目錄，執行 ``cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release``
    #. 編譯：在 ``FlatBuffers`` 的根目錄，執行 ``make``

    編譯完成後，會在跟目錄生成 ``flatc``，這個可執行文件是 ``visualize.py`` 執行所依賴的。

**visualize.py使用方法**

在tensorflow/tensorflow/lite/tools 目錄下，執行以下指令

.. code-block:: bash

    python visualize.py mnist_savedmodel_quantized.tflite mnist_savedmodel_quantized.html

生成關鍵資訊的可視化圖表

.. figure:: /_static/image/deployment/visualize1.png
    :width: 100%
    :align: center

模型結構

.. figure:: /_static/image/deployment/visualize2.png
    :width: 40%
    :align: center

可以發現，Input/Output 格式都是 ``FLOAT32`` 的多維陣列，Input 的 min 和 max 分別是 0.0 和 255.0。

跟 Float 模型對比，Input/Output 格式是一致的，所以可以重複使用 Float 模型 Android 部署過程中的配置。

.. hint::
    
    暫不確定這裡是否是 TF2.0 上的優化，如果是這樣的話，對開發者來說是非常友好的，這樣就正規化 Float 和 Quantized 模型處理了。

具體配置如下：

.. code-block:: java

    // Quantized模型相關參數
    // com/dpthinker/mnistclassifier/model/QuantSavedModelConfig.java
    public class QuantSavedModelConfig extends BaseModelConfig {
        @Override
        protected void setConfigs() {
            setModelName("mnist_savedmodel_quantized.tflite");

            setNumBytesPerChannel(4);

            setDimBatchSize(1);
            setDimPixelSize(1);

            setDimImgWeight(28);
            setDimImgHeight(28);

            setImageMean(0);
            setImageSTD(255.0f);
        }

        @Override
        public void addImgValue(ByteBuffer imgData, int val) {
            imgData.putFloat(((val & 0xFF) - getImageMean()) / getImageSTD());
        }
    }

執行結果如下:

.. figure:: /_static/image/deployment/quantized.png
    :width: 40%
    :align: center

Float 模型與 Quantized 模型大小與性能對比：

.. list-table:: 
   :header-rows: 1
   :align: center

   * - 模型類別
     - Float
     - Quantized
   * - 模型大小
     - 312K
     - 82K
   * - 運行性能
     - 5.858854ms
     - 1.439062ms

可以發現， Quantized 模型在模型大小和執行性能上相對 Float 模型都有非常大的提升。不過，在筆者測試的過程中，發現有些圖片在 Float 模型上辨識正確的，在 Quantized 模型上會辨識錯，可見 ``Quantization`` 對模型的辨識解析度還是有影響的。由於在邊緣設備上資源有限，因此需要在模型大小、執行速度與辨識解析度上找到平衡。

總結
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
本節介紹如何從零開始部署 TFLite 到 Android 應用中，包括：

#. 如何將訓練好的 MNIST SavedModel 模型，轉換為 Float 模型和 Quantized 模型
#. 如何使用 ``visualize.py`` 和解讀其結果資訊
#. 如何將轉換後的模型部署到 Android 應用中

筆者剛開始寫這部分內容的時候還是 TF1.0，在最近（2019年10月初）跟TF2.0的時候，發現有了很多變化，整體上是比原來更簡單了。不過文件部分很多還是講的比較模糊，很多地方還是需要看原始碼摸索。

.. hint::
    本節Android相關程式碼存放路徑：
    ``https://github.com/snowkylin/tensorflow-handbook/tree/master/source/android``

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 194 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>
