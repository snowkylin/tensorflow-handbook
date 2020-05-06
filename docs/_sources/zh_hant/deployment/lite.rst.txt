TensorFlow Lite（Jinpeng）
====================================================

TensorFlow Lite是TensorFlow在移動和IoT等邊緣設備端的解決方案，提供了Java、Python和C++ API庫，可以運行在Android、iOS和Raspberry Pi等設備上。2019年是5G元年，萬物互聯的時代已經來臨，作爲TensorFlow在邊緣設備上的基礎設施，TFLite將會是愈發重要的角色。

目前TFLite只提供了推理功能，在服務器端進行訓練後，經過如下簡單處理即可部署到邊緣設備上。

* 模型轉換：由於邊緣設備計算等資源有限，使用TensorFlow訓練好的模型，模型太大、運行效率比較低，不能直接在移動端部署，需要通過相應工具進行轉換成適合邊緣設備的格式。

* 邊緣設備部署：本節以android爲例，簡單介紹如何在android應用中部署轉化後的模型，完成Mnist圖片的識別。

模型轉換
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
轉換工具有兩種：命令行工具和Python API

TF2.0對模型轉換工具發生了非常大的變化，推薦大家使用Python API進行轉換，命令行工具只提供了基本的轉化功能。轉換後的原模型爲 ``FlatBuffers`` 格式。 ``FlatBuffers`` 原來主要應用於遊戲場景，是谷歌爲了高性能場景創建的序列化庫，相比Protocol Buffer有更高的性能和更小的大小等優勢，更適合於邊緣設備部署。

轉換方式有兩種：Float格式和Quantized格式

爲了熟悉兩種方式我們都會使用，針對Float格式的，先使用命令行工具 ``tflite_convert`` ，其跟隨TensorFlow一起安裝（見 `一般安裝步驟 <https://tf.wiki/zh/basic/installation.html#id1>`_ ）。 

在終端執行如下命令：

.. code-block::

    tflite_convert -h

輸出結果如下，即該命令的使用方法：

.. code-block::

    usage: tflite_convert [-h] --output_file OUTPUT_FILE
                          (--saved_model_dir SAVED_MODEL_DIR | --keras_model_file KERAS_MODEL_FILE)
      --output_file OUTPUT_FILE
                            Full filepath of the output file.
      --saved_model_dir SAVED_MODEL_DIR
                            Full path of the directory containing the SavedModel.
      --keras_model_file KERAS_MODEL_FILE
                            Full filepath of HDF5 file containing tf.Keras model.

在 `TensorFlow模型導出 <https://tf.wiki/zh/deployment/export.html>`_ 中，我們知道TF2.0支持兩種模型導出方法和格式SavedModel和Keras Sequential。

SavedModel導出模型轉換：

.. code-block:: bash

    tflite_convert --saved_model_dir=saved/1 --output_file=mnist_savedmodel.tflite

Keras Sequential導出模型轉換：

.. code-block:: bash

    tflite_convert --keras_model_file=mnist_cnn.h5 --output_file=mnist_sequential.tflite

到此，已經得到兩個TensorFlow Lite模型。因爲兩者後續操作基本一致，我們只處理SavedModel格式的，Keras Sequential的轉換可以按類似方法處理。

Android部署
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

現在開始在Android環境部署，爲了獲取SDK和gradle編譯環境等資源，需要先給Android Studio配置proxy或者使用鏡像。

**配置build.gradle**

將 ``build.gradle`` 中的maven源 ``google()`` 和 ``jcenter()`` 分別替換爲阿里雲鏡像地址，如下：

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

新建一個Android Project，打開 ``app/build.gradle`` 添加如下信息：

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

#. ``aaptOptions`` 設置tflite文件不壓縮，確保後面tflite文件可以被Interpreter正確加載。
#. ``org.tensorflow:tensorflow-lite`` 的最新版本號可以在這裡查詢 https://bintray.com/google/tensorflow/tensorflow-lite

設置好後，sync和build整個工程，如果build成功說明，配置成功。

**添加tflite文件到assets文件夾**

在app目錄先新建assets目錄，並將 ``mnist_savedmodel.tflite`` 文件保存到assets目錄。重新編譯apk，檢查新編譯出來的apk的assets文件夾是否有 ``mnist_cnn.tflite`` 文件。

點擊菜單Build->Build APK(s)觸發apk編譯，apk編譯成功點擊右下角的EventLog。點擊最後一條信息中的 ``analyze`` 鏈接，會觸發apk analyzer查看新編譯出來的apk，若在assets目錄下存在 ``mnist_savedmodel.tflite`` ，則編譯打包成功，如下：

.. code-block::

    assets
         |__mnist_savedmodel.tflite

**加載模型**

使用如下函數將 ``mnist_savedmodel.tflite`` 文件加載到memory-map中，作爲Interpreter實例化的輸入

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

    memory-map可以把整個文件映射到虛擬內存中，用於提升tflite模型的讀取性能。更多請參考： `JDK API介紹 <https://docs.oracle.com/javase/8/docs/api/java/nio/channels/FileChannel.html#map-java.nio.channels.FileChannel.MapMode-long-long->`_

實例化Interpreter，其中acitivity是爲了從assets中獲取模型，因爲我們把模型編譯到assets中，只能通過 ``getAssets()`` 打開。

.. code-block:: java

    mTFLite = new Interpreter(loadModelFile(activity));

memory-map後的 ``MappedByteBuffer`` 直接作爲 ``Interpreter`` 的輸入， ``mTFLite`` （ ``Interpreter`` ）就是轉換後模型的運行載體。

**運行輸入**

我們使用MNIST test測試集中的圖片作爲輸入，mnist圖像大小28*28，單像素，因爲我們輸入的數據需要設置成如下格式

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

將MNIST圖片轉化成 ``ByteBuffer`` ，並保持到 ``imgData`` （  ``ByteBuffer`` ）中

.. code-block:: java

    // 將輸入的Bitmap轉化爲Interpreter可以識別的ByteBuffer
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


``convertBitmapToByteBuffer`` 的輸出即爲模型運行的輸入。

**運行輸出**

定義一個1*10的多維數組，因爲我們只有10個label，具體代碼如下

.. code-block:: java

    privateFloat[][] mLabelProbArray = newFloat[1][10];

運行結束後，每個二級元素都是一個label的概率。

**運行及結果處理**

開始運行模型，具體代碼如下

.. code-block:: java

    mTFLite.run(imgData, mLabelProbArray);

針對某個圖片，運行後 ``mLabelProbArray`` 的內容就是各個label識別的概率。對他們進行排序，找出Top的label並界面呈現給用戶.

在Android應用中，筆者使用了 ``View.OnClickListener()`` 觸發 ``"image/*"`` 類型的 ``Intent.ACTION_GET_CONTENT`` ，進而獲取設備上的圖片（只支持MNIST標準圖片）。然後，通過 ``RadioButtion`` 的選擇情況，確認加載哪種轉換後的模型，並觸發真正分類操作。這部分比較簡單，請讀者自行閱讀代碼即可，不再展開介紹。

選取一張MNIST測試集中的圖片進行測試，得到結果如下：

.. figure:: /_static/image/deployment/mnist_float.png
    :width: 40%
    :align: center

.. hint::
    
    注意我們這裡直接用 ``mLabelProbArray`` 數值中的index作爲label了，因爲MNIST的label完全跟index從0到9匹配。如果是其他的分類問題，需要根據實際情況進行轉換。

Quantization模型轉換
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::
    Quantized模型是對原模型進行轉換過程中，將float參數轉化爲uint8類型，進而產生的模型會更小、運行更快，但是精度會有所下降。

前面我們介紹了Float 模型的轉換方法，接下來我們要展示下 Quantized 模型，在TF1.0上，可以使用命令行工具轉換 Quantized模型。在筆者嘗試的情況看在TF2.0上，命令行工具目前只能轉換爲Float 模型，Python API只能轉換爲 Quantized 模型。

Python API轉換方法如下：

.. code-block:: bash

    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model('saved/1')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    open("mnist_savedmodel_quantized.tflite", "wb").write(tflite_quant_model)

最終轉換後的 Quantized模型即爲同級目錄下的 ``mnist_savedmodel_quantized.tflite`` 。

相對TF1.0，上面的方法簡化了很多，不需要考慮各種各樣的參數，谷歌一直在優化開發者的使用體驗。

在TF1.0上，我們可以使用 ``tflite_convert`` 獲得模型具體結構，然後通過graphviz轉換爲pdf或png等方便查看。
在TF2.0上，提供了新的一步到位的工具 ``visualize.py`` ，直接轉換爲html文件，除了模型結構，還有更清晰的關鍵信息總結。

.. hint::
    
    ``visualize.py`` 目前看應該還是開發階段，使用前需要先從github下載最新的 ``TensorFlow`` 和 ``FlatBuffers`` 源碼，並且兩者要在同一目錄，因爲 ``visualize.py`` 源碼中是按兩者在同一目錄寫的調用路徑。

    下載 TensorFlow：

    .. code-block:: bash    
        
        git clone git@github.com:tensorflow/tensorflow.git
    
    下載 FlatBuffers：
    
    .. code-block:: bash    
    
        git clone git@github.com:google/flatbuffers.git
    
    編譯 FlatBuffers：（筆者使用的Mac，其他平台請大家自行配置，應該不麻煩）
    
    #. 下載cmake：執行 ``brew install cmake``
    #. 設置編譯環境：在 ``FlatBuffers`` 的根目錄，執行 ``cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release``
    #. 編譯：在 ``FlatBuffers`` 的根目錄，執行 ``make``

    編譯完成後，會在跟目錄生成 ``flatc``，這個可執行文件是 ``visualize.py`` 運行所依賴的。

**visualize.py使用方法**

在tensorflow/tensorflow/lite/tools目錄下，執行如下命令

.. code-block:: bash

    python visualize.py mnist_savedmodel_quantized.tflite mnist_savedmodel_quantized.html

生成可視化報告的關鍵信息

.. figure:: /_static/image/deployment/visualize1.png
    :width: 100%
    :align: center

模型結構

.. figure:: /_static/image/deployment/visualize2.png
    :width: 40%
    :align: center

可見，Input/Output格式都是 ``FLOAT32`` 的多維數組，Input的min和max分別是0.0和255.0。

跟Float模型對比，Input/Output格式是一致的，所以可以復用Float模型Android部署過程中的配置。

.. hint::
    
    暫不確定這裡是否是TF2.0上的優化，如果是這樣的話，對開發者來說是非常友好的，如此就歸一化了Float和Quantized模型處理了。

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

運行效果如下:

.. figure:: /_static/image/deployment/quantized.png
    :width: 40%
    :align: center

Float模型與 Quantized模型大小與性能對比：

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

可見， Quantized模型在模型大小和運行性能上相對Float模型都有非常大的提升。不過，在筆者試驗的過程中，發現有些圖片在Float模型上識別正確的，在 Quantized模型上會識別錯，可見 ``Quantization`` 對模型的識別精度還是有影響的。在邊緣設備上資源有限，需要在模型大小、運行速度與識別精度上找到一個權衡。

總結
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
本節介紹了如何從零開始部署TFLite到Android應用中，包括：

#. 如何將訓練好的MNIST SavedModel模型，轉換爲Float模型和 Quantized模型
#. 如何使用 ``visualize.py`` 和解讀其結果信息
#. 如何將轉換後的模型部署到Android應用中

筆者剛開始寫這部分內容的時候還是TF1.0，在最近（2019年10月初）跟TF2.0的時候，發現有了很多變化，整體上是比原來更簡單了。不過文檔部分很多還是講的比較模糊，很多地方還是需要看源碼摸索。

.. hint::
    本節Android相關代碼存放路徑：
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
