TensorFlow Lite（Jinpeng）
====================================================

在移动和IoT等边缘设备端，TensorFlow提供了基础平台TensorFlow Lite，提供了Java、Python、C++ API库，可以运行在Android、iOS和Raspberry Pi等设备上。2019年是5G元年，万物互联的时代已经来临，作为TensorFlow在边缘设备上的基础设施，TFLite将会是愈发重要的角色。

目前TFLite只提供了推理功能，在服务器端进行训练后，可以经过简单处理后部署到边缘设备上。

模型转换：由于边缘设备计算等资源有限，使用TensorFlow训练好的模型，模型太大、运行效率比较低，不能直接在移动端部署，需要通过相应工具进行转换成适合边缘设备的格式。

边缘设备部署：本节以android为例，简单介绍如何在android应用中部署转化后的模型，完成Mnist图片的识别。

模型转换
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
转换工具有两种：
命令行工具
Python API

TF2.0对模型转换工具发生了非常大的变化，推荐大家使用Python API进行转换，命令行工具只提供了基本的转化功能。转换后的原模型为``FlatBuffers``格式。``FlatBuffers``原来主要应用于游戏场景，是谷歌为了高性能场景创建的序列化库，相比Protocol Buffer有更高的性能和大小优势，更适合于端侧部署。

转换方式有两种：``Float格式``和``Quantized格式``。

为了熟悉两种方式我们都会使用，针对Float格式的，先使用命令行工具``tflite_convert``，其跟随TensorFlow一起安装（见`一般安装步骤<https://tf.wiki/zh/basic/installation.html#id1>`_）。 

在终端执行如下命令::

    tflite_convert -h

输出结果如下，即该命令的使用方法::

    usage: tflite_convert [-h] --output_file OUTPUT_FILE
                          (--saved_model_dir SAVED_MODEL_DIR | --keras_model_file KERAS_MODEL_FILE)
    
    Command line tool to run TensorFlow Lite Converter.
    
    optional arguments:
      -h, --help            show this help message and exit
      --output_file OUTPUT_FILE
                            Full filepath of the output file.
      --saved_model_dir SAVED_MODEL_DIR
                            Full path of the directory containing the SavedModel.
      --keras_model_file KERAS_MODEL_FILE
                            Full filepath of HDF5 file containing tf.Keras model.

在`TensorFlow模型导出<https://tf.wiki/zh/deployment/export.html>`_中，我们知道TF2.0支持两种模型导出方法和格式SavedModel（saved/1）和Keras Sequential（mnist_cnn.h5）。

SavedModel导出模型转换：
.. code-block:: bash

    tflite_convert --saved_model_dir=saved/1 --output_file=mnist_savedmodel.tflite

Keras Sequential导出模型转换：

.. code-block:: bash

    tflite_convert --keras_model_file=mnist_cnn.h5 --output_file=mnist_sequential.tflite

到此，已经得到两个TensorFlow Lite模型。因为两者后续操作基本一致，我们只处理SavedModel格式的，Keras Sequential请自行探索。

Android部署
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

现在开始在Android环境部署，对于国内的读者，因为获取SDK和gradle编译环境等资源，需要先给Android Studio配置proxy或者使用国内的镜像，请自行查看代码中的配置，或自行百度或谷歌。

**配置app/build.gradle**

新建一个Android Project，打开 ``app/build.gradle`` 添加如下信息::

    android {
        aaptOptions {
            noCompress "tflite" // 编译apk时，不压缩tflite文件
        }
    }

    dependencies {
        implementation 'org.tensorflow:tensorflow-lite:+' // +号代表获取最新版本
    }

其中，

#. ``aaptOptions`` 设置tflite文件不压缩，确保后面tflite文件可以被Interpreter正确加载。
#. ``org.tensorflow:tensorflow-lite`` 的最新版本号可以在这里查询 https://bintray.com/google/tensorflow/tensorflow-lite

设置好后，sync和build整个工程，如果build成功说明，配置成功。

**添加tflite文件到assets文件夹**

在app目录先新建assets目录，并将 ``mnist_savedmodel.tflite`` 文件保存到assets目录。重新编译apk，检查新编译出来的apk的assets文件夹是否有 ``mnist_cnn.tflite`` 文件。

点击菜单Build->Build APK(s)触发apk编译，apk编译成功点击右下角的EventLog。点击最后一条信息中的`analyze`链接，会触发apk analyzer查看新编译出来的apk，若在assets目录下存在``mnist_savedmodel.tflite``则编译打包成功，如下::

    assets
         |__mnist_savedmodel.tflite

**加载模型**

使用如下函数将 ``mnist_savedmodel.tflite`` 文件加载到memory-map中，作为Interpreter实例化的输入

.. code-block:: java

    private static final String MODEL_PATH = "mnist_savedmodel.tflite";

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

实例化Interpreter，其中this为当前acitivity

.. code-block:: java

    tflite = new Interpreter(loadModelFile(this));

**运行输入**

我们使用mnist test测试集中的某张图片作为输入，mnist图像大小28*28，单像素。这样我们输入的数据需要设置成如下格式

.. code-block:: java

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    private ByteBuffer imgData = null;

    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 1;

    private static final int DIM_IMG_WIDTH = 28;
    private static final int DIM_IMG_HEIGHT = 28;

    protected void onCreate() {
        imgData = ByteBuffer.allocateDirect(
            4 * DIM_BATCH_SIZE * DIM_IMG_WIDTH * DIM_IMG_HEIGHT * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());
    }

将mnist图片转化成 ``ByteBuffer`` ，并保持到 ``imgData`` 中

.. code-block:: java

    /** Preallocated buffers for storing image data in. */
    private int[] intValues = new int[DIM_IMG_WIDTH * DIM_IMG_HEIGHT];

    /** Writes Image data into a {@code ByteBuffer}. */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }

        // Rewinds this buffer. The position is set to zero and the mark is discarded.
        imgData.rewind();

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_WIDTH; ++i) {
            for (int j = 0; j < DIM_IMG_HEIGHT; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat(val);
            }
        }
    }

``convertBitmapToByteBuffer`` 的输出即为模型运行的输入。

**运行输出**

定义一个1*10的多维数组，因为我们只有1个batch和10个label（TODO：need double check），具体代码如下

.. code-block:: java

    private float[][] labelProbArray = new float[1][10];

运行结束后，每个二级元素都是一个label的概率。

**运行及结果处理**

开始运行模型，具体代码如下

.. code-block:: java

    tflite.run(imgData, labelProbArray);

针对某个图片，运行后 ``labelProbArray`` 的内容如下，也就是各个label识别的概率

.. code-block:: java

    index 0 prob is 0.0
    index 1 prob is 0.0
    index 2 prob is 0.0
    index 3 prob is 1.0
    index 4 prob is 0.0
    index 6 prob is 0.0
    index 7 prob is 0.0
    index 8 prob is 0.0
    index 9 prob is 0.0

接下来，我们要做的就是根据对这些概率进行排序，找出Top的label并界面呈现给用户.


Quantization 模型转换（稍后处理）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

还有一种quantization的转化方法，这种转化命令如下：

.. code-block:: bash

    tflite_convert \
      --output_file=keras_mnist_quantized_uint8.tflite \
      --keras_model_file=mnist_cnn.h5 \
      --inference_type=QUANTIZED_UINT8 \
      --mean_values=128 \
      --std_dev_values=127 \
      --default_ranges_min=0 \
      --default_ranges_max=255 \
      --input_arrays=conv2d_1_input \
      --output_arrays=dense_2/Softmax

细心的读者肯定会问，上图中有很多参数是怎么来的呢？我们可以使用 ``tflite_convert`` 获得模型具体结构，命令如下：

.. code-block:: bash

    tflite_convert \
      --output_file=keras_mnist.dot \
      --output_format=GRAPHVIZ_DOT \
      --keras_model_file=mnist_cnn.h5

dot是一种graph description language，可以用graphviz的dot命令转化为pdf或png等可视化图。

.. code-block:: bash

    dot -Tpng -O keras_mnist.dot

这样就转化为一张图了，如下：

.. figure:: /_static/image/deployment/keras_mnist.dot.png
    :width: 80%
    :align: center

很明显的可以看到如下信息：

入口：

.. code-block:: bash

    conv2d_1_input
    Type: Float [1×28×28×1]
    MinMax: [0, 255]

出口：

.. code-block:: bash

    dense_2/Softmax
    Type: Float [1×10]

因此，可以知道

``--input_arrays`` 就是 ``conv2d_1_input``

``--output_arrays`` 就是 ``dense_2/Softmax``

``--default_ranges_min`` 就是 ``0``

``--default_ranges_max`` 就是 ``255``


关于 ``--mean_values`` 和 ``--std_dev_values`` 的用途::

    QUANTIZED_UINT8的quantized模型期望的输入是[0,255], 需要有个跟原始的float类型输入有个对应关系。

    mean_values和std_dev_values就是为了实现这个对应关系

    mean_values对应float的float_min

    std_dev_values对应255 / (float_max - float_min)

因此，可以知道

``--mean_values`` 就是 ``0``

``--std_dev_values`` 就是 ``1``