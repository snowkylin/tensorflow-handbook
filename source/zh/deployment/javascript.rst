TensorFlow in JavaScript（Huan）
==========================================================

    **Atwood’s Law**
     
    “Any application that can be written in JavaScript, will eventually be written in JavaScript.”
     
     -- Jeff Atwood, Founder of StackOverflow.com


    “JavaScript now works.”
     
     -- Paul Graham, YC Founder

TensorFlow.js 简介
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /_static/image/javascript/tensorflow-js.gif
    :width: 60%
    :align: center

TensorFlow.js 是 TensorFlow 的 JavaScript 版本，支持GPU硬件加速，可以运行在 Node.js 或浏览器环境中。它不但支持完全基于 JavaScript 从头开发、训练和部署模型，也可以用来运行已有的 Python 版 TensorFlow 模型，或者基于现有的模型进行继续训练。

.. figure:: /_static/image/javascript/architecture.gif
    :width: 60%
    :align: center

TensorFlow.js 支持 GPU 硬件加速。在 Node.js 环境中，如果有 CUDA 环境支持，或者在浏览器环境中，有 WebGL 环境支持，那么 TensorFlow.js 可以使用硬件进行加速。

微信小程序也提供了官方插件，封装了TensorFlow.js库，利用小程序WebGL API给第三方小程序调用时提供GPU加速。

本章，我们将基于 TensorFlow.js 1.0，向大家简单的介绍如何基于 ES6 的 JavaScript 进行 TensorFlow.js 的开发，然后提供两个例子，并基于例子进行详细的讲解和介绍，最终实现使用纯 JavaScript 进行 TensorFlow 模型的开发、训练和部署。

本章中提到的 JavaScript 版 TensorFlow 的相关代码，使用说明，和训练好的模型文件及参数，都可以在作者的 GitHub 上找到。地址： https://github.com/huan/tensorflow-handbook-javascript

浏览器中使用 TensorFlow.js 的优势
--------------------------------------------

.. figure:: /_static/image/javascript/chrome-ml.png
    :width: 60%
    :align: center

TensorFlow.js 可以让我们直接在浏览器中加载 TensorFlow，让用户立即通过本地的CPU/GPU资源进行我们所需要的机器学习运算，更灵活的进行AI应用的开发。

浏览器中进行机器学习，相对比与服务器端来讲，将拥有以下四大优势：

* 不需要安装软件或驱动（打开浏览器即可使用）；
* 可以通过浏览器进行更加方便的人机交互；
* 可以通过手机浏览器，调用手机硬件的各种传感器（如：GPS、电子罗盘、加速度传感器、摄像头等）；
* 用户的数据可以无需上传到服务器，在本地即可完成所需操作。

通过这些优势，TensorFlow.js 将带给开发者带来极高的灵活性。比如在 Google Creative Lab 在2018年7月发布的 Move Mirror 里，我们可以在手机上打开浏览器，通过手机摄像头检测视频中用户的身体动作姿势，然后通过对图片数据库中类似身体动作姿势的检索，给用户显示一个最能够和他当前动作相似的照片。在Move Mirror的运行过程中，数据没有上传到服务器，所有的运算都是在手机本地，基于手机的CPU/GPU完成的，而这项技术，将使Servreless与AI应用结合起来成为可能。

.. figure:: /_static/image/javascript/move-mirror.jpg
    :width: 60%
    :align: center

- Move Mirror 地址：https://experiments.withgoogle.com/move-mirror
- Move Mirror 所使用的 PoseNet 地址：https://github.com/tensorflow/tfjs-models/tree/master/posenet

TensorFlow.js 性能对比
--------------------------------------------

TensorFlow.js 的性能如何，Google 官方做了一份基于 MobileNet 的评测，可以作为参考。具体评测是基于 MobileNet 的 TensorFlow 模型，将其 JavaScript 版本和 Python 版本各运行两百次。

其评测结论如下。

手机浏览器性能：

.. figure:: /_static/image/javascript/performance-mobile.png
    :width: 60%
    :align: center

TensorFlow.js 在手机浏览器中运行一次推理：

1. 在 IPhoneX 上需要时间为 22ms
1. 在 Pixel3 上需要时间为 100ms

与 TensorFlow Lite 代码基准相比，手机浏览器中的 TensorFlow.js 在 IPhoneX 上的运行时间为基准的1.2倍，在 Pixel3 上运行的时间为基准的 1.8 倍。

台式机浏览器性能：

在浏览器中，TensorFlow.js 可以使用 WebGL 进行硬件加速，将 GPU 资源使用起来。

.. figure:: /_static/image/javascript/performance-browser.gif
    :width: 60%
    :align: center

TensorFlow.js 在浏览器中运行一次推理：

* 在 CPU 上需要时间为 97ms
* 在 GPU (WebGL)上需要时间为 10ms

与 Python 代码基准相比，浏览器中的 TensorFlow.js 在 CPU 上的运行时间为基准的1.7倍，在 GPU (WebGL) 上运行的时间为基准的3.8倍。

Node.js 性能：

在 Node.js 中，TensorFlow.js 使用 TensorFlow 的 C Binding ，所以基本上可以达到和 Python 接近的效果。

.. figure:: /_static/image/javascript/performance-node.png
    :width: 60%
    :align: center

TensorFlow.js 在 Node.js 运行一次推理：

* 在 CPU 上需要时间为56ms
* 在 GPU(CUDA) 上需要时间为14ms

与 Python 代码基准相比，Node.js 的 TensorFlow.js 在 CPU 上的运行时间与基准相同，在 GPU（CUDA） 上运行的时间是基准的1.6倍。


TensorFlow.js 环境配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在浏览器中使用 TensorFlow.js
--------------------------------------------

在 Node.js 中使用 TensorFlow.js
--------------------------------------------

服务器端使用 JavaScript ，首先需要按照 `NodeJS.org <https://nodejs.org>`_ 官网的说明，完成安装最新版本的 Node.js 。

然后，完成以下四个步骤即可完成配置：

1. 确认 Node.js 版本（v10，或更新）::

    $ node --verion
    v10.5.0

    $ npm --version
    6.4.1

2. 建立 TensorFlow.js 项目目录::

    $ mkdir tfjs
    $ cd tfjs

3. 安装 TensorFlow.js::

    # 初始化项目管理文件 package.json
    $ npm init -y

    # 安装 tfjs 库，纯 JavaScript 版本
    $ npm install @tensorflow/tfjs 

    # 安装 tfjs-node 库，C Binding 版本
    $ npm install @tensorflow/tfjs-node 

    # 安装 tfjs-node-gpu 库，支持 CUDA GPU 加速
    $ npm install @tensorflow/tfjs-node-gpu

4. 确认 Node.js 和 TensorFlow.js 工作正常::

    $ node
    > require('@tensorflow/tfjs').version
    {
        'tfjs-core': '1.3.1',
        'tfjs-data': '1.3.1',
        'tfjs-layers': '1.3.1',
        'tfjs-converter': '1.3.1',
        tfjs: '1.3.1'
    }
    > 

如果你看到了上面的 ``tfjs-core``, ``tfjs-data``, ``tfjs-layers`` 和 ``tfjs-converter`` 的输出信息，那么就说明环境配置没有问题了。

在微信小程序中使用 TensorFlow.js
--------------------------------------------

TensorFlow.js 微信小程序插件封装了 TensorFlow.js 库，用于提供给第三方小程序调用。

在使用插件前，首先要在小程序管理后台的“设置-第三方服务-插件管理”中添加插件。开发者可登录小程序管理后台，通过 appid _wx6afed118d9e81df9_ 查找插件并添加。本插件无需申请，添加后可直接使用。

例子可以看 TFJS Mobilenet: `物体识别小程序 <https://github.com/tensorflow/tfjs-wechat/tree/master/demo/mobilenet>`_

`TensorFlow.js 微信小程序官方文档地址 <https://mp.weixin.qq.com/wxopen/plugindevdoc?appid=wx6afed118d9e81df9>`_

.. admonition:: TensorFlow.js 微信小程序教程

    为了推动微信小程序中人工智能应用的发展，Google 专门为微信小程序打造了最新 TensorFlow.js 插件，并联合 Google 认证机器学习专家、微信、腾讯课堂 NEXT 学院，联合推出了“【NEXT学院】TensorFlow.js遇到小程序”课程，帮助小程序开发者带来更加易于上手和流畅的 TensorFlow.js 开发体验。

    本课程主要介绍了如何将 TensorFlow.js 插件嵌入到微信小程序中，并基于其进行开发。课程中以一个姿态检测的模型 PoseNet 作为案例，介绍了 TensorFlow.js 插件导入到微信小程序开发工具中后，在项目开发中的配置，功能调用，加载模型等方法应用；此外，还介绍了在 Python 环境下训练好的模型如何转换并载入到小程序中。

    本章作者也参与了课程制作，课程中的案列简单有趣易上手，通过学习，可以快速熟悉 TensorFlow.js 在小程序中的开发和应用.有兴趣的读者可以前往 NEXT 学院，进行后续深度学习。

    课程地址：<https://ke.qq.com/course/428263>


TensorFlow.js 模型部署
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

通过 TensorFlow.js 加载 Python 模型
--------------------------------------------

一般 TensorFlow 的模型，以 Python 版本为例，会被存储为以下三种格式之一：

* TensorFlow SavedModel
* TensorFlow Hub Module

Google 目前最佳实践中，推荐使用 SavedModel 方法进行模型保存。同时所有以上格式，都可以通过 tensorflowjs-converter 转换器，将其转换为可以直接被 TensorFlow.js 加载的格式，在JavaScript语言中进行使用。

``tensorflowjs_converter`` 可以将Python存储的模型格式，转换为JavaScript可以直接调用的模型格式。

安装 ``tensorflowjs_converter`` ::

    $ pip install tensorflowjs


``tensorflowjs_converter`` 的使用细节，可以通过 ``--help`` 参数查看程序帮助::

    $ tensorflowjs_converter --help

以下我们以 MobilenetV1 为例，看一下如何对模型文件进行转换操作，并将可以被 TensorFlow.js 加载的模型文件，存放到 ``/mobilenet/tfjs_model`` 目录下。

1. 转换 SavedModel：将 ``/mobilenet/saved_model`` 转换到 ``/mobilenet/tfjs_model`` ::

    tensorflowjs_converter \
        --input_format=tf_saved_model \
        --output_node_names='MobilenetV1/Predictions/Reshape_1' \
        --saved_model_tags=serve \
        /mobilenet/saved_model \
        /mobilenet/tfjs_model

2. 转换 TFHub：将 ``https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1`` 转换到 ``/mobilenet/tfjs_model`` ::

    tensorflowjs_converter \
        --input_format=tf_hub \
        'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' \
        /mobilenet/tfjs_model
    
转换完成的模型，保存为了两类文件：

    - `model.json` 模型架构
    - `group1-shard*of*` 模型参数

举例来说，我们对 MobileNet v2 转换出来的文件，如下：

    /mobilenet/tfjs_model/model.json
    /mobilenet/tfjs_model/group1-shard1of5
    ...
    /mobilenet/tfjs_model/group1-shard5of5


为了加载转换完成的模型文件，我们需要安装 ``tfjs-converter`` 和 ``@tensorflow/tfjs`` 模块::

    $ npm install @tensorflow/tfjs

然后，我们就可以通过 JavaScript 来加载 TensorFlow 模型了！

.. code-block:: javascript

    import * as tf from '@tensorflow/tfjs';

    const MODEL_URL = '/mobilenet/tfjs_model/model.json';

    const model = await tf.loadGraphModel(MODEL_URL);

    const cat = document.getElementById('cat');
    model.execute(tf.browser.fromPixels(cat))

使用 TensorFlow.js 模型库
--------------------------------------------

TensorFlow.js 提供了一系列预训练好的模型，方便大家快速的给自己的程序引入人工智能能力。

模型库 GitHub 地址：https://github.com/tensorflow/tfjs-models，其中模型分类包括图像识别、语音识别、人体姿态识别、物体识别、文字分类等。

由于这些API默认模型文件都存储在谷歌云上，直接使用会导致中国用户无法直接读取。在程序内使用模型API时要提供 modelUrl 的参数，可以指向谷歌中国的镜像服务器。

谷歌云的base url是 https://storage.googleapis.com， 中国镜像的base url是 https://www.gstaticcnapps.cn 模型的url path是一致的。以 posenet模型为例：

- 谷歌云地址是：**https://storage.googleapis.com**/tfjs-models/savedmodel/posenet/mobilenet/float/050/model-stride16.json
- 中国镜像地址是：**https://www.gstaticcnapps.cn**/tfjs-models/savedmodel/posenet/mobilenet/float/050/model-stride16.json

TensorFlow.js 模型训练 *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

与 TensorFlow Serving 和 TensorFlow Lite 不同，TensorFlow.js 不仅支持模型的部署和推断，还支持直接在 TensorFlow.js 中进行模型训练、

在 TensorFlow 基础章节中，我们已经用 Python 实现过，针对某城市在 2013-2017 年的房价的任务，通过对该数据进行线性回归，即使用线性模型 :math:`y = ax + b` 来拟合上述数据，此处 :math:`a` 和 :math:`b` 是待求的参数。

下面我们改用 TensorFlow.js 来实现一个 JavaScript 版本。

首先，我们定义数据，进行基本的归一化操作。

.. literalinclude:: /_static/code/zh/deployment/javascript/index.html
    :lines: 5-12

接下来，我们来求线性模型中两个参数 ``a`` 和 ``b`` 的值。

使用 ``loss()`` 计算损失；
使用 ``optimizer.minimize()`` 自动更新模型参数。

.. literalinclude:: /_static/code/zh/deployment/javascript/index.html
    :lines: 14-35

从下面的输出样例中我们可以看到，已经拟合的比较接近了。

::

    a: 0.9339302778244019, b: 0.08108722418546677
    x: 0, pred: 0.08, true: 0.00
    x: 1, pred: 0.31, true: 0.36
    x: 2, pred: 0.55, true: 0.55
    x: 3, pred: 0.78, true: 0.82
    x: 4, pred: 1.02, true: 1.00

可以直接在浏览器中运行，完整的 HTML 代码如下：

.. literalinclude:: /_static/code/zh/deployment/javascript/index.html
    :lines: 1-

.. admonition:: 在浏览器中和 Node 环境中引入 TFJS 的区别

    我们本节的例子中，是在浏览器中引用 TFJS 库。如果你需要在 Node 环境中操作，那么需要使用 `import * as tf from '@tensorflow/tfjs'` 替换掉 `<script src="http://unpkg.com/@tensorflow/tfjs/dist/tf.min.js"></script>` 进行 TFJS 引入。
