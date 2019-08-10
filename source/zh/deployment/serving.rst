TensorFlow Serving
==========================

当我们将模型训练完毕后，往往需要将模型在生产环境中部署。最常见的方式，是在服务器上提供一个API，即客户机向服务器的某个API发送特定格式的请求，服务器收到请求数据后通过模型进行计算，并返回结果。如果仅仅是做一个Demo，不考虑高并发和性能问题，其实配合 `Flask <https://palletsprojects.com/p/flask/>`_ 等Python下的Web框架就能非常轻松地实现服务器API。不过，如果是在真的实际生产环境中部署，这样的方式就显得力不从心了。这时，TensorFlow为我们提供了TensorFlow Serving这一组件，能够帮助我们在实际生产环境中灵活且高性能地部署机器学习模型。

TensorFlow Serving安装
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow Serving可以使用apt-get或Docker安装。在生产环境中，推荐 `使用Docker部署TensorFlow Serving <https://www.tensorflow.org/tfx/serving/docker>`_ 。不过此处出于教学目的，介绍依赖环境较少的 `apt-get安装 <https://www.tensorflow.org/tfx/serving/setup#installing_using_apt>`_ 。

.. warning:: 软件的安装方法往往具有时效性，本节的更新日期为2019年8月。若遇到问题，建议参考 `TensorFlow网站上的最新安装说明 <https://www.tensorflow.org/tfx/serving/setup>`_ 进行操作。

首先设置安装源：

::

    # 添加Google的TensorFlow Serving源
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
    # 添加gpg key
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

更新源后，即可使用apt-get安装TensorFlow Serving

::

    sudo apt-get update
    sudo apt-get install tensorflow-model-server

.. hint:: 在运行curl和apt-get命令时，可能需要设置代理。

    curl设置代理的方式为 ``-x`` 选项或设置 ``http_proxy`` 环境变量，即

    ::

        export http_proxy=http://代理服务器IP:端口

    或

    ::

        curl -x http://代理服务器IP:端口 URL

    apt-get设置代理的方式为 ``-o`` 选项，即

    ::

        sudo apt-get -o Acquire::http::proxy="http://代理服务器IP:端口" ...

    Windows 10下，可以在 `Linux子系统（WSL） <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ 内使用相同的方式安装TensorFlow Serving。

TensorFlow Serving模型部署
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow Serving可以直接读取SavedModel格式的模型进行部署（导出模型到SavedModel文件的方法见 :ref:`前文 <savedmodel>` ）。使用以下命令即可：

::

    tensorflow_model_server \
        --rest_api_port=端口号（如8501） \
        --model_name=模型名 \
        --model_base_path="SavedModel格式模型的文件夹绝对地址（不含版本号）"

.. note:: TensorFlow Serving支持热更新模型，其典型的模型文件夹结构如下：

    ::
        
        /saved_model_files
            /1      # 版本号为1的模型文件
                /assets
                /variables
                saved_model.pb
            ...
            /N      # 版本号为N的模型文件
                /assets
                /variables
                saved_model.pb

    
    上面1~N的子文件夹代表不同版本号的模型。当指定 ``--model_base_path`` 时，只需要指定根目录的 **绝对地址** （不是相对地址）即可。例如，如果上述文件夹结构存放在 ``home/snowkylin`` 文件夹内，则 ``--model_base_path`` 应当设置为 ``home/snowkylin/saved_model_files`` （不附带模型版本号）。TensorFlow Serving会自动选择版本号最大的模型进行载入。 

Keras Sequential模式模型的部署
---------------------------------------------------

由于Sequential模式的输入和输出都很固定，因此这种类型的模型很容易部署，无需其他额外操作。例如，要将 :ref:`前文使用SavedModel导出的MNIST手写体识别模型 <savedmodel>` （使用Keras Sequential模式建立）以 ``MLP`` 的模型名在 ``8501`` 端口进行部署，可以直接使用以下命令：

::

    tensorflow_model_server \
        --rest_api_port=8501 \
        --model_name=MLP \
        --model_base_path="/home/.../.../saved"  # 文件夹绝对地址根据自身情况填写，无需加入版本号

然后就可以按照 :ref:`后文的介绍 <call_serving_api>` ，使用gRPC或者RESTful API在客户端调用模型了。

自定义Keras模型的部署
---------------------------------------------------

使用继承 ``tf.keras.Model`` 类建立的自定义Keras模型的自由度相对更高。因此当使用TensorFlow Serving部署模型时，对导出的SavedModel文件也有更多的要求：

- 需要导出到SavedModel格式的方法（比如 ``call`` ）不仅需要使用 ``@tf.function`` 修饰，还要在修饰时指定 ``input_signature`` 参数，以显式说明输入的形状。该参数传入一个由 ``tf.TensorSpec`` 组成的列表，指定每个输入张量的形状和类型。例如，对于MNIST手写体数字识别，我们的输入是一个 ``[None, 28, 28, 1]`` 的四维张量（ ``None`` 表示第一维即Batch Size的大小不固定），此时我们可以将模型的 ``call`` 方法做以下修饰：

.. code-block:: python
    :emphasize-lines: 4

    class MLP(tf.keras.Model):
        ...

        @tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)])
        def call(self, inputs):
            ...

- 在将模型使用 ``tf.saved_model.save`` 导出时，需要通过 ``signature`` 参数提供待导出的函数的签名（Signature）。简单说来，由于自定义的模型类里可能有多个方法都需要导出，因此，需要告诉TensorFlow Serving每个方法在被客户端调用时分别叫做什么名字。例如，如果我们希望客户端在调用模型时使用 ``call`` 这一签名来调用 ``model.call`` 方法时，我们可以在导出时传入 ``signature`` 参数，以 ``dict`` 的键值对形式告知导出的方法对应的签名，代码如下：

.. code-block:: python
    :emphasize-lines: 3

    model = MLP()
    ...
    tf.saved_model.save(model, "saved_with_signature/1", signatures={"call": model.call})

以上两步均完成后，即可使用以下命令部署：

::

    tensorflow_model_server \
        --rest_api_port=8501 \
        --model_name=MLP \
        --model_base_path="/home/.../.../saved_with_signature"  # 修改为自己模型的绝对地址

.. _call_serving_api:

在客户端调用以TensorFlow Serving部署的模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/tfx/serving/api_rest
    http://www.ruanyifeng.com/blog/2014/05/restful_api.html

TensorFlow Serving支持以gRPC和RESTful API调用以TensorFlow Serving部署的模型。本手册主要介绍较为通用的RESTful API方法。

RESTful API以标准的HTTP POST方法进行交互，请求和回复均为JSON对象。为了调用服务器端的模型，我们在客户端向服务器发送以下格式的请求：

服务器URI： ``http://服务器地址:端口号/v1/models/模型名:predict`` 

请求内容：

::

    {
        "signature_name": "需要调用的函数签名（Sequential模式不需要）",
        "instances": 输入数据
    }

回复为：

::

    {
        "predictions": 返回值
    }

Python客户端示例
------------------------------------------------------

以下示例使用 `Python的Requests库 <https://2.python-requests.org//zh_CN/latest/user/quickstart.html>`_ （你可能需要使用 ``pip install requests`` 安装该库）向本机的TensorFlow Serving服务器发送MNIST测试集的前10幅图像并返回预测结果，同时与测试集的真实标签进行比较。

.. literalinclude:: /_static/code/zh/savedmodel/keras/client.py

输出：

::

    [7 2 1 0 4 1 4 9 6 9]
    [7 2 1 0 4 1 4 9 5 9]

可见预测结果与真实标签值非常接近。

对于自定义的Keras模型，在发送的数据中加入 ``signature_name`` 键值即可，即将上面代码的 ``data`` 建立过程改为

.. literalinclude:: /_static/code/zh/savedmodel/custom/client.py
    :lines: 8-11

Node.js客户端示例（Ziyang）
------------------------------------------------------

以下示例使用 `Node.js <https://nodejs.org/zh-cn/>`_ 将下图转换为28*28的灰度图，发送给本机的TensorFlow Serving服务器，并输出返回的预测值和概率。（其中使用了 `图像处理库jimp <https://github.com/oliver-moran/jimp>`_ 和 `HTTP库superagent <https://visionmedia.github.io/superagent/>`_ ，可使用 ``npm install jimp`` 和 ``npm install superagent`` 安装）

.. figure:: /_static/image/serving/test_pic_tag_5.png
    :align: center

    ``test_pic_tag_5.png`` ：一个由作者手写的数字5。（运行下面的代码时可下载该图片并放在与代码同一目录下）

.. literalinclude:: /_static/code/zh/savedmodel/keras/client.js
    :language: javascript

运行结果为：

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
    我们猜这个数字是5，概率是0.846008837

可见输出结果符合预期。

.. note:: 如果你不熟悉HTTP POST，可以参考 `这里 <https://www.runoob.com/tags/html-httpmethods.html>`_ 。事实上，当你在用浏览器填写表单（比方说性格测试）并点击“提交”按钮，然后获得返回结果（比如说“你的性格是ISTJ”）时，就很有可能是在向服务器发送一个HTTP POST请求并获得了服务器的回复。

    RESTful API是一个流行的API设计理论，可以参考 `这里 <http://www.ruanyifeng.com/blog/2014/05/restful_api.html>`_ 获得简要介绍。

    关于TensorFlow Serving的RESTful API的完整使用方式可参考 `文档 <https://www.tensorflow.org/tfx/serving/api_rest>`_ 。



