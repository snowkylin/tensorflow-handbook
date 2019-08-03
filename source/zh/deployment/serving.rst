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

TensorFlow Serving可以直接读取 :ref:`SavedModel格式的模型 <savedmodel>` 进行部署。使用以下命令即可：

::

    tensorflow_model_server \
        --rest_api_port=端口号（如8501） \
        --model_name=模型名 \
        --model_base_path="SavedModel格式模型的文件夹绝对地址"

.. literalinclude:: /_static/code/zh/savedmodel/train_and_export.py
    :emphasize-lines: 15, 36-37

::

    tensorflow_model_server \
        --rest_api_port=8501 \
        --model_name=MLP \
        --model_base_path="/home/.../.../saved_with_signature"  # 文件夹绝对地址根据自身情况填写

.. literalinclude:: /_static/code/zh/savedmodel/client.py

