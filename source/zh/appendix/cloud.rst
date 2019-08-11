在云端使用TensorFlow
============================================

.. _colab:

在Colab中使用TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Google Colab是谷歌的免费在线交互式Python运行环境，且提供GPU支持，使得机器学习开发者们无需在自己的电脑上安装环境，就能随时随地从云端访问和运行自己的机器学习代码。

.. admonition:: 学习资源

    - `Colab官方教程 <https://colab.research.google.com/notebooks/welcome.ipynb>`_
    - `Google Colab Free GPU Tutorial <https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d>`_ （`中文翻译 <https://juejin.im/post/5c05e1bc518825689f1b4948>`_）

进入Colab（https://colab.research.google.com），新建一个Python3笔记本，界面如下：

.. figure:: /_static/image/colab/new.png
    :width: 100%
    :align: center

如果需要使用GPU，则点击菜单“代码执行程序-更改运行时类型”，在“硬件加速器”一项中选择“GPU”，如下图所示：

.. figure:: /_static/image/colab/select_env.png
    :width: 40%
    :align: center

我们在主界面输入一行代码，例如 ``import tensorflow as tf`` ，然后按 ``ctrl + enter`` 执行代码（如果直接按下 ``enter`` 是换行，可以一次输入多行代码并运行）。此时Colab会自动连接到云端的运行环境，并将状态显示在右上角。

运行完后，点击界面左上角的“+代码”，此时界面上会新增一个输入框，我们输入 ``tf.__version__`` ，再次按下 ``ctrl + enter`` 执行代码，以查看Colab默认的TensorFlow版本，执行情况如下：

.. figure:: /_static/image/colab/tf_version.png
    :width: 100%
    :align: center

.. tip:: Colab支持代码提示，可以在输入 ``tf.`` 后按下 ``tab`` 键，即会弹出代码提示的下拉菜单。

可见，截至本文写作时，Colab中的TensorFlow默认版本是1.14.0。在Colab中，可以使用 ``!pip install`` 或者 ``!apt-get install`` 来安装Colab中尚未安装的Python库或Linux软件包。比如在这里，我们希望使用TensorFlow 2.0 beta1版本，即点击左上角的“+代码”，输入::

    !pip install tensorflow-gpu==2.0.0-beta1

按下 ``ctrl + enter`` 执行，结果如下：

.. figure:: /_static/image/colab/install_tf.png
    :width: 100%
    :align: center

可见，Colab提示我们重启运行环境以使用新安装的TensorFlow版本。于是我们点击运行框最下方的Restart Runtime（或者菜单“代码执行程序-重新启动代码执行程序”），然后再次导入TensorFlow并查看版本，结果如下：

.. figure:: /_static/image/colab/view_tf_version.png
    :width: 100%
    :align: center

我们可以使用 ``tf.test.is_gpu_available`` 函数来查看当前环境的GPU是否可用：

.. figure:: /_static/image/colab/view_gpu.png
    :width: 100%
    :align: center

可见，我们成功在Colab中配置了TensorFlow 2.0环境并启用了GPU支持。

你甚至可以通过 ``!nvidia-smi`` 查看当前的GPU信息：

.. figure:: /_static/image/colab/nvidia_smi.png
    :width: 100%
    :align: center

可见GPU的型号为Tesla T4。

.. _GCP:

在Google Cloud Platform（GCP）中使用TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://medium.com/@kstseng/%E5%9C%A8-google-cloud-platform-%E4%B8%8A%E4%BD%BF%E7%94%A8-gpu-%E5%92%8C%E5%AE%89%E8%A3%9D%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E7%9B%B8%E9%97%9C%E5%A5%97%E4%BB%B6-1b118e291015
    
`Google Cloud Platform（GCP） <https://cloud.google.com/>`_ 是Google的云计算服务。GCP收费灵活，默认按时长计费。也就是说，你可以迅速建立一个带GPU的实例，训练一个模型，然后立即关闭（关机或删除实例）。GCP只收取在实例开启时所产生的费用，关机时只收取磁盘存储的费用，删除后即不再继续收费。

我们可以通过两种方式在GCP中使用TensorFlow：使用Compute Engine建立带GPU的实例，或使用AI Platform中的Notebook建立带GPU的在线JupyterLab环境。

在Compute Engine建立带GPU的实例并部署TensorFlow
----------------------------------------------------------------

GCP的Compute Engine类似于AWS、阿里云等，允许用户快速建立自己的虚拟机实例。在Compute Engine中，可以很方便地建立具有GPU的虚拟机实例，只需要进入Compute Engine的VM实例（https://console.cloud.google.com/compute/instances），并在创建实例的时候选择GPU类型和数量即可。

.. figure:: /_static/image/gcp/create_instance.png
    :width: 100%
    :align: center

需要注意两点：

1. 只有特定区域的机房具有GPU，且不同类型的GPU地区范围也不同，可参考 `GCP官方文档 <https://cloud.google.com/compute/docs/gpus>`_ 并选择适合的地区建立实例；
#. 默认情况下GCP账号的GPU配额非常有限（可能是怕你付不起钱？）。你很可能需要在使用前申请提升自己账号在特定地区的特定型号GPU的配额，可参考 `GCP官方文档：申请提升配额 <https://cloud.google.com/compute/quotas?hl=zh-cn#requesting_additional_quota>`_ ，GCP会有工作人员手动处理申请，并给你的邮箱发送邮件通知，大约需要数小时至两个工作日不等。

当建立好具有GPU的GCP虚拟机实例后，配置工作与在本地基本相同。系统中默认并没有NVIDIA显卡驱动，依然需要自己安装。

以下命令示例了在Tesla K80，Ubuntu 18.04 LTS的GCP虚拟机实例中配置NVIDIA 410驱动、CUDA 10.0、cuDNN 7.6.0以及TensorFlow 2.0 beta环境的过程：

.. code-block:: bash

    sudo apt-get install build-essential    # 安装编译环境
    wget http://us.download.nvidia.com/tesla/410.104/NVIDIA-Linux-x86_64-410.104.run    # 下载NVIDIA驱动
    sudo bash NVIDIA-Linux-x86_64-410.104.run   # 安装驱动（一路Next）
    # nvidia-smi  # 查看虚拟机中的GPU型号
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  # 下载Miniconda
    bash Miniconda3-latest-Linux-x86_64.sh      # 安装Miniconda（安装完需要重启终端）
    conda create -n tf2.0-beta-gpu python=3.6
    conda activate tf2.0-beta-gpu
    conda install cudatoolkit=10.0
    conda install cudnn=7.6.0
    pip install tensorflow-gpu==2.0.0-beta1

输入 ``nvidia-smi`` 会显示：

.. code-block:: bash

    ~$ nvidia-smi
    Fri Jul 12 10:30:37 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.104      Driver Version: 410.104      CUDA Version: 10.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   63C    P0    88W / 149W |      0MiB / 11441MiB |    100%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

.. _notebook:

使用AI Platform中的Notebook建立带GPU的在线JupyterLab环境
----------------------------------------------------------------

如果你不希望繁杂的配置，希望迅速获得一个开箱即用的在线交互式Python环境，可以使用GCP的AI Platform中的Notebook。其预安装了JupyterLab，可以理解为Colab的付费升级版，具备更多功能且限制较少。

进入 https://console.cloud.google.com/mlengine/notebooks ，点击“新建实例-TensorFlow 2.0-With 1 NVIDIA Tesla K80”，界面如下：

.. figure:: /_static/image/gcp/create_notebook.png
    :width: 100%
    :align: center

也可以点击“自定义”来进一步配置实例，例如选择区域、GPU类型和个数，与创建Compute Engine实例类似。

.. hint:: 和Compute Engine实例一样，你很可能需要在这里选择自己适合的区域，以及申请提升自己账号在特定地区的特定型号GPU的配额。

建立完成后，点击“打开JUPYTERLAB”，即可进入以下界面：

.. figure:: /_static/image/gcp/notebook_index.png
    :width: 100%
    :align: center

建立一个Python 3笔记本，测试TensorFlow环境：

.. figure:: /_static/image/gcp/notebook_test.png
    :width: 100%
    :align: center

我们还可以点击左上角的“+”号，新建一个终端：

.. figure:: /_static/image/gcp/notebook_terminal.png
    :width: 100%
    :align: center

在阿里云上使用 GPU 实例运行 Tensorflow（Ziyang）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

国内也有部分云服务商（如 `阿里云 <https://cn.aliyun.com/product/ecs/gpu>`_ 和 `腾讯云 <https://cloud.tencent.com/product/gpu>`_ ）提供了 GPU 实例，且可按量计费。至本手册撰写时，具备单个GPU的实例价格在数元（Tesla P4）至二十多元（Tesla V100）每小时不等。以下我们简要介绍在阿里云使用 GPU 实例。

.. hint:: 根据不同的地区、配置和付费方式，实例的价格也是多样化的，请根据需要合理选择。如果是临时需要的计算任务，可以考虑按量付费以及使用抢占式VPS，以节约资金。

访问 https://cn.aliyun.com/product/ecs/gpu ，点击购买，界面如下：

.. figure:: /_static/image/aliyun/vps_select.png
    :width: 100%
    :align: center

此处，我们选择一个带有 Tesla P4 计算卡的实例。

在系统镜像中，阿里云提供多种选择，可以根据需要选择合适的镜像。

.. figure:: /_static/image/aliyun/os_image_config_with_driver.png
    :width: 100%
    :align: center

    如果选择“公共镜像”，可以根据提示选择提前预装GPU驱动，可以避免后续安装驱动的麻烦。

.. figure:: /_static/image/aliyun/os_image_with_RAPIDS.png
    :width: 100%
    :align: center

    在“镜像市场”中，官方也提供了适合深度学习的定制镜像。在本示例中我们选择预装了 NVIDIA RAPIDS 的 Ubuntu 16.04 镜像。 

然后，通过 ssh 连接上我们选购的服务器，并使用 ``nvidia-smi`` 查看 GPU 信息：

.. code-block:: bash

    (rapids) root@iZ8vb2567465uc1ty3f4ovZ:~# nvidia-smi
    Sun Aug 11 23:53:52 2019
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla P4            On   | 00000000:00:07.0 Off |                    0 |
    | N/A   29C    P8     6W /  75W |      0MiB /  7611MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

确认了驱动无误之后，其他操作就可以照常执行了。

.. hint:: 阿里云这类国内的云服务提供商一般对于 VPS 的端口进行了安全策略限制，请关注所使用的端口是否在安全策略的放行列表中，以免影响Tensorflow Serving和Tensorboard的使用。