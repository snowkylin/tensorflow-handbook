在容器和云端使用TensorFlow
============================================

.. _install_by_docker:

使用Docker部署TensorFlow环境
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint:: 本部分面向没有Docker经验的读者。对于已熟悉Docker的读者，可直接参考 `TensorFlow官方文档 <https://www.tensorflow.org/install/docker>`_ 进行部署。

Docker是轻量级的容器（Container）环境，通过将程序放在虚拟的“容器”或者说“保护层”中运行，既避免了配置各种库、依赖和环境变量的麻烦，又克服了虚拟机资源占用多、启动慢的缺点。使用Docker部署TensorFlow的步骤如下：

1. 安装 `Docker <https://www.docker.com/>`_ 。Windows下，下载官方网站的安装包进行安装即可。Linux下建议使用 `官方的快速脚本 <https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-using-the-convenience-script>`_ 进行安装，即命令行下输入：

::

    wget -qO- https://get.docker.com/ | sh

如果当前的用户非root用户，可以执行 ``sudo usermod -aG docker your-user`` 命令将当前用户加入 ``docker`` 用户组。重新登录后即可直接运行Docker。

Linux下通过以下命令启动Docker服务：

::

    sudo service docker start

2. 拉取TensorFlow映像。Docker将应用程序及其依赖打包在映像文件中，通过映像文件生成容器。使用 ``docker image pull`` 命令拉取适合自己需求的TensorFlow映像，例如：

::

    docker image pull tensorflow/tensorflow:latest-py3        # 最新稳定版本TensorFlow（Python 3.5，CPU版）
    docker image pull tensorflow/tensorflow:latest-gpu-py3    # 最新稳定版本TensorFlow（Python 3.5，GPU版）

更多映像版本可参考 `TensorFlow官方文档 <https://www.tensorflow.org/install/docker#download_a_tensorflow_docker_image>`_ 。

.. tip:: 在国内，推荐使用 `DaoCloud的Docker映像镜像 <https://www.daocloud.io/mirror>`_ ，将显著提高下载速度。


3. 基于拉取的映像文件，创建并启动TensorFlow容器。使用  ``docker container run`` 命令创建一个新的TensorFlow容器并启动。

**CPU版本的TensorFlow：**

::

    docker container run -it tensorflow/tensorflow:latest-py3 bash

.. hint::  ``docker container run`` 命令的部分选项如下：

    * ``-it`` 让docker运行的容器能够在终端进行交互，具体而言：

        * ``-i`` （ ``--interactive`` ）：允许与容器内的标准输入 (STDIN) 进行交互。
        * ``-t`` （ ``--tty`` ）：在新容器中指定一个伪终端。

    * ``--rm`` ：当容器中的进程运行完毕后自动删除容器。
    * ``tensorflow/tensorflow:latest-py3`` ：新容器基于的映像。如果本地不存在指定的映像，会自动从公有仓库下载。
    * ``bash`` 在容器中运行的命令（进程）。Bash是大多数Linux系统的默认Shell。

**GPU版本的TensorFlow：**

若需在TensorFlow Docker容器中开启GPU支持，需要具有一块NVIDIA显卡并已正确安装驱动程序（详见 :ref:`“TensorFlow安装”一章 <gpu_tensorflow>` ）。同时需要安装 `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ 。依照官方文档中的quickstart部分逐行输入命令即可。

.. warning:: 当前nvidia-docker仅支持Linux。

安装完毕后，在 ``docker container run`` 命令中添加 ``--runtime=nvidia`` 选项，并基于具有GPU支持的TensorFlow Docker映像启动容器即可，即：

::

    docker container run -it --runtime=nvidia tensorflow/tensorflow:latest-gpu-py3 bash

.. admonition:: Docker常用命令

    映像（image）相关操作：

    ::

        docker image pull [image_name]  # 从仓库中拉取映像[image_name]到本机 
        docker image ls                 # 列出所有本地映像
        docker image rm [image_name]    # 删除名为[image_name]的本地映像

    容器（container）相关操作：

    ::
        
        docker container run [image_name] [command] # 基于[image_name]映像建立并启动容器，并运行[command]
        docker container ls                         # 列出本机正在运行的容器
                                                    # （加入--all参数列出所有容器，包括已停止运行的容器）
        docker container rm [container_id]          # 删除ID为[container_id]的容器

    Docker入门教程可参考 `阮一峰的Docker入门教程 <http://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html>`_ 和 `Docker Cheat Sheet <https://www.docker.com/sites/default/files/Docker_CheatSheet_08.09.2016_0.pdf>`_ 。

.. _GCP:

在Google Cloud Platform（GCP）中使用TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://medium.com/@kstseng/%E5%9C%A8-google-cloud-platform-%E4%B8%8A%E4%BD%BF%E7%94%A8-gpu-%E5%92%8C%E5%AE%89%E8%A3%9D%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E7%9B%B8%E9%97%9C%E5%A5%97%E4%BB%B6-1b118e291015
    
Google Cloud Platform（GCP）是Google的云计算服务。其Compute Engine类似于AWS、阿里云等，允许用户快速建立自己的虚拟机实例。且GCP收费灵活，默认按时长计费。也就是说，你可以迅速建立一个带GPU的实例，训练一个模型，然后立即关闭（关机或删除实例）。GCP只收取在实例开启时所产生的费用，关机时只收取磁盘存储的费用，删除后即不再继续收费。

在 `Google Cloud Platform（GCP） <https://cloud.google.com/>`_ 中，可以很方便地建立具有GPU的虚拟机实例，只需要在创建实例（Compute Engine - VM实例 - 创建实例）的时候选择GPU类型和数量即可。不过需要注意两点：

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

在Colab中使用TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^