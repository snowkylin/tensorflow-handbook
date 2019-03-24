TensorFlow环境配置与管理
============================================

GPU的使用与分配
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

很多时候的场景是：实验室/公司研究组里有许多学生/研究员都需要使用GPU，但多卡的机器只有一台，这时就需要注意如何分配显卡资源。

命令 ``nvidia-smi`` 可以查看机器上现有的GPU及使用情况（在Windows下，将 ``C:\Program Files\NVIDIA Corporation\NVSMI`` 加入Path环境变量中即可，或Windows 10下可使用任务管理器的“性能”标签查看显卡信息）。

使用环境变量 ``CUDA_VISIBLE_DEVICES`` 可以控制程序所使用的GPU。假设发现四卡的机器上显卡0,1使用中，显卡2,3空闲，Linux终端输入::

    export CUDA_VISIBLE_DEVICES=2,3

或在代码中加入

.. code-block:: python

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

即可指定程序只在显卡2,3上运行。

默认情况下，TensorFlow将使用几乎所有可用的显存，以避免内存碎片化所带来的性能损失。可以通过 ``tf.ConfigProto`` 类来设置TensorFlow使用显存的策略。具体方式是实例化一个 ``tf.ConfigProto`` 类，设置参数，并在运行 ``tf.enable_eager_execution()`` 时指定Config参数。以下代码通过 ``allow_growth`` 选项设置TensorFlow仅在需要时申请显存空间：

.. code-block:: python

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)

以下代码通过 ``per_process_gpu_memory_fraction`` 选项设置TensorFlow固定消耗40%的GPU显存：

.. code-block:: python

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    tf.enable_eager_execution(config=config)

Graph Execution下，也可以在实例化新的session时传入 tf.ConfigPhoto 类来进行设置。

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

在Colab中使用TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^