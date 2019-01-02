TensorFlow安装
===================

TensorFlow的最新安装步骤可参考官方网站上的说明（https://tensorflow.google.cn/install）。TensorFlow支持Python、Java、Go、C等多种编程语言以及Windows、OSX、Linux等多种操作系统，此处及后文均以主流的Python 3.6为准。

..
    .. tip:: 如果只是安装一个运行在自己电脑上的，**无需GPU加速计算** 的简易环境，不希望在环境配置上花费太多精力，可以按以下步骤简易安装（以Windows系统为例）：

        - 下载并安装Python集成环境 `Anaconda <https://www.anaconda.com/download/>`_ （Python 3.6版本）；
        - 下载并安装Python的IDE `PyCharm <http://www.jetbrains.com/pycharm/>`_ （Community版本，或学生可申请Professional版本的 `免费授权 <https://sales.jetbrains.com/hc/zh-cn/articles/207154369>`_）；
        - 打开开始菜单中的“Anaconda Prompt”，输入 ``pip install tensorflow``；
        - 启动PyCharm，新建工程（使用默认python interpreter），在工程内新建一个Python文件。

        完毕。

安装步骤
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. 安装Python环境。此处建议安装 `Anaconda <https://www.anaconda.com/>`_ 的Python 3.6版本（后文均以此为准），这是一个开源的Python发行版本，提供了一个完整的科学计算环境，包括NumPy、SciPy等常用科学计算库。当然，你有权选择自己喜欢的Python环境。注意截至本手册撰写时，TensorFlow尚未支持Python 3.7版本；

2. 使用Anaconda自带的conda包管理器建立一个Python虚拟环境，并进入该虚拟环境。在命令行下输入：

::

    conda create --name tensorflow  # “tensorflow”是你建立的Python虚拟环境的名字
    conda activate tensorflow       # 进入名为“tensorflow”的虚拟环境

3. 使用conda包管理器安装TensorFlow。在命令行下输入：

::

    conda install tensorflow        # TensorFlow CPU版本

或

::

    conda install tensorflow-gpu    # TensorFlow GPU版本

.. tip:: 

    1. 使用conda包管理器安装GPU版本的TensorFlow时，会自动安装对应版本的CUDA Toolkit和cuDNN；
    2. 在Windows下，需要打开开始菜单中的“Anaconda Prompt”进入Anaconda的命令行环境；
    3. 在国内，推荐使用 `清华大学开源软件镜像站的Anaconda镜像 <https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/>`_ ，将显著提高下载速度。

.. admonition:: conda包管理器

    conda包管理器是Anaconda自带的Python包管理器，可以帮助我们轻松地安装各种Python包（包括TensorFlow）。常用命令如下：

    ::

        conda install [package-name]    # 安装名为[package-name]的Python包
        conda update [package-name]     # 更新名为[package-name]的Python包
        conda remove [package-name]     # 删除名为[package-name]的Python包
        conda list                      # 列出已安装的所有Python包

.. admonition:: Python虚拟环境

    在Python开发中，很多时候我们希望每个应用有一个独立的Python环境（比如应用1需要用到TensorFlow 1.X，而应用2使用TensorFlow 2.0）。这时，Python虚拟环境即可为一个应用创建一套“隔离”的Python运行环境。使用Python的包管理器conda即可轻松地创建Python虚拟环境。常用命令如下：

    ::

        conda create --name [env-name]      # 建立名为[env-name]的Python虚拟环境
        conda activate [env-name]           # 进入名为[env-name]的Python虚拟环境
        conda deactivate                    # 退出当前的Python虚拟环境
        conda env remove --name [env-name]  # 删除名为[env-name]的Python虚拟环境
        conda env list                      # 列出所有Python虚拟环境

.. admonition:: GPU版本的TensorFlow

    GPU版本的TensorFlow可以利用NVIDIA GPU强大的计算加速能力，使TensorFlow的运行更为高效，尤其是可以成倍提升模型训练的速度。

    在安装GPU版本的TensorFlow前，你需要具有一块不太旧的NVIDIA显卡 [#gpu_version]_ ，以及正确安装NVIDIA显卡驱动程序。

    NVIDIA显卡驱动程序在Linux系统上的安装往往不会一帆风顺。对于Ubuntu系统，有一个很简易的NVIDIA驱动安装方法：在系统设置（System Setting）里面选软件与更新（Software & Updates），然后点选Additional Drivers里面的“Using NVIDIA binary driver”选项并点选右下角的“Apply Changes”即可，系统即会自动安装NVIDIA驱动。如果需要在Linux下手动安装NVIDIA驱动，注意在安装前禁用系统自带的开源显卡驱动Nouveau、禁用主板的Secure Boot功能。更详细的指导可以参考 `这篇文章 <https://www.linkedin.com/pulse/installing-nvidia-cuda-80-ubuntu-1604-linux-gpu-new-victor/>`_ 。

    .. [#gpu_version] 具体而言，该显卡的CUDA Compute Capability须不低于3.0，可以到 `NVIDIA的官方网站 <https://developer.nvidia.com/cuda-gpus/>`_ 查询自己所用显卡的CUDA Compute Capability。

第一个程序
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

安装完毕后，我们来编写一个简单的程序来验证安装。

在命令行下输入 ``conda activate tensorflow`` 进入之前建立的安装有TensorFlow的Python虚拟环境，再输入 ``python`` 进入Python环境，逐行输入以下代码：

.. code-block:: python

    import tensorflow as tf
    tf.enable_eager_execution()

    A = tf.constant([[1, 2], [3, 4]])
    B = tf.constant([[5, 6], [7, 8]])
    C = tf.matmul(A, B)

    print(C)

如果能够最终输出::

    tf.Tensor(
    [[19 22]
    [43 50]], shape=(2, 2), dtype=int32)

说明TensorFlow已安装成功。运行途中可能会输出一些TensorFlow的提示信息，属于正常现象。

此处使用的是Python语言，关于Python语言的入门教程可以参考 `runoob网站的Python 3教程 <http://www.runoob.com/python3/python3-tutorial.html>`_ 或 `廖雪峰的Python教程 <https://www.liaoxuefeng.com>`_ ，本手册之后将默认读者拥有Python语言的基本知识。不用紧张，Python语言易于上手，而TensorFlow本身也不会用到Python语言的太多高级特性。关于Python的IDE，建议使用 `PyCharm <http://www.jetbrains.com/pycharm/>`_ 。

.. tip:: 如果你是学生并有.edu结尾的邮箱的话，可以在 `这里 <http://www.jetbrains.com/student/>`_ 申请PyCharm的免费Professional版本授权。
