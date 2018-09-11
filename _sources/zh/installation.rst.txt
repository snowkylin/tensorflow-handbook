TensorFlow安装
================

TensorFlow的最新安装步骤可参考官方网站上的说明（https://tensorflow.google.cn/install）。TensorFlow支持Python、Java、Go、C等多种编程语言以及Windows、OSX、Linux等多种操作系统，此处及后文均以主流的Python语言为准。

以下提供简易安装和正式安装两种途径，供不同层级的读者选用。

简易安装
^^^^^^^^^^^^
如果只是安装一个运行在自己电脑上的，无需GPU的简易环境，不希望在环境配置上花费太多精力，建议按以下步骤安装（以Windows系统为例）：

- 下载并安装Python集成环境 `Anaconda <https://www.anaconda.com/download/>`_ （Python 3.6版本）；
- 下载并安装Python的IDE `PyCharm <http://www.jetbrains.com/pycharm/>`_ （Community版本，或学生可申请Professional版本的 `免费授权 <https://sales.jetbrains.com/hc/zh-cn/articles/207154369>`_）；
- 打开开始菜单中的“Anaconda Prompt”，输入 ``pip install tensorflow``；
- 启动PyCharm，新建工程（使用默认python interpreter），在工程内新建一个Python文件。

完毕。

正式安装
^^^^^^^^^^^^
该部分包含了更多安装上的细节（如建立conda环境），以及GPU版本TensorFlow的环境配置方法。

安装前的环境配置
-------------------------------
正式安装TensorFlow前，需要为其配置合适的环境。步骤如下：

1. 检查自己的电脑是否具有NVIDIA显卡。如有，建议安装GPU版本的TensorFlow，以利用GPU强大的计算加速能力 [#f1]_ ，否则可以安装CPU版本。具体而言，该显卡的CUDA Compute Capability须不低于3.0，可以到 `NVIDIA的官方网站 <https://developer.nvidia.com/cuda-gpus/>`_ 查询自己所用显卡的CUDA Compute Capability；
2. 安装Python环境。此处建议安装Anaconda，这是一个开源的Python发行版本，提供了一个完整的科学计算环境，包括NumPy、SciPy等常用科学计算库。当然，你有权选择自己喜欢的Python环境。注意截至本手册撰写时，TensorFlow在Windows下的安装仅支持Python 3.X版本；

   * 安装Anaconda时，可以选择将Anaconda目录添加到系统的PATH中（虽然安装程序不推荐这样做），这样可以直接在命令行环境下使用Anaconda的各项功能。当然，不添加的话也可以使用开始菜单中的Anaconda Prompt进入命令行的Anaconda环境。

3. （针对GPU版本）安装NVIDIA显卡驱动程序、 `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_ 和 `cuDNN <https://developer.nvidia.com/cudnn>`_ 。值得注意的事项有：

   * 建议的顺序是：先安装最新版NVIDIA显卡驱动程序，再安装CUDA（安装时不要选择同时安装驱动），最后安装cuDNN。CUDA附带的显卡驱动程序可能过旧；
   * 在Ubuntu下有一个很简易的驱动安装方法：在系统设置（System Setting）里面选软件与更新（Software & Updates），然后点选Additional Drivers里面的“Using NVIDIA binary driver”选项并点选右下角的“Apply Changes”即可，系统即会自动安装NVIDIA驱动。否则，NVIDIA显卡驱动程序在Linux系统上的安装往往不会一帆风顺，注意在安装前禁用系统自带的开源显卡驱动Nouveau、禁用主板的Secure Boot功能。更详细的指导可以参考 `这篇文章 <https://www.linkedin.com/pulse/installing-nvidia-cuda-80-ubuntu-1604-linux-gpu-new-victor/>`_ ；
   * CUDA Toolkit和cuDNN的版本一定要与TensorFlow官方网站安装说明的版本一致，注意官方网站安装说明里要求安装的版本可能并非最新版本；
   * cuDNN的安装方式比较特殊，你需要手动将下载的安装包复制到CUDA的安装目录下。

安装
----------------

在Anaconda环境下的安装过程如下（以Windows系统为例）：

1. 新建一个叫做 ``tensorflow`` 的conda环境

::

    conda create -n tensorflow python=X.X # 注意这里的X.X填写自己Python环境的版本，例如3.6

2. 激活环境

::

    activate tensorflow

3. 使用pip安装TensorFlow

安装CPU版本
::

    pip install tensorflow

安装GPU版本
::

    pip install tensorflow-gpu

如有需要，也可以安装TensorFlow的Nightly版本，该版本较之于正式版本会具有一些最新的特性（例如在TensorFlow 1.8版本以前，本手册主要使用的Eager Execution模式只在Nightly版本中提供），然而稳定度可能稍弱。在一个新的虚拟环境里运行 ``pip install tf-nightly`` （CPU版本）或 ``pip install tf-nightly-gpu`` （GPU版本）即可。注意，若安装GPU版本，其往往要求安装比正式版要求中更新的CUDA和cuDNN。好在CUDA和cuDNN的不同版本是可以共存的。

如果使用pip命令安装速度较慢，可以尝试 `清华大学开源软件镜像站的TensorFlow镜像 <https://mirrors.tuna.tsinghua.edu.cn/help/tensorflow/>`_。

第一个程序
^^^^^^^^^^^^^^^

安装完毕后，我们来编写一个简单的程序来验证安装。

在命令行下输入 ``activate tensorflow`` 进入之前建立的安装有TensorFlow的conda环境，再输入 ``python`` 进入Python环境，逐行输入以下代码：

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

此处使用的是Python语言，关于Python语言的入门教程可以参考 http://www.runoob.com/python3/python3-tutorial.html 或 https://www.liaoxuefeng.com ，本手册之后将默认读者拥有Python语言的基本知识。不用紧张，Python语言易于上手，而TensorFlow本身也不会用到Python语言的太多高级特性。关于Python的IDE，建议使用 `PyCharm <http://www.jetbrains.com/pycharm/>`_ 。如果你是学生并有.edu结尾的邮箱的话，可以在 `这里 <http://www.jetbrains.com/student/>`_ 申请免费的授权。如果没有，也可以下载社区版本的PyCharm，主要功能差别不大。

.. [#f1] GPU加速的效果与模型类型和GPU的性能有关，如果CPU性能较高，但GPU仅有入门级的性能，其实速度提升不大，大概1-2倍。不过如果GPU性能强大的话（例如，本手册写作时，NVIDIA GeForce GTX 1080 Ti或NVIDIA GeForce TITAN系列是市场上性能较强大的显卡型号），对于特定模型，十几倍甚至更高的加速效果也是可以达到的。同时，GPU的加速效果与任务本身也有关。入门级的TensorFlow模型往往不需要太高的计算性能，CPU版本的TensorFlow足以胜任，因此可以待到掌握TensorFlow的基本知识后，再决定是否购入更高级的GPU以得到更快的训练速度。

升级到新版本
^^^^^^^^^^^^^^^^^^^^^

TensorFlow的版本频繁更新，如果希望升级当前的TensorFlow版本，请进入安装有TensorFlow的conda环境下输入

::

    pip install tensorflow --upgrade

如果你想安装特定版本的TensorFlow，请输入

::

    pip install tensorflow==1.8.0   # 1.8.0为指定版本

升级有风险，可能出现升级后TensorFlow导入出错的情况，比较简单的方式是删除当前conda环境后重新安装一遍。以下conda命令可能会有用

::

    conda list                                              # 列出当前conda环境下所有package及版本
    conda env list                                          # 列出所有conda环境
    conda create --name new_env_name --clone old_env_name   # 备份当前conda环境`old_env_name`到`new_env_name`
    conda env remove -n tensorflow                          # 删除名为`tensorflow`的conda环境
