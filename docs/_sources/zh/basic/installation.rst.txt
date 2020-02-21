TensorFlow安装与环境配置
======================================

TensorFlow的最新安装步骤可参考官方网站上的说明（https://tensorflow.google.cn/install）。TensorFlow支持Python、Java、Go、C等多种编程语言以及Windows、OSX、Linux等多种操作系统，此处及后文均以Python 3.7为准。

.. hint:: 本章介绍在一般的个人电脑或服务器上直接安装TensorFlow 2的方法。关于在容器环境（Docker）、云平台中部署TensorFlow或在线上环境中使用TensorFlow的方法，见附录 :doc:`使用Docker部署TensorFlow环境 <../appendix/docker>` 和 :doc:`在云端使用TensorFlow <../appendix/cloud>` 。软件的安装方法往往具有时效性，本节的更新日期为2019年10月。

一般安装步骤
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. 安装Python环境。此处建议安装 `Anaconda <https://www.anaconda.com/>`_ 的Python 3.7版本（后文均以此为准），这是一个开源的Python发行版本，提供了一个完整的科学计算环境，包括NumPy、SciPy等常用科学计算库。当然，你有权选择自己喜欢的Python环境。Anaconda的安装包可在 `这里 <https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/>`_ 获得。

2. 使用Anaconda自带的conda包管理器建立一个Conda虚拟环境，并进入该虚拟环境。在命令行下输入：

::

    conda create --name tf2 python=3.7      # “tf2”是你建立的conda虚拟环境的名字
    conda activate tf2                      # 进入名为“tf2”的conda虚拟环境

3. 使用Python包管理器pip安装TensorFlow。在命令行下输入：

::

    pip install tensorflow

等待片刻即安装完毕。

.. tip:: 

    1. 也可以使用 ``conda install tensorflow`` 来安装TensorFlow，不过conda源的版本往往更新较慢，难以第一时间获得最新的TensorFlow版本；
    2. 从 TensorFlow 2.1 开始，pip 包 ``tensorflow`` 即同时包含 GPU 支持，无需通过特定的 pip 包 ``tensorflow-gpu`` 安装GPU版本。如果对pip包的大小敏感，可使用 ``tensorflow-cpu`` 包安装仅支持CPU的TensorFlow版本。
    3. 在Windows下，需要打开开始菜单中的“Anaconda Prompt”进入Anaconda的命令行环境；
    4. 在国内环境下，推荐使用国内的pypi镜像和Anaconda镜像，将显著提升pip和conda的下载速度；
        
        - 清华大学的pypi镜像：https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
        - 清华大学的Anaconda镜像：https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/
    5. 如果对磁盘空间要求严格（比如服务器环境），可以安装 `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ ，仅包含Python和Conda，其他的包可自己按需安装。Miniconda的安装包可在 `这里 <https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/>`_ 获得。

.. admonition:: pip和conda包管理器

    pip是最为广泛使用的Python包管理器，可以帮助我们获得最新的Python包并进行管理。常用命令如下：

    ::

        pip install [package-name]              # 安装名为[package-name]的包
        pip install [package-name]==X.X         # 安装名为[package-name]的包并指定版本X.X
        pip install [package-name] --proxy=代理服务器IP:端口号         # 使用代理服务器安装
        pip install [package-name] --upgrade    # 更新名为[package-name]的包
        pip uninstall [package-name]            # 删除名为[package-name]的包
        pip list                                # 列出当前环境下已安装的所有包
    
    conda包管理器是Anaconda自带的包管理器，可以帮助我们在conda环境下轻松地安装各种包。相较于pip而言，conda的通用性更强（不仅是Python包，其他包如CUDA Toolkit和cuDNN也可以安装），但conda源的版本更新往往较慢。常用命令如下：

    ::

        conda install [package-name]        # 安装名为[package-name]的包
        conda install [package-name]=X.X    # 安装名为[package-name]的包并指定版本X.X
        conda update [package-name]         # 更新名为[package-name]的包
        conda remove [package-name]         # 删除名为[package-name]的包
        conda list                          # 列出当前环境下已安装的所有包
        conda search [package-name]         # 列出名为[package-name]的包在conda源中的所有可用版本

    conda中配置代理：在用户目录下的 .condarc 文件中添加以下内容：

    ::

        proxy_servers:
            http: http://代理服务器IP:端口号

.. admonition:: Conda虚拟环境

    在Python开发中，很多时候我们希望每个应用有一个独立的Python环境（比如应用1需要用到TensorFlow 1.X，而应用2使用TensorFlow 2.0）。这时，Conda虚拟环境即可为一个应用创建一套“隔离”的Python运行环境。使用Python的包管理器conda即可轻松地创建Conda虚拟环境。常用命令如下：

    ::

        conda create --name [env-name]      # 建立名为[env-name]的Conda虚拟环境
        conda activate [env-name]           # 进入名为[env-name]的Conda虚拟环境
        conda deactivate                    # 退出当前的Conda虚拟环境
        conda env remove --name [env-name]  # 删除名为[env-name]的Conda虚拟环境
        conda env list                      # 列出所有Conda虚拟环境

.. _gpu_tensorflow:

GPU版本TensorFlow安装指南
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GPU版本的TensorFlow可以利用NVIDIA GPU强大的计算加速能力，使TensorFlow的运行更为高效，尤其是可以成倍提升模型训练的速度。

在安装GPU版本的TensorFlow前，你需要具有一块不太旧的NVIDIA显卡，以及正确安装NVIDIA显卡驱动程序、CUDA Toolkit和cuDNN。

GPU硬件的准备
-------------------------------------------

TensorFlow对NVIDIA显卡的支持较为完备。对于NVIDIA显卡，要求其CUDA Compute Capability须不低于3.5，可以到 `NVIDIA的官方网站 <https://developer.nvidia.com/cuda-gpus/>`_ 查询自己所用显卡的CUDA Compute Capability。目前，AMD的显卡也开始对TensorFlow提供支持，可访问  `这篇博客文章 <https://medium.com/tensorflow/amd-rocm-gpu-support-for-tensorflow-33c78cc6a6cf>`_  查看详情。

NVIDIA驱动程序的安装
-------------------------------------------

**Windows** 

Windows环境中，如果系统具有NVIDIA显卡，则往往已经自动安装了NVIDIA显卡驱动程序。如未安装，直接访问 `NVIDIA官方网站 <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_ 下载并安装对应型号的最新公版驱动程序即可。

**Linux** 

在服务器版Linux系统下，同样访问 `NVIDIA官方网站 <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_ 下载驱动程序（为 ``.run`` 文件），并使用 ``sudo bash DRIVER_FILE_NAME.run`` 命令安装驱动即可。在安装之前，可能需要使用 ``sudo apt-get install build-essential`` 安装合适的编译环境。

在具有图形界面的桌面版Linux系统上，NVIDIA显卡驱动程序需要一些额外的配置，否则会出现无法登录等各种错误。如果需要在Linux下手动安装NVIDIA驱动，注意在安装前进行以下步骤（以Ubuntu为例）：

- 禁用系统自带的开源显卡驱动Nouveau（在 ``/etc/modprobe.d/blacklist.conf`` 文件中添加一行 ``blacklist nouveau`` ，使用 ``sudo update-initramfs -u`` 更新内核，并重启）
- 禁用主板的Secure Boot功能
- 停用桌面环境（如 ``sudo service lightdm stop``）
- 删除原有NVIDIA驱动程序（如 ``sudo apt-get purge nvidia*``）

.. tip:: 对于桌面版Ubuntu系统，有一个很简易的NVIDIA驱动安装方法：在系统设置（System Setting）里面选软件与更新（Software & Updates），然后点选Additional Drivers里面的“Using NVIDIA binary driver”选项并点选右下角的“Apply Changes”即可，系统即会自动安装NVIDIA驱动，但是通过这种安装方式安装的NVIDIA驱动往往版本较旧。

NVIDIA驱动程序安装完成后，可在命令行下使用 ``nvidia-smi`` 命令检查是否安装成功，若成功则会打印出当前系统安装的NVIDIA驱动信息，形式如下：

::
    
    $ nvidia-smi
    Mon Jun 10 23:19:54 2019
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 419.35       Driver Version: 419.35       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 106... WDDM  | 00000000:01:00.0  On |                  N/A |
    | 27%   51C    P8    13W / 180W |   1516MiB /  6144MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0       572    C+G   Insufficient Permissions                   N/A      |
    +-----------------------------------------------------------------------------+

.. hint:: 命令 ``nvidia-smi`` 可以查看机器上现有的GPU及使用情况。（在Windows下，将 ``C:\Program Files\NVIDIA Corporation\NVSMI`` 加入Path环境变量中即可，或Windows 10下可使用任务管理器的“性能”标签查看显卡信息）

更详细的GPU环境配置指导可以参考 `这篇文章 <https://www.linkedin.com/pulse/installing-nvidia-cuda-80-ubuntu-1604-linux-gpu-new-victor/>`_ 和 `这篇中文博客 <https://blog.csdn.net/wf19930209/article/details/81877822>`_ 。

CUDA Toolkit和cuDNN的安装
-------------------------------------------

在Anaconda环境下，推荐使用 

::

    conda install cudatoolkit=X.X
    conda install cudnn=X.X.X

安装CUDA Toolkit和cuDNN，其中X.X和X.X.X分别为需要安装的CUDA Toolkit和cuDNN版本号，必须严格按照 `TensorFlow官方网站所说明的版本 <https://www.tensorflow.org/install/gpu#software_requirements>`_ 安装。例如，对于TensorFlow 2.1，可使用::

    conda install cudatoolkit=10.1
    conda install cudnn=7.6.5

在安装前，可使用 ``conda search cudatoolkit`` 和 ``conda search cudnn`` 搜索conda源中可用的版本号。

当然，也可以按照 `TensorFlow官方网站上的说明 <https://www.tensorflow.org/install/gpu>`_ 手动下载CUDA Toolkit和cuDNN并安装，不过过程会稍繁琐。

使用conda包管理器安装GPU版本的TensorFlow时，会自动安装对应版本的CUDA Toolkit和cuDNN。conda源的更新往往较慢，如果对版本不太介意，也可以直接使用 ``conda install tensorflow-gpu`` 进行安装。

第一个程序
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

安装完毕后，我们来编写一个简单的程序来验证安装。

在命令行下输入 ``conda activate tf2`` 进入之前建立的安装有TensorFlow的Conda虚拟环境，再输入 ``python`` 进入Python环境，逐行输入以下代码：

.. code-block:: python

    import tensorflow as tf

    A = tf.constant([[1, 2], [3, 4]])
    B = tf.constant([[5, 6], [7, 8]])
    C = tf.matmul(A, B)

    print(C)

如果能够最终输出::

    tf.Tensor(
    [[19 22]
    [43 50]], shape=(2, 2), dtype=int32)

说明TensorFlow已安装成功。运行途中可能会输出一些TensorFlow的提示信息，属于正常现象。

.. warning:: 如果你在Windows下安装了TensorFlow 2.1正式版，可能会在导入TensorFlow时出现 `DLL载入错误 <https://github.com/tensorflow/tensorflow/issues/35749>`_ 。此时安装 `Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019 <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_ 即可正常使用。

此处使用的是Python语言，关于Python语言的入门教程可以参考 `runoob网站的Python 3教程 <http://www.runoob.com/python3/python3-tutorial.html>`_ 或 `廖雪峰的Python教程 <https://www.liaoxuefeng.com>`_ ，本手册之后将默认读者拥有Python语言的基本知识。不用紧张，Python语言易于上手，而TensorFlow本身也不会用到Python语言的太多高级特性。

IDE设置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于机器学习的研究者和从业者，建议使用 `PyCharm <http://www.jetbrains.com/pycharm/>`_ 作为Python开发的IDE。

在新建项目时，你需要选定项目的Python Interpreter，也就是用怎样的Python环境来运行你的项目。在安装部分，你所建立的每个Conda虚拟环境其实都有一个自己独立的Python Interpreter，你只需要将它们添加进来即可。选择“Add”，并在接下来的窗口选择“Existing Environment”，在Interpreter处选择 ``Anaconda安装目录/envs/所需要添加的Conda环境名字/python.exe`` （Linux下无 ``.exe`` 后缀）并点击“OK”即可。如果选中了“Make available to all projects”，则在所有项目中都可以选择该Python Interpreter。注意，在Windows下Anaconda的默认安装目录比较特殊，一般为  ``C:\Users\用户名\Anaconda3\`` 或 ``C:\Users\用户名\AppData\Local\Continuum\anaconda3`` 。此处 ``AppData`` 是隐藏文件夹。

对于TensorFlow开发而言，PyCharm的Professonal版本非常有用的一个特性是 **远程调试** （Remote Debugging）。当你编写程序的终端机性能有限，但又有一台可远程ssh访问的高性能计算机（一般具有高性能GPU）时，远程调试功能可以让你在终端机编写程序的同时，在远程计算机上调试和运行程序（尤其是训练模型）。你在终端机上对代码和数据的修改可以自动同步到远程机，在实际使用的过程中如同在远程机上编写程序一般，与串流游戏有异曲同工之妙。不过远程调试对网络的稳定性要求高，如果需要长时间训练模型，建议登录远程机终端直接训练模型（Linux下可以结合 ``nohup`` 命令 [#nohup]_ ，让进程在后端运行，不受终端退出的影响）。远程调试功能的具体配置步骤见 `PyCharm文档 <https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html>`_ 。

.. tip:: 如果你是学生并有.edu结尾的邮箱的话，可以在 `这里 <http://www.jetbrains.com/student/>`_ 申请PyCharm的免费Professional版本授权。

对于TensorFlow及深度学习的业余爱好者或者初学者， `Visual Studio Code <https://code.visualstudio.com/>`_ 或者一些在线的交互式Python环境（比如免费的 `Google Colab <https://colab.research.google.com/>`_ ）也是不错的选择。Colab的使用方式可参考 :ref:`附录 <colab>` 。

.. warning:: 如果你使用的是旧版本的 PyCharm ，可能会在安装 TensorFlow 2 后出现部分代码自动补全功能丧失的问题。升级到新版的 PyCharm （2019.3及以后版本）即可解决这一问题。


.. [#nohup] 关于  ``nohup`` 命令可参考 https://www.ibm.com/developerworks/cn/linux/l-cn-nohup/

TensorFlow所需的硬件配置 *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint:: 对于学习而言，TensorFlow的硬件门槛并不高。甚至，借助 :ref:`免费 <colab>` 或 :ref:`灵活 <gcp>` 的云端计算资源，只要你有一台能上网的电脑，就能够熟练掌握TensorFlow！

在很多人的刻板印象中，TensorFlow乃至深度学习是一件非常“吃硬件”的事情，以至于一接触TensorFlow，第一件事情可能就是想如何升级自己的电脑硬件。不过，TensorFlow所需的硬件配置很大程度是视任务和使用环境而定的：

- 对于TensorFlow初学者，无需硬件升级也可以很好地学习和掌握TensorFlow。本手册中的大部分教学示例，大部分当前主流的个人电脑（即使没有GPU）均可胜任，无需添置其他硬件设备。对于本手册中部分计算量较大的示例（例如 :ref:`在cats_vs_dogs数据集上训练CNN图像分类 <cats_vs_dogs>` ），一块主流的NVIDIA GPU会大幅加速训练。如果自己的个人电脑难以胜任，可以考虑在云端（例如 :ref:`免费的 Colab <colab>` ）进行模型训练。
- 对于参加数据科学竞赛（比如Kaggle）或者经常在本机进行训练的个人爱好者或开发者，一块高性能的NVIDIA GPU往往是必要的。CUDA核心数和显存大小是决定显卡机器学习性能的两个关键参数，前者决定训练速度，后者决定可以训练多大的模型以及训练时的最大Batch Size，对于较大规模的训练而言尤其敏感。
- 对于前沿的机器学习研究（尤其是计算机视觉和自然语言处理领域），多GPU并行训练是标准配置。为了快速迭代实验结果以及训练更大规模的模型以提升性能，4卡、8卡或更高的GPU数量是常态。

作为参考，笔者给出截至本手册撰写时，自己所在工作环境的一些硬件配置：

- 笔者写作本书的示例代码时，除了分布式和云端训练相关章节，其他部分均使用一台Intel i5处理器，16GB DDR3内存的普通台式机（未使用GPU）在本地开发测试，部分计算量较大的模型使用了一块淘宝上180元购买的 NVIDIA P106-90 （单卡3GB显存）矿卡进行训练；
- 在笔者的研究工作中，长年使用一块 NVIDIA GTX 1060 （单卡6GB显存）在本地环境进行模型的基础开发和调试；
- 笔者所在的实验室使用一台4块 NVIDIA GTX 1080 Ti （单卡11GB显存）并行的工作站和一台10块 NVIDIA GTX 1080 Ti （单卡11GB显存）并行的服务器进行前沿计算机视觉模型的训练；
- 笔者合作过的公司使用8块 NVIDIA Tesla V100 （单卡32GB显存）并行的服务器进行前沿自然语言处理（如大规模机器翻译）模型的训练。

尽管科研机构或公司使用的计算硬件配置堪称豪华，不过与其他前沿科研领域（例如生物）动辄几十上百万的仪器试剂费用相比，依然不算太贵（毕竟一台六七万至二三十万的深度学习服务器就可以供数位研究者使用很长时间）。因此，机器学习相对而言还是十分平易近人的。

关于深度学习工作站的具体配置，由于硬件行情更新较快，故不在此列出具体配置，推荐关注 `知乎问题 - 如何配置一台适用于深度学习的工作站？ <https://www.zhihu.com/question/33996159>`_ ，并结合最新市场情况进行配置。