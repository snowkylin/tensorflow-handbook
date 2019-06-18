TensorFlow安装
===================

TensorFlow的最新安装步骤可参考官方网站上的说明（https://tensorflow.google.cn/install）。TensorFlow支持Python、Java、Go、C等多种编程语言以及Windows、OSX、Linux等多种操作系统，此处及后文均以主流的Python 3.6为准。关于在容器环境中部署TensorFlow的方法，见附录中 :ref:`install_by_docker` 。

..
    .. tip:: 如果只是安装一个运行在自己电脑上的，**无需GPU加速计算** 的简易环境，不希望在环境配置上花费太多精力，可以按以下步骤简易安装（以Windows系统为例）：

        - 下载并安装Python集成环境 `Anaconda <https://www.anaconda.com/download/>`_ （Python 3.6版本）；
        - 下载并安装Python的IDE `PyCharm <http://www.jetbrains.com/pycharm/>`_ （Community版本，或学生可申请Professional版本的 `免费授权 <https://sales.jetbrains.com/hc/zh-cn/articles/207154369>`_）；
        - 打开开始菜单中的“Anaconda Prompt”，输入 ``pip install tensorflow``；
        - 启动PyCharm，新建工程（使用默认python interpreter），在工程内新建一个Python文件。

        完毕。

一般安装步骤
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. 安装Python环境。此处建议安装 `Anaconda <https://www.anaconda.com/>`_ 的Python 3.6版本（后文均以此为准），这是一个开源的Python发行版本，提供了一个完整的科学计算环境，包括NumPy、SciPy等常用科学计算库。当然，你有权选择自己喜欢的Python环境。注意截至本手册撰写时，TensorFlow尚未支持Python 3.7版本；

2. 使用Anaconda自带的conda包管理器建立一个Conda虚拟环境，并进入该虚拟环境。在命令行下输入：

::

    conda create --name tensorflow python=3.6   # “tensorflow”是你建立的Conda虚拟环境的名字
    conda activate tensorflow                   # 进入名为“tensorflow”的虚拟环境

3. 使用Python包管理器pip安装TensorFlow。在命令行下输入：

::

    pip install tensorflow        # TensorFlow CPU版本

或

::

    pip install tensorflow-gpu    # TensorFlow GPU版本，需要具有NVIDIA显卡及正确安装驱动程序，详见下文

等待片刻即安装完毕。

.. tip:: 

    1. 也可以使用 ``conda install tensorflow`` 或者 ``conda install tensorflow-gpu`` 来安装TensorFlow，不过conda源的版本往往更新较慢，难以第一时间获得最新的TensorFlow版本；
    2. 在Windows下，需要打开开始菜单中的“Anaconda Prompt”进入Anaconda的命令行环境；
    3. 在国内环境下，推荐使用 `清华大学的pypi镜像 <https://mirrors.tuna.tsinghua.edu.cn/help/pypi/>`_ ，将显著提升pip的下载速度。

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

在安装GPU版本的TensorFlow前，你需要具有一块不太旧的NVIDIA显卡，以及正确安装NVIDIA显卡驱动程序、CUDA Toolkit和cnDNN。

GPU硬件的准备
-------------------------------------------

TensorFlow对NVIDIA显卡的支持较为完备。对于NVIDIA显卡，要求其CUDA Compute Capability须不低于3.0，可以到 `NVIDIA的官方网站 <https://developer.nvidia.com/cuda-gpus/>`_ 查询自己所用显卡的CUDA Compute Capability。目前，AMD的显卡也开始对TensorFlow提供支持，可访问  `这篇博客文章 <https://medium.com/tensorflow/amd-rocm-gpu-support-for-tensorflow-33c78cc6a6cf>`_  查看详情。

NVIDIA驱动程序的安装
-------------------------------------------

Windows环境中，如果系统具有NVIDIA显卡，则往往已经自动安装了NVIDIA显卡驱动程序。如未安装，直接访问NVIDIA官方网站下载并安装对应型号的最新公版驱动程序即可。

不过，在Linux系统上，NVIDIA显卡驱动程序的安装往往不会一帆风顺。对于Ubuntu系统，有一个很简易的NVIDIA驱动安装方法：在系统设置（System Setting）里面选软件与更新（Software & Updates），然后点选Additional Drivers里面的“Using NVIDIA binary driver”选项并点选右下角的“Apply Changes”即可，系统即会自动安装NVIDIA驱动，但是通过这种安装方式安装的NVIDIA驱动往往版本较旧。如果需要在Linux下手动安装NVIDIA驱动，注意在安装前：

- 禁用系统自带的开源显卡驱动Nouveau（在 ``/etc/modprobe.d/blacklist.conf`` 文件中添加一行 ``blacklist nouveau`` ，使用 ``sudo update-initramfs -u`` 更新内核，并重启）
- 禁用主板的Secure Boot功能
- 停用桌面环境（如 ``sudo service lightdm stop``）
- 删除原有NVIDIA驱动程序（如 ``sudo apt-get purge nvidia*``）

然后即可在 `NVIDIA官方网站 <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_ 下载驱动程序（为 ``.run`` 文件），并使用 ``sudo bash DRIVER_FILE_NAME.run`` 命令安装驱动。

安装完后可在命令行下使用 ``nvidia-smi`` 命令检查是否安装成功，若成功则会打印出当前系统安装的NVIDIA驱动信息，形式如下：

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

更详细的指导可以参考 `这篇文章 <https://www.linkedin.com/pulse/installing-nvidia-cuda-80-ubuntu-1604-linux-gpu-new-victor/>`_ 和 `这篇中文博客 <https://blog.csdn.net/wf19930209/article/details/81877822>`_ 。

CUDA Toolkit和cnDNN的安装
-------------------------------------------

在Anaconda环境下，推荐使用 

::

    conda install cudatoolkit=X.X
    conda install cudnn=X.X.X

安装CUDA Toolkit和cnDNN，其中X.X和X.X.X分别为需要安装的CUDA Toolkit和cuDNN版本号，必须严格按照TensorFlow官方网站所说明的版本安装。在安装前，可使用 ``conda search cudatoolkit`` 和 ``conda search cudnn`` 搜索conda源中可用的版本号。

当然，也可以按照 `TensorFlow官方网站上的说明 <https://www.tensorflow.org/install/gpu>`_ 手动下载CUDA Toolkit和cuDNN并安装，不过过程会稍繁琐。

使用conda包管理器安装GPU版本的TensorFlow时，会自动安装对应版本的CUDA Toolkit和cuDNN。conda源的更新较慢，如果对版本不太介意，推荐直接使用 ``conda install tensorflow-gpu`` 进行安装。

第一个程序
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

安装完毕后，我们来编写一个简单的程序来验证安装。

在命令行下输入 ``conda activate tensorflow`` 进入之前建立的安装有TensorFlow的Conda虚拟环境，再输入 ``python`` 进入Python环境，逐行输入以下代码：

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

此处使用的是Python语言，关于Python语言的入门教程可以参考 `runoob网站的Python 3教程 <http://www.runoob.com/python3/python3-tutorial.html>`_ 或 `廖雪峰的Python教程 <https://www.liaoxuefeng.com>`_ ，本手册之后将默认读者拥有Python语言的基本知识。不用紧张，Python语言易于上手，而TensorFlow本身也不会用到Python语言的太多高级特性。

IDE设置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本手册建议使用 `PyCharm <http://www.jetbrains.com/pycharm/>`_ 作为Python开发的IDE。

在新建项目时，你需要选定项目的Python Interpreter，也就是用怎样的Python环境来运行你的项目。在安装部分，你所建立的每个Conda虚拟环境其实都有一个自己独立的Python Interpreter，你只需要将它们添加进来即可。选择“Add”，并在接下来的窗口选择“Existing Environment”，在Interpreter处选择 ``Anaconda安装目录/envs/所需要添加的Conda环境名字/python.exe`` （Linux下无 ``.exe`` 后缀）并点击“OK”即可。如果选中了“Make available to all projects”，则在所有项目中都可以选择该Python Interpreter。注意，在Windows下Anaconda的默认安装目录比较特殊，一般为  ``C:\Users\用户名\Anaconda3\`` ，即当前Windows用户的用户目录下。

对于TensorFlow开发而言，PyCharm的Professonal版本非常有用的一个特性是 **远程调试** （Remote Debugging）。当你编写程序的终端机性能有限，但又有一台可远程ssh访问的高性能计算机（一般具有高性能GPU）时，远程调试功能可以让你在终端机编写程序的同时，在远程计算机上调试和运行程序（尤其是训练模型）。你在终端机上对代码和数据的修改可以自动同步到远程机，在实际使用的过程中如同在远程机上编写程序一般，与串流游戏有异曲同工之妙。不过远程调试对网络的稳定性要求高，如果需要长时间训练模型，建议登录远程机终端直接训练模型（Linux下可以结合 ``nohup`` 命令 [#nohup]_ ，让进程在后端运行，不受终端退出的影响）。远程调试功能的具体配置步骤见 `PyCharm文档 <https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html>`_ 。

.. tip:: 如果你是学生并有.edu结尾的邮箱的话，可以在 `这里 <http://www.jetbrains.com/student/>`_ 申请PyCharm的免费Professional版本授权。

.. [#nohup] 关于  ``nohup`` 命令可参考 https://www.ibm.com/developerworks/cn/linux/l-cn-nohup/

TensorFlow所需的硬件配置 *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


