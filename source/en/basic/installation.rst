Installation and Environment Configuration
==========================================

For the latest installation steps of TensorFlow please refer to the instructions on the official website (https://tensorflow.google.cn/install). TensorFlow supports multiple programming languages such as Python, Java, Go, and C; as well as multiple operating systems such as Windows, OSX, and Linux. Python 3.7 will be used as sstandard hereafter.

.. admonition:: Hint 

    This chapter describes how to install TensorFlow 2.0 directly on normal PCs or servers. Please refer to the appendix :doc:`Deploy TensorFlow from Docker <../appendix/docker>` and :doc:`Use TensorFlow on cloud <../appendix/cloud>` for deploying TensorFlow in Docker, on cloud or using TensorFlow on online platforms. Software installation methods are usually time-sensitive, and the update date of this section is October 2019.

General steps for installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Install Python environment. Installing  `Anaconda <https://www.anaconda.com/>`_ with Python V3.7, which is a open-source Python distribution providing a full environment for scientific computation, including common scientific computation libraries like NumPy and SciPy, is advised. You have rights to choose your own favourite Python environment, of course. The installation package of Anaconda can be obtained `here <https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/>`_.

2. Use Anaconda's own conda package manager to build a Conda virtual environment, then enter it and input the following commands at the command line:

::

    conda create --name tf2 python=3.7     # "tf2" is the name of the Conda virtual environment that you build
    conda activate tf2                     # enter the "tf2" virtual environment

3. Use pip, the Python package manager, to install TensorFlow. Input these at the command line:

::

    pip install tensorflow

Wait for a moment before finishing installation.

.. admonition:: Tip

    1. You can also use ``conda install tensorflow`` or ``conda install tensorflow-gpu`` to install TensorFlow. However, the conda source version is often updated in a less frequency, thus making it harder to acquire the latest version of TensorFlow in the first place;
    2. Starting with TensorFlow 2.1, the pip package ``tensorflow`` also includes GPU support, eliminating the need to install the GPU version through a specific pip package ``tensorflow-gpu``. If you are sensitive to the size of the pip package, you can use the ``tensorflow-cpu`` package to install the TensorFlow version that only supports CPU.
    3. If you use Windows, You need to enter Anaconda command line environment by clicking "Anaconda Prompt" in the start menu;
    4. If the limit on disk space is strict (like on a server), you can install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ instead, which only includes Python and Conda while the installation of other packages is at your discretion. You can obtain Miniconda installation package `here <https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/>`_.
    5. If you encountered "Could not find a version that satisfies the requirement tensorflow" error when installing TensorFlow with pip, it is more likely that you are using a 32-bit (x86) Python environment. Please change to 64-bit Python. You can check whether your Python version is 32-bit (e.g. ``[MSC v.XXXX 32 bit (Intel)]``) or 64-bit (e.g. ``[MSC v.XXXX 64 bit (AMD64)]``) by typing ``python`` in the command line to enter the Python interactive interface and checking the prompt information when entering the interface.

.. admonition:: pip and conda package manager

    Pip is the most widely used Python package manager that can help us obtain the latest Python packages and manage them. Common commands are listed below:

    ::

        pip install [package-name]              # install a package named [package-name]
        pip install [package-name]==X.X         # install a package named [package-name] with designated version of X.X
        pip install [package-name] --proxy=ip:port         # use proxy server to install package
        pip install [package-name] --upgrade    # update a package named [package-name]
        pip uninstall [package-name]            # uninstall a package named [package-name]
        pip list                                # list all installed packages under the current environment
    
    Conda is Anaconda's own package manager that can enable us to install various kinds of packages under conda environment with ease. Conda possesses better versality compared to pip (not only installing Python packages, but also others like CUDA Toolkit and cuDNN), but with less update frequency. Common commands are listed below:

    ::

        conda install [package-name]        # install a package named [package-name]
        conda install [package-name]=X.X    # install a package named [package-name] with designated version of X.X
        conda update [package-name]         # update a package named [package-name]
        conda remove [package-name]         # 删除名为[package-name]的包
        conda list                          # uninstall a package named [package-name]
        conda search [package-name]         # list all available versions of a package named [package-name] from conda source

    To configure proxy in conda, add the following text in the .condarc file under the user directory:

    ::

        proxy_servers:
            http: http://ip:port

.. admonition:: Conda virtual environment

    When developing with Python, we often wish to grant each application an independent Python environment (like when app 1 needs TensorFlow 1.X and app 2 needs TensorFlow 2.0). At this moment, the conda virtual environment can provide a set of "isolated" Python environment for an application. It is easy to create a conda virtual environment by using the Python package manager. Common commands are listed below:

    ::

        conda create --name [env-name]      # create a conda virtual environment named [env-name]
        conda activate [env-name]           # enter the conda virtual environment named [env-name]
        conda deactivate                    # exit the current conda virtual envrionment
        conda env remove --name [env-name]  # remove the conda virtual environment named [env-name]
        conda env list                      # list all conda virtual environments

.. _gpu_tensorflow:

Guide for TensorFlow GPU version installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GPU version of TensorFlow is able to utilize the powerful computing acceleration of NVIDIA GPU, making TensorFlow run more efficiently, especially multiplying the speed of training models.

Before installing TensorFlow GPU version, you need a not ratherly old NVIDIA graphics card, a properly installed driver, CUDA Toolkit and cuDNN.

Preperations for GPU hardwares
------------------------------

TensorFlow provides relatively complete support to NVIDIA graphics card. For NVIDIA graphics card, its CUDA Compute Capability must be at least 3.0. You can check your current card's CUDA Compute Capability on `NVIDIA official site <https://developer.nvidia.com/cuda-gpus/>`_. Now AMD graphics cards also start to provide support to TensorFlow. You can view `this blog article <https://medium.com/tensorflow/amd-rocm-gpu-support-for-tensorflow-33c78cc6a6cf>`_ for details.

Installation of NVIDIA drivers
------------------------------

**Windows** 

If your Windows system contain a NVIDIA graphics card, the NVIDIA driver is often installed automatically. If not, visit `NVIDIA official site <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_, download and install corresponding latest driver.

**Linux** 

If you use a Linux server version, samely visit `NVIDIA oficial site <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_ to download drivers (``.run`` file), then use ``sudo bash DRIVER_FILE_NAME.run`` command to install the driver. Before installation, you may need to use ``sudo apt-get install build-essential`` to install a proper compiling environment.

If you use a Linux desktop version with GUI, some extra configuration of NVIDIA graphics card driver is required, otherwise there may occur errors such as being unable to login. If you need to install an NVIDIA driver manually, remember to conduct the following steps before installation (take Ubuntu as example):

- Disable the system's own open-source graphics card driver Nouveau (add a line ``blacklist nouveau`` in ``/etc/modprobe.d/blacklist.conf`` then run ``sudo update-initramfs -u`` to update the kernel and then reboot);
- Disable Secure Boot on the motherboard;
- Disable desktop environment (e.g. ``sudo service lightdm stop``);
- Remove all existing NVIDIA driver programs (e.g. ``sudo apt-get purge nvidia*``).

.. admonition:: Tip 

    For desktop Ubuntu system, there is a rather easy way to install NVIDIA driver: clicking "Software & Updates" in System Setting, then ticking on "Using NVIDIA binary driver" checkbox and clicking "Apply Changes" in the right-down corner in "Additional Drivers". However, this may lead to a installation of a ratherly old version of NVIDIA driver.

After finishing installation of the NVIDIA driver, you can use ``nvidia-smi`` at the command line to check if it is successfully installed. It will print out information of the currently installed NVIDIA driver if succeeded, with the following forms:

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

.. admonition:: Hint

    The ``nvidia-smi`` command can view the existing GPUs on the machine and their usage. (On Windows, add ``C:\Program Files\NVIDIA Corporation\NVSMI`` to Path environment variable, or check graphics card information under the "performance" label of the task manager on Windows 10.)

For detailed instructions of GPU environment configuration, you can refer to `this article <https://www.linkedin.com/pulse/installing-nvidia-cuda-80-ubuntu-1604-linux-gpu-new-victor/>`_ and `this CHINESE blog <https://blog.csdn.net/wf19930209/article/details/81877822>`_.

Installation of CUDA Toolkit and cuDNN
--------------------------------------

In the Anaconda environment, it is recommended to use

::

    conda install cudatoolkit=X.X
    conda install cudnn=X.X.X

to install CUDA Toolkit and cuDNN, where X.X and X.X.X are respectively the version of CUDA Toolkit and cuDNN that be installed. Before installation, you can use ``conda search cudatoolkit`` and ``conda search cudnn`` to search for available version number from the conda source. For example, for TensorFlow 2.1, you can use::

    conda install cudatoolkit=10.1
    conda install cudnn=7.6.5

Of course you can also follow `the instructions from TensorFlow official site <https://www.tensorflow.org/install/gpu>`_ to download and install CUDA Toolkit and cuDNN manually. But this may be relatively complicated.

When using conda package manager to install the GPU version of TensorFlow, CUDA Toolkit and cuDNN with corresponding versions are also installed automatically. The updates from conda source are less frequent, but if you are okay with its versions, it is recommended to directly use ``conda install tensorflow-gpu`` for installation.

Your first program
^^^^^^^^^^^^^^^^^^

After installation, we can write a simple program to verify it.

Input ``conda activate tensorflow`` at the command line to enter the previously created conda virtual environment with TensorFlow installed, then input ``python`` to enter the Python environment. Run the following codes line by line:

.. code-block:: python

    import tensorflow as tf

    A = tf.constant([[1, 2], [3, 4]])
    B = tf.constant([[5, 6], [7, 8]])
    C = tf.matmul(A, B)

    print(C)

If it finally outputs::

    tf.Tensor(
    [[19 22]
    [43 50]], shape=(2, 2), dtype=int32)

Then it means that TensorFlow is installed successfully. There may be some TensorFlow prompts when running, which is normal.

.. admonition:: Some possible error messages and solutions when importing TensorFlow

    If you have TensorFlow 2.1 installed on Windows, you may experience a `DLL loading error when importing TensorFlow <https://github.com/tensorflow/tensorflow/issues/35749>`_ error. You can solve it by installing `Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019 <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_ .

    If your CPU is too old or entry-level (e.g., Intel's Atom series processors), your python environment may crash directly when importing TensorFlow. This is due to the lack of AVX instruction set. The AVX instruction set is added by default in the official version of TensorFlow in version 1.6 and later. If your CPU does not support the AVX instruction set, it will report an error (you can use CPU-Z on Windows or ``cat /proc/cpuinfo`` on Linux to see whether your CPU support AVX). In this case, it is recommended that you use a community version of your own hardware and software environment, such as `yaroslavvb/tensorflow-community-wheels <https://github.com/yaroslavvb/tensorflow-community-wheels>`_ on GitHub. As of June 2020, `this issue <https://github.com/yaroslavvb/tensorflow-community-wheels/issues/153>`_ includes the latest version of TensorFlow compiled under Ubuntu with AVX removed. You may also consider recompiling TensorFlow under your own platform. 

Here we use Python language. For tutorials for Python language you can refer to `runoob Python 3 tutorial <http://www.runoob.com/python3/python3-tutorial.html>`_ or `Liao Xuefeng's Python Tutorial <https://www.liaoxuefeng.com>`_. This handbook may assume that the readers possess basic knowledge of the Python langauge. Do not be neverous. Python is easy to learn, and TensorFlow is not involved with Python advanced features.

IDE configuration
^^^^^^^^^^^^^^^^^

For researchers and practitioners of machine learning, it is advised to use `PyCharm <http://www.jetbrains.com/pycharm/>`_ as the IDE for Python development.

When creating a new project, you need to indicate its Python Interpreter, that is, what Python environment to use to run your project. In fact, in the installation part, every conda virtual environment you created owns its independent Python Interpreter. You only need to add them. Choose "Add", then select "Existing Environment" in the following window. After that, select "[Anaconda installation path]/envs/[the name of the conda enviroment that be added]/python.exe" (exclude ".exe" suffix on Linux) at Interpreter, and finally click "OK". If you tick on "Make available to all projects", then all projects and select this Python Interpreter. Note that the default installation directory of Anaconda on Windows is special. It is usually ``C:\Users\[user name]\Anaconda3\`` or ``C:\Users\[user name]\AppData\Local\Continuum\anaconda3``, where ``AppData`` is a hidden folder.

For TensorFlow development, one of the most useful feature of the PyCharm Professional version is remote debugging. When you code on a terminal with limited performance while you also possess a high-performance computer (which usually contains high-performance GPUs) that can be accessed remotely by ssh, the remote debugging feature allows you to code on your local terminal as well as debug and run programs remotely (especially for model training). All modifications of codes and data that you make on the local terminal are automatically syncronized with the remote machine, which makes it seem to code on the remote machine, just like playing games with streaming. However, it requires high stability of internet connection when using remote debugging. If you need to train models in a long period of time, it is advised to login and train them directly on the remote machine (on Linux, by combining ``nohup`` command [#nohup]_, making the process run in the background, avoiding termination when existing shell). For detailed configuration steps of the remote debugging feature, please refer to `PyCharm documentation <https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html>`_.

.. admonition:: Tip

    If you are a student with a mail address ended with .edu, you can apply for a free PyCharm Professional version license `here <http://www.jetbrains.com/student/>`_.

For amateurs and beginners of TensorFlow and deep learning, `Visual Studio Code <https://code.visualstudio.com/>`_ or some online interactive Python environment (like the free `Google Colab <https://colab.research.google.com/>`_) are also good choices. For the usage of Colab please refer to :ref:`appendix <colab>`.

.. admonition:: Warning 

    If you are using an older version of PyCharm, you may experience a loss of some code auto-completion after installing TensorFlow 2. Upgrading to the new version of PyCharm (2019.3 and later) will resolve this issue.

.. [#nohup] Please refer to https://www.ibm.com/developerworks/cn/linux/l-cn-nohup/ for details of the ``nohup`` command.

The hardware configuration for TensorFlow *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Hint

    TensorFlow does not require much for hardwares for beginners. Meanwhile, with :ref:`free <colab>` or :ref:`flexible <gcp>` online computing resources, you can learn TensorFlow easily just with a computer connected to the internet!

In the stereotype of the most, TensorFlow and other deep learning frameworks "costs" hardwares greatly, so that the first thing for some people when starting with TensorFlow is to upgrade the computer's hardware. However, the required hardware for TensorFlow is largely based on the task and the circumstance:

- For TensorFlow beginners, you can also learn and master TensorFlow well without upgrading your hardwares. Most of the examples in this handbook are suitable for almost all popular PC (even without a GPU) without adding any other device. For some examples in this handbook that requires more computation (e.g. :ref:`train CNN image classification on cats_vs_dogs dataset <cats_vs_dogs>`), a common NVIDIA GPU may help greatly. If your own computer is not adequate for this, you may consider using online resources (e.g. :ref:`The free Colab <colab>`) to train models.

- For individuals or developers that particapte data science competitions (like Kaggle) or train models locally, a high-performance NVIDIA GPU is often necessary. The number of CUDA cores and the size of the graphics memory are two key factors of the GPU performance in machine learning, while the former determines the training speedm and the latter the size of the model and the maximum batch size, which are particularly sensitive in large-scale training.

- For edging machine learning researching (especially in computer vision and natural language processing), training with parallel mutliple GPUs are standard conditions. It is common to use 4, 8, or even more GPUs for rapid iterations of experiment results and acceleration of training large-scale models.

For reference, I give out my own hardware configuration of the development environment when I write this handbook:

- When I designed the example programs in this handbook, except for chapters related to distributed and cloud training, I test them locally on an ordinary desktop with Intel i5 CPUs and a 16 GB DDR3 memory without GPUs. For some models with larger scale of compuation, I train them with a 180-yuan NVIDIA P106-90 mining card that was bought on Taobao;
- I use a NVIDIA GTX 1060 (6 GB graphics memory) locally to conduct basic development and debugging for the long term of my research career. 
- The research laboratory I am in possesses a workstation with 4 NVIDIA GTX 1080 Ti paralleled (11 GB graphics memory per card) and a server with 10 NVIDIA GTX 1080 Ti paralleled (11 GB graphics memory per card) for training of edging computer vision models.
- The company that I once worked with use a server with 8 NVIDIA Tesla V100 paralleled (32 GB graphics memory per card) for training of the edging natural langauge processing models.

Although the hardware configurations of the research institutions and companies are deluxe, they are not as expensive as those apparatus and reagents that cost even millions of dollars in other edging scientific researching fields (e.g. biology). A deep learning server that costs from 10 to 50 thousands of dollars is able to serve several researchers for long. Thus machine learning is rather affordable for most of people.

For detailed configuration of a deep learning workstation, I am not going to list them due to the rapid update of the hardwares. It is recommended to follow the `question on Zhihu - How to configure a workstation for deep learning? <https://www.zhihu.com/question/33996159>`_ and combine with the latest market circumstances to DIY or order a workstation.
