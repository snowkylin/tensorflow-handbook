TensorFlow Installation
=========================

The latest installation guide for TensorFlow can be found on its official website (https://tensorflow.google.com/install). TensorFlow supports a variety of programming languages such as Python, Java, Go, C, and various operating systems such as Windows, OSX, and Linux. Python language is used here and below.

Two ways of installing is provided below for readers of different levels.

Simple Installation
^^^^^^^^^^^^^^^^^^^^^
If you just install a simple environment that runs on your own computer without a GPU and you don't want to spend too much energy on the environment configuration, it is recommended to install it according to the following steps (take Windows system as an example):

- Download and install the Python integration environment `Anaconda <https://www.anaconda.com/download/>`_ (Python 3.6 version);
- Download and install the Python IDE `PyCharm <http://www.jetbrains.com/pycharm/>`_ (Community version, or students can apply for the Professional version of the free license);
- Open "Anaconda Prompt" in the start menu and type ``pip install tensorflow``.

Detailed Installation Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This section contains more details on the installation (such as establishing a conda environment) and the environment configuration method of the GPU version of TensorFlow.

Environment Setting
-------------------------------
Before you officially install TensorFlow, you need to configure the right environment for it. Proceed as follows:

1. Check if your computer has an NVIDIA graphics card. If so, it is recommended to install the GPU version of TensorFlow to take advantage of the GPU's powerful computational acceleration [#f1]_, otherwise the CPU version can be installed. Specifically, the CUDA Compute Capability of the graphics card must be at least 3.0. You can check the CUDA Compute Capability of your graphics card on `the official website of NVIDIA <https://developer.nvidia.com/cuda-gpus/>`_;
2. Install the Python environment. It is recommended to install Anaconda, an open source Python distribution that provides a complete scientific computing environment, including common scientific computing libraries such as NumPy and SciPy. Of course, you have the right to choose your favorite Python environment. Note that as of this writing, TensorFlow's installation under Windows only supports the Python 3.X version;

    * When installing Anaconda, you can choose to add the Anaconda directory to your system's PATH (although the installer does not recommend this), so you can use Anaconda's features directly in the command line environment. Of course, if you don't add it, you can use the Anaconda Prompt in the start menu to enter the command line Anaconda environment.

3. (For GPU version) Install NVIDIA graphics driver, `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_ and `cuDNN <https://developer.nvidia.com/cudnn>`_. Some things to note are:

    * The recommended order is: first install the latest version of NVIDIA graphics driver, then install CUDA (do not choose to install the driver at the same time), and finally install cuDNN. The graphics card driver included with CUDA may be too old;
    * There is a very simple driver installation method under Ubuntu: select Software & Updates in System Setting, then click the "Using NVIDIA binary driver" option in Additional Drivers and click on the lower right corner. The "Apply Changes" can be installed, the system will automatically install the NVIDIA driver. Otherwise, the installation of NVIDIA graphics driver on Linux system will not be smooth, pay attention to disable the system's own open source graphics driver Nouveau, disable the motherboard's Secure Boot function before installation. For more detailed guidance, please refer to `this article <https://www.linkedin.com/pulse/installing-nvidia-cuda-80-ubuntu-1604-linux-gpu-new-victor/>`_ ;
    * The version of CUDA Toolkit and cuDNN must be consistent with the version of the TensorFlow official website installation instructions. Note that the version required to be installed in the official website installation instructions may not be the latest version;
    * cuDNN is installed in a special way. You need to manually copy the downloaded installation package to the CUDA installation directory.


Installation
----------------

The installation process in Anaconda environment is as follows (take Windows system as an example):

1. Create a new conda environment called ``tensorflow``

::

    Conda create -n tensorflow python=X.X # Note that X.X here fills in the version of your own Python environment, such as 3.6

2. Activate the environment

::

    Activate tensorflow

3. Install TensorFlow using pip

Install the CPU version
::

    Pip install tensorflow

Install GPU version
::

    Pip install tensorflow-gpu

If necessary, you can also install the Nightly version of TensorFlow, which has some of the latest features compared to the official version (for example, before TensorFlow version 1.8, the Eager Execution mode used in this manual is only available in the Nightly version), however Stability may be slightly weaker. Run ``pip install tf-nightly`` (CPU version) or ``pip install tf-nightly-gpu`` (GPU version) in a new virtual environment. Note that if you install a GPU version, it often requires installation of CUDA and cuDNN that are newer than the official version requirements. Fortunately, different versions of CUDA and cuDNN can coexist.

If you use the pip command to install slowly, you can try the TensorFlow image of the `Tsinghua University open source software image station <https://mirrors.tuna.tsinghua.edu.cn/help/tensorflow/>`_.

The First Program
^^^^^^^^^^^^^^^^^

After the installation is complete, let's write a simple program to verify the installation.

Enter ``activate tensorflow`` at the command line. Go to the previously created conda environment with TensorFlow installed, then type ``python`` into the Python environment and enter the following code line by line:

.. code-block:: python

    import tensorflow as tf
    tf.enable_eager_execution()

    A = tf.constant([[1, 2], [3, 4]])
    B = tf.constant([[5, 6], [7, 8]])
    C = tf.matmul(A, B)

    print(C)

If you can finally output::

    tf.Tensor(
    [[19 22]
    [43 50]], shape=(2, 2), dtype=int32)

The TensorFlow has been successfully installed. Some TensorFlow prompts may be output during the operation, which is normal.

The Python language is used here. The introductory tutorial on the Python language can be found at http://www.runoob.com/python3/python3-tutorial.html or https://www.liaoxuefeng.com. The default reader will be used after this manual. Have the basic knowledge of the Python language. Without being nervous, the Python language is easy to use, and TensorFlow itself does not use too many advanced features of the Python language. For Python IDEs, it is recommended to use `PyCharm <http://www.jetbrains.com/pycharm/>`_ . If you are a student and have a .edu-terminated mailbox, you can apply for a free license `here <http://www.jetbrains.com/student/>`_ . If not, you can also download the community version of PyCharm, the main features are not much different.

.. [#f1] The effect of GPU acceleration is related to the model type and the performance of the GPU. If the CPU performance is high, but the GPU only has entry-level performance, the speed is not much improved, about 1-2 times. However, if the GPU performance is strong (for example, when writing this manual, NVIDIA GeForce GTX 1080 Ti or NVIDIA GeForce TITAN series is a high performance graphics card on the market), for a specific model, more than ten times or more acceleration can be Reached. At the same time, the acceleration effect of the GPU is also related to the task itself. The entry-level TensorFlow model often does not require too much computational performance. The CPU version of TensorFlow is sufficient, so you can wait until you have mastered the basics of TensorFlow before deciding whether to purchase a more advanced GPU for faster training.
