附录5：TensorFlow环境配置与管理
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

使用Docker部署TensorFlow环境
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在Colab中使用TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^