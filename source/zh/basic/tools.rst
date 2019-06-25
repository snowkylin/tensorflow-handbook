TensorFlow工具
=====================================

.. admonition:: 前置知识

    * `Python的序列化模块Pickle <http://www.runoob.com/python3/python3-inputoutput.html>`_ （非必须）
    * `Python的特殊函数参数**kwargs <https://eastlakeside.gitbooks.io/interpy-zh/content/args_kwargs/Usage_kwargs.html>`_ （非必须）

Checkpoint：变量的保存与恢复
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/beta/guide/checkpoints

很多时候，我们希望在模型训练完成后能将训练好的参数（变量）保存起来。在需要使用模型的其他地方载入模型和参数，就能直接得到训练好的模型。可能你第一个想到的是用Python的序列化模块 ``pickle`` 存储 ``model.variables``。但不幸的是，TensorFlow的变量类型 ``ResourceVariable`` 并不能被序列化。

好在TensorFlow提供了 ``tf.train.Checkpoint`` 这一强大的变量保存与恢复类，可以使用其 ``save()`` 和 ``restore()`` 方法将TensorFlow中所有包含Checkpointable State的对象进行保存和恢复。具体而言，``tf.train.Optimizer`` 、 ``tf.Variable`` 、 ``tf.keras.Layer`` 或者 ``tf.keras.Model`` 实例都可以被保存。其使用方法非常简单，我们首先声明一个Checkpoint：

.. code-block:: python

    checkpoint = tf.train.Checkpoint(model=model)

这里 ``tf.train.Checkpoint()`` 接受的初始化参数比较特殊，是一个 ``**kwargs`` 。具体而言，是一系列的键值对，键名可以随意取，值为需要保存的对象。例如，如果我们希望保存一个继承 ``tf.keras.Model`` 的模型实例 ``model`` 和一个继承 ``tf.train.Optimizer`` 的优化器 ``optimizer`` ，我们可以这样写：

.. code-block:: python

    checkpoint = tf.train.Checkpoint(myAwesomeModel=model, myAwesomeOptimizer=optimizer)

这里 ``myAwesomeModel`` 是我们为待保存的模型 ``model`` 所取的任意键名。注意，在恢复变量的时候，我们还将使用这一键名。

接下来，当模型训练完成需要保存的时候，使用：

.. code-block:: python

    checkpoint.save(save_path_with_prefix)

就可以。 ``save_path_with_prefix`` 是保存文件的目录+前缀。

.. note:: 例如，在源代码目录建立一个名为save的文件夹并调用一次 ``checkpoint.save('./save/model.ckpt')`` ，我们就可以在可以在save目录下发现名为 ``checkpoint`` 、  ``model.ckpt-1.index`` 、 ``model.ckpt-1.data-00000-of-00001`` 的三个文件，这些文件就记录了变量信息。``checkpoint.save()`` 方法可以运行多次，每运行一次都会得到一个.index文件和.data文件，序号依次累加。

当在其他地方需要为模型重新载入之前保存的参数时，需要再次实例化一个checkpoint，同时保持键名的一致。再调用checkpoint的restore方法。就像下面这样：

.. code-block:: python

    model_to_be_restored = MyModel()                                        # 待恢复参数的同一模型
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)   # 键名保持为“myAwesomeModel”
    checkpoint.restore(save_path_with_prefix_and_index)

即可恢复模型变量。 ``save_path_with_prefix_and_index`` 是之前保存的文件的目录+前缀+编号。例如，调用 ``checkpoint.restore('./save/model.ckpt-1')`` 就可以载入前缀为 ``model.ckpt`` ，序号为1的文件来恢复模型。

当保存了多个文件时，我们往往想载入最近的一个。可以使用 ``tf.train.latest_checkpoint(save_path)`` 这个辅助函数返回目录下最近一次checkpoint的文件名。例如如果save目录下有 ``model.ckpt-1.index`` 到 ``model.ckpt-10.index`` 的10个保存文件， ``tf.train.latest_checkpoint('./save')`` 即返回 ``./save/model.ckpt-10`` 。

总体而言，恢复与保存变量的典型代码框架如下：

.. code-block:: python

    # train.py 模型训练阶段

    model = MyModel()
    # 实例化Checkpoint，指定保存对象为model（如果需要保存Optimizer的参数也可加入）
    checkpoint = tf.train.Checkpoint(myModel=model)     
    # ...（模型训练代码）
    # 模型训练完毕后将参数保存到文件（也可以在模型训练过程中每隔一段时间就保存一次）
    checkpoint.save('./save/model.ckpt')               

.. code-block:: python

    # test.py 模型使用阶段

    model = MyModel()
    checkpoint = tf.train.Checkpoint(myModel=model)             # 实例化Checkpoint，指定恢复对象为model
    checkpoint.restore(tf.train.latest_checkpoint('./save'))    # 从文件恢复模型参数
    # 模型使用代码


.. note:: ``tf.train.Checkpoint`` 与以前版本常用的 ``tf.train.Saver`` 相比，强大之处在于其支持在Eager Execution下“延迟”恢复变量。具体而言，当调用了 ``checkpoint.restore()`` ，但模型中的变量还没有被建立的时候，Checkpoint可以等到变量被建立的时候再进行数值的恢复。Eager Execution下，模型中各个层的初始化和变量的建立是在模型第一次被调用的时候才进行的（好处在于可以根据输入的张量形状而自动确定变量形状，无需手动指定）。这意味着当模型刚刚被实例化的时候，其实里面还一个变量都没有，这时候使用以往的方式去恢复变量数值是一定会报错的。比如，你可以试试在train.py调用 ``tf.keras.Model`` 的 ``save_weight()`` 方法保存model的参数，并在test.py中实例化model后立即调用 ``load_weight()`` 方法，就会出错，只有当调用了一遍model之后再运行 ``load_weight()`` 方法才能得到正确的结果。可见， ``tf.train.Checkpoint`` 在这种情况下可以给我们带来相当大的便利。另外， ``tf.train.Checkpoint`` 同时也支持Graph Execution模式。

最后提供一个实例，以前章的 :ref:`多层感知机模型 <mlp>` 为例展示模型变量的保存和载入：

.. literalinclude:: /_static/code/zh/tools/save_and_restore/mnist.py
    :emphasize-lines: 20, 30-32, 38-39

在代码目录下建立save文件夹并运行代码进行训练后，save文件夹内将会存放每隔100个batch保存一次的模型变量数据。在命令行参数中加入 ``--mode=test`` 并再次运行代码，将直接使用最后一次保存的变量值恢复模型并在测试集上测试模型性能，可以直接获得95%左右的准确率。

.. admonition:: 使用 ``tf.train.CheckpointManager`` 删除旧的Checkpoint以及自定义文件编号

    在模型的训练过程中，我们往往每隔一定步数保存一个Checkpoint并进行编号。不过很多时候我们会有这样的需求：

    - 在长时间的训练后，程序会保存大量的Checkpoint，但我们只想保留最后的几个Checkpoint；
    - Checkpoint默认从1开始编号，每次累加1，但我们可能希望使用别的编号方式（例如使用当前Batch的编号作为文件编号）。

    这时，我们可以使用TensorFlow的 ``tf.train.CheckpointManager`` 来实现以上需求。具体而言，在定义Checkpoint后接着定义一个CheckpointManager：

    .. code-block:: python

        checkpoint = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(checkpoint, directory='./save', checkpoint_name='model.ckpt', max_to_keep=k)

    此处， ``directory`` 参数为文件保存的路径， ``checkpoint_name`` 为文件名前缀（不提供则默认为 ``ckpt`` ）， ``max_to_keep`` 为保留的Checkpoint数目。

    在需要保存模型的时候，我们直接使用 ``manager.save()`` 即可。如果我们希望自行指定保存的Checkpoint的编号，则可以在保存时加入 ``checkpoint_number`` 参数。例如 ``manager.save(checkpoint_number=100)`` 。

    以下提供一个实例，展示使用CheckpointManager限制仅保留最后三个Checkpoint文件，并使用batch的编号作为Checkpoint的文件编号。

    .. literalinclude:: /_static/code/zh/tools/save_and_restore/mnist_manager.py
        :emphasize-lines: 22, 34

..
    AutoGraph：动态图转静态图 *
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    `AutoGraph <https://www.tensorflow.org/guide/autograph>`_ 

    SavedModel：模型的封装 *
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


TensorBoard：训练过程可视化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/tensorboard/r2/get_started

有时，你希望查看模型训练过程中各个参数的变化情况（例如损失函数loss的值）。虽然可以通过命令行输出来查看，但有时显得不够直观。而TensorBoard就是一个能够帮助我们将训练过程可视化的工具。

首先在代码目录下建立一个文件夹（如 ``./tensorboard`` ）存放TensorBoard的记录文件，并在代码中实例化一个记录器：

.. code-block:: python
    
    summary_writer = tf.summary.create_file_writer('./tensorboard')     # 参数为记录文件所保存的目录

接下来，当需要记录训练过程中的参数时，通过with语句指定希望使用的记录器，并对需要记录的参数（一般是scalar）运行 ``tf.summary.scalar(name, tensor, step=batch_index)`` ，即可将训练过程中参数在step时候的值记录下来。这里的step参数可根据自己的需要自行制定，一般可设置为当前训练过程中的batch序号。整体框架如下：

.. code-block:: python

    summary_writer = tf.summary.create_file_writer('./tensorboard')    
    # 开始模型训练
    for batch_index in range(num_batches):
        # ...（训练代码，当前batch的损失值放入变量loss中）
        with summary_writer.as_default():                               # 希望使用的记录器
            tf.summary.scalar("loss", loss, step=batch_index)
            tf.summary.scalar("MyScalar", my_scalar, step=batch_index)  # 还可以添加其他自定义的变量

每运行一次 ``tf.summary.scalar()`` ，记录器就会向记录文件中写入一条记录。除了最简单的标量（scalar）以外，TensorBoard还可以对其他类型的数据（如图像，音频等）进行可视化，详见 `TensorBoard文档 <https://www.tensorflow.org/tensorboard/r2/get_started>`_ 。

当我们要对训练过程可视化时，在代码目录打开终端（如需要的话进入TensorFlow的conda环境），运行::

    tensorboard --logdir=./tensorboard

然后使用浏览器访问命令行程序所输出的网址（一般是http://计算机名称:6006），即可访问TensorBoard的可视界面，如下图所示：

.. figure:: /_static/image/tools/tensorboard.png
    :width: 100%
    :align: center

默认情况下，TensorBoard每30秒更新一次数据。不过也可以点击右上角的刷新按钮手动刷新。

TensorBoard的使用有以下注意事项：

* 如果需要重新训练，需要删除掉记录文件夹内的信息并重启TensorBoard（或者建立一个新的记录文件夹并开启TensorBoard， ``--logdir`` 参数设置为新建立的文件夹）；
* 记录文件夹目录保持全英文。

最后提供一个实例，以前章的 :ref:`多层感知机模型 <mlp>` 为例展示TensorBoard的使用：

.. literalinclude:: /_static/code/zh/tools/tensorboard/mnist.py
    :emphasize-lines: 11, 19-20

``tf.data`` ：数据预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

