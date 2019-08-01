TensorFlow常用模块
=====================================

.. admonition:: 前置知识

    * `Python的序列化模块Pickle <http://www.runoob.com/python3/python3-inputoutput.html>`_ （非必须）
    * `Python的特殊函数参数**kwargs <https://eastlakeside.gitbooks.io/interpy-zh/content/args_kwargs/Usage_kwargs.html>`_ （非必须）
    * `Python的迭代器 <https://www.runoob.com/python3/python3-iterator-generator.html>`_ 

``tf.train.Checkpoint`` ：变量的保存与恢复
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/beta/guide/checkpoints

很多时候，我们希望在模型训练完成后能将训练好的参数（变量）保存起来。在需要使用模型的其他地方载入模型和参数，就能直接得到训练好的模型。可能你第一个想到的是用Python的序列化模块 ``pickle`` 存储 ``model.variables``。但不幸的是，TensorFlow的变量类型 ``ResourceVariable`` 并不能被序列化。

好在TensorFlow提供了 ``tf.train.Checkpoint`` 这一强大的变量保存与恢复类，可以使用其 ``save()`` 和 ``restore()`` 方法将TensorFlow中所有包含Checkpointable State的对象进行保存和恢复。具体而言，``tf.keras.optimizer`` 、 ``tf.Variable`` 、 ``tf.keras.Layer`` 或者 ``tf.keras.Model`` 实例都可以被保存。其使用方法非常简单，我们首先声明一个Checkpoint：

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

.. _tfdata:

``tf.data`` ：数据集的构建与预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/beta/guide/data

很多时候，我们希望使用自己的数据集来训练模型。然而，面对一堆格式不一的原始数据文件，将其预处理并读入程序的过程往往十分繁琐，甚至比模型的设计还要耗费精力。比如，为了读入一批图像文件，我们可能需要纠结于python的各种图像处理包（比如 ``pillow`` ），自己设计Batch的生成方式，最后还可能在运行的效率上不尽如人意。为此，TensorFlow提供了 ``tf.data`` 这一模块，包括了一套灵活的数据集构建API，能够帮助我们快速、高效地构建数据输入的流水线，尤其适用于数据量巨大的场景。

数据集对象的建立
-------------------------------------------

``tf.data`` 的核心是 ``tf.data.Dataset`` 类，提供了对数据集的高层封装。``tf.data.Dataset`` 由一系列的可迭代访问的元素（element）组成，每个元素包含一个或多个张量。比如说，对于一个由图像组成的数据集，每个元素可以是一个形状为 ``长×宽×通道数`` 的图片张量，也可以是由图片张量和图片标签张量组成的元组（Tuple）。

最基础的建立 ``tf.data.Dataset`` 的方法是使用 ``tf.data.Dataset.from_tensor_slices()`` ，适用于数据量较小（能够整个装进内存）的情况。具体而言，如果我们的数据集中的所有元素通过张量的第0维，拼接成一个大的张量（例如，前节的MNIST数据集的训练集即为一个 ``[60000, 28, 28, 1]`` 的张量，表示了60000张28*28的单通道灰度图像），那么我们提供一个这样的张量或者第0维大小相同的多个张量作为输入，即可按张量的第0维展开来构建数据集，数据集的元素数量为张量第0位的大小。具体示例如下：

.. literalinclude:: /_static/code/zh/tools/tfdata/tutorial.py
    :lines: 1-14
    :emphasize-lines: 11

输出::

    2013 12000
    2014 14000
    2015 15000
    2016 16500
    2017 17500

.. warning:: 当提供多个张量作为输入时，张量的第0维大小必须相同，且必须将多个张量作为元组（Tuple，即使用Python中的小括号）拼接并作为输入。

类似地，我们可以载入前章的MNIST数据集：

.. literalinclude:: /_static/code/zh/tools/tfdata/tutorial.py
    :lines: 16-25
    :emphasize-lines: 5

输出

.. figure:: /_static/image/tools/mnist_1.png
    :width: 40%
    :align: center

.. hint:: TensorFlow Datasets提供了一个基于 ``tf.data.Datasets`` 的开箱即用的数据集集合，相关内容可参考 :ref:`TensorFlow Datasets <tfds>`。例如，使用以下语句：

    .. code-block:: python

        import tensorflow_datasets as tfds
        dataset = tfds.load("mnist", split=tfds.Split.TRAIN)

    即可快速载入MNIST数据集。

数据集对象的预处理
-------------------------------------------

``tf.data.Dataset`` 类为我们提供了多种数据集预处理方法。最常用的如：

- ``Dataset.map(f)`` ：对数据集中的每个元素应用函数 ``f`` ，得到一个新的数据集（这部分往往结合 ``tf.io`` 进行读写和解码文件， ``tf.image`` 进行图像处理）；
- ``Dataset.shuffle(buffer_size)`` ：将数据集打乱（设定一个固定大小的缓冲区（Buffer），取出前 ``buffer_size`` 个元素放入，并从缓冲区中随机采样，采样后的数据用后续数据替换）；
- ``Dataset.batch(batch_size)`` ：将数据集分成批次，即对每 ``batch_size`` 个元素，使用 ``tf.stack()`` 在第0维合并，成为一个元素。
- ``Dataset.prefetch()`` ：预取出数据集中的若干个元素

除此以外，还有 ``Dataset.repeat()`` （重复数据集的元素）、 ``Dataset.reduce()`` （与Map相对的聚合操作）、 ``Dataset.take()``（）等，可参考 `API文档 <https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/Dataset>`_ 进一步了解。

以下以MNIST数据集进行示例。

使用 ``Dataset.map()`` 将所有图片旋转90度：

.. literalinclude:: /_static/code/zh/tools/tfdata/tutorial.py
    :lines: 27-37
    :emphasize-lines: 1-5

输出

.. figure:: /_static/image/tools/mnist_1_rot90.png
    :width: 40%
    :align: center

使用 ``Dataset.batch()`` 将数据集划分批次，每个批次的大小为4：

.. literalinclude:: /_static/code/zh/tools/tfdata/tutorial.py
    :lines: 38-45
    :emphasize-lines: 1

输出

.. figure:: /_static/image/tools/mnist_batch.png
    :width: 100%
    :align: center

使用 ``Dataset.shuffle()`` 将数据打散后再设置批次，缓存大小设置为10000：

.. literalinclude:: /_static/code/zh/tools/tfdata/tutorial.py
    :lines: 47-54
    :emphasize-lines: 1

输出

.. figure:: /_static/image/tools/mnist_shuffle_1.png
    :width: 100%
    :align: center
    
    第一次运行

.. figure:: /_static/image/tools/mnist_shuffle_2.png
    :width: 100%
    :align: center
    
    第二次运行

可见每次的数据都会被随机打散。

.. admonition:: ``Dataset.shuffle()`` 时缓冲区大小 ``buffer_size`` 的设置

    ``tf.data.Dataset`` 作为一个针对大规模数据设计的迭代器，本身无法方便地获得自身元素的数量或随机访问元素。因此，为了高效且较为充分地打散数据集，需要一些特定的方法。``Dataset.shuffle()`` 采取了以下方法：

    - 设定一个固定大小为 ``buffer_size`` 的缓冲区（Buffer）；
    - 初始化时，取出数据集中的前 ``buffer_size`` 个元素放入缓冲区；
    - 每次需要从数据集中取元素时，即从缓冲区中随机采样一个元素并取出，然后从后续的元素中取出一个放回到之前被取出的位置，以维持缓冲区的大小。

    因此，缓冲区的大小需要根据数据集的特性和数据排列顺序特点来进行合理的设置。比如：

    - 当 ``buffer_size`` 设置为1时，其实等价于没有进行任何打散；
    - 当数据集的标签顺序分布极为不均匀（例如二元分类时数据集前N个的标签为0，后N个的标签为1）时，较小的缓冲区大小会使得训练时取出的Batch数据很可能全为同一标签，从而影响训练效果。一般而言，数据集的顺序分布若较为随机，则缓冲区的大小可较小，否则则需要设置较大的缓冲区。

数据集元素的获取与使用
-------------------------------------------
构建好数据并预处理后，我们需要从其中迭代获取数据以用于训练。``tf.data.Dataset`` 是一个Python的可迭代对象，因此可以使用For循环迭代获取数据，即：

.. code-block:: python

    dataset = tf.data.Dataset.from_tensor_slices((A, B, C, ...))
    for a, b, c, ... in dataset:
        # 对张量a, b, c等进行操作，例如送入模型进行训练

也可以使用 ``iter()`` 显式创建一个Python迭代器并使用 ``next()`` 获取下一个元素，即：

.. code-block:: python

    dataset = tf.data.Dataset.from_tensor_slices((A, B, C, ...))
    it = iter(dataset)
    a_0, b_0, c_0, ... = next(it)
    a_1, b_1, c_1, ... = next(it)

Keras支持使用 ``tf.data.Dataset`` 直接作为输入。当调用 ``tf.keras.Model`` 的 ``fit()`` 和 ``evaluate()`` 方法时，可以将参数中的输入数据 ``x`` 指定为一个元素格式为 ``(输入数据, 标签数据)`` 的 ``Dataset`` ，并忽略掉参数中的标签数据 ``y`` 。例如，对于上述的MNIST数据集，常规的Keras训练方式是：

.. code-block:: python

    model.fit(x=train_data, y=train_label, epochs=num_epochs, batch_size=batch_size)

使用 ``tf.data.Dataset`` 后，我们可以直接传入 ``Dataset`` ：

.. code-block:: python

    model.fit(mnist_dataset, epochs=num_epochs)

由于已经通过 ``Dataset.batch()`` 方法划分了数据集的批次，所以这里也无需提供批次的大小。

.. _cats_vs_dogs:

实例：cats_vs_dogs图像分类
-------------------------------------------

以下代码以猫狗图片二分类任务为示例，展示了使用 ``tf.data`` 结合 ``tf.io`` 和 ``tf.image`` 建立 ``tf.data.Dataset`` 数据集，并进行训练和测试的完整过程。数据集可至 `这里 <https://www.floydhub.com/fastai/datasets/cats-vs-dogs>`_ 下载。

.. literalinclude:: /_static/code/zh/tools/tfdata/cats_vs_dogs.py
    :lines: 1-51
    :emphasize-lines: 14-17, 29-33, 51

使用以下代码进行测试：

.. literalinclude:: /_static/code/zh/tools/tfdata/cats_vs_dogs.py
    :lines: 53-67

``@tf.function`` ：Graph Execution模式 *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``@tf.function`` 基础使用方法
-------------------------------------------

..
    https://www.tensorflow.org/beta/guide/autograph
    https://www.tensorflow.org/guide/autograph
    https://www.tensorflow.org/beta/tutorials/eager/tf_function
    https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/
    https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/
    https://pgaleone.eu/tensorflow/tf.function/2019/05/10/dissecting-tf-function-part-3/

在TensorFlow 2.0中，推荐使用 ``@tf.function`` （而非1.X中的 ``tf.Session`` ）实现Graph Execution，从而将模型转换为易于部署且高性能的TensorFlow图模型。只需要将我们希望以Graph Execution模式运行的代码封装在一个函数内，并在函数前加上 ``@tf.function`` 即可，如下例所示。

.. warning:: 并不是任何函数都可以被 ``@tf.function`` 修饰！``@tf.function`` 使用静态编译将函数内的代码转换成计算图，因此对函数内可使用的语句有一定限制（仅支持Python语言的一个子集），且需要函数内的操作本身能够被构建为计算图。建议在函数内只使用TensorFlow的原生操作，不要使用过于复杂的Python语句，函数参数只包括TensorFlow张量或NumPy数组，并最好是能够按照计算图的思想去构建函数（换言之，``@tf.function`` 只是给了你一种更方便的写计算图的方法，而不是一颗能给任何函数加速的 `银子弹 <https://en.wikipedia.org/wiki/No_Silver_Bullet>`_ ）。详细内容可参考 `AutoGraph Capabilities and Limitations <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/LIMITATIONS.md>`_ 。

.. literalinclude:: /_static/code/zh/model/autograph/main.py
    :emphasize-lines: 11, 18

运行400个Batch进行测试，加入 ``@tf.function`` 的程序耗时35.5秒，未加入 ``@tf.function`` 的纯Eager Execution程序耗时43.8秒。可见 ``@tf.function`` 带来了一定的性能提升。一般而言，当模型由较多小的操作组成的时候， ``@tf.function`` 带来的提升效果较大。而当模型的操作数量较少，但单一操作均很耗时的时候，则 ``@tf.function`` 带来的性能提升不会太大。

..
    https://www.tensorflow.org/beta/guide/autograph
    Functions can be faster than eager code, for graphs with many small ops. But for graphs with a few expensive ops (like convolutions), you may not see much speedup.

``@tf.function`` 内在机制
-------------------------------------------

当被 ``@tf.function`` 修饰的函数第一次被调用的时候，进行以下操作：

- 在Eager Execution模式关闭的环境下，函数内的代码依次运行。也就是说，每个 ``tf.`` 方法都只是定义了计算节点，而并没有进行任何实质的计算。这与TensorFlow 1.X的Graph Execution是一致的；
- 使用AutoGraph将函数中的Python控制流语句转换成TensorFlow计算图中的对应节点（比如说 ``while`` 和 ``for`` 语句转换为 ``tf.while`` ， ``if`` 语句转换为 ``tf.cond`` 等等；
- 基于上面的两步，建立函数内代码的计算图表示（为了保证图的计算顺序，图中还会自动加入一些 ``tf.control_dependencies`` 节点）；
- 运行一次这个计算图；
- 基于函数的名字和输入的函数参数的类型生成一个哈希值，并将建立的计算图缓存到一个哈希表中。

在被 ``@tf.function`` 修饰的函数之后再次被调用的时候，根据函数名和输入的函数参数的类型计算哈希值，检查哈希表中是否已经有了对应计算图的缓存。如果是，则直接使用已缓存的计算图，否则重新按上述步骤建立计算图。

以下是一个测试题：

.. literalinclude:: /_static/code/zh/model/autograph/quiz.py

思考一下，上面这段程序的结果是什么？

答案是::

    The function is running in Python
    1
    2
    2
    The function is running in Python
    0.1
    0.2
    The function is running in Python
    1
    The function is running in Python
    2
    1
    The function is running in Python
    0.1
    The function is running in Python
    0.2
    0.1

当计算 ``f(a)`` 时，由于是第一次调用该函数，TensorFlow进行了以下操作：

- 将函数内的代码依次运行了一遍（因此输出了文本）；
- 构建了计算图，然后运行了一次该计算图（因此输出了1）。这里 ``tf.print(x)`` 可以作为计算图的节点，但Python内置的 ``print`` 则不能被转换成计算图的节点。因此，计算图中只包含了 ``tf.print(x)`` 这一操作；
- 将该计算图缓存到了一个哈希表中（如果之后再有类型为 ``tf.int32`` ，shape为空的张量输入，则重复使用已构建的计算图）。

计算 ``f(b)`` 时，由于b的类型与a相同，所以TensorFlow重复使用了之前已构建的计算图并运行（因此输出了2）。这里由于并没有真正地逐行运行函数中的代码，所以函数第一行的文本输出代码没有运行。计算 ``f(b_)`` 时，TensorFlow自动将numpy的数据结构转换成了TensorFlow中的张量，因此依然能够复用之前已构建的计算图。

计算 ``f(c)`` 时，虽然张量 ``c`` 的shape和 ``a`` 、 ``b`` 均相同，但类型为 ``tf.float32`` ，因此TensorFlow重新运行了函数内代码（从而再次输出了文本）并建立了一个输入为 ``tf.float32`` 类型的计算图。

计算 ``f(d)`` 时，由于 ``d`` 和 ``c`` 的类型相同，所以TensorFlow复用了计算图，同理没有输出文本。

之后的计算结果则显示出 ``@tf.function`` 对Python内置的整数和浮点数类型的处理方式。简而言之，只有当值完全一致的时候， ``@tf.function`` 才会复用之前建立的计算图，而并不会自动将Python内置的整数或浮点数等转换成张量。因此，当函数参数包含Python内置整数或浮点数时，需要额外小心。一般而言，应当只在指定超参数等少数场合使用Python内置类型作为被 ``@tf.function`` 修饰的函数的参数。

..
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/function
    Note that unlike other TensorFlow operations, we don't convert python numerical inputs to tensors. Moreover, a new graph is generated for each distinct python numerical value, for example calling g(2) and g(3) will generate two new graphs (while only one is generated if you call g(tf.constant(2)) and g(tf.constant(3))). Therefore, python numerical inputs should be restricted to arguments that will have few distinct values, such as hyperparameters like the number of layers in a neural network. This allows TensorFlow to optimize each variant of the neural network.

下一个思考题：

.. literalinclude:: /_static/code/zh/model/autograph/quiz_2.py

这段代码的输出是::

    tf.Tensor(1.0, shape=(), dtype=float32)
    tf.Tensor(2.0, shape=(), dtype=float32)
    tf.Tensor(3.0, shape=(), dtype=float32)

正如同正文里的例子一样，你可以在被 ``@tf.function`` 修饰的函数里调用 ``tf.Variable`` 、 ``tf.keras.optimizers`` 、 ``tf.keras.Model`` 等包含有变量的数据结构。一旦被调用，这些结构将作为隐含的参数提供给函数。当这些结构内的值在函数内被修改时，在函数外也同样生效。

AutoGraph：将Python控制流转换为TensorFlow计算图
--------------------------------------------------------------------------------------

前面提到，``@tf.function`` 使用名为AutoGraph的机制将函数中的Python控制流语句转换成TensorFlow计算图中的对应节点。以下是一个示例，使用 ``tf.autograph`` 模块的低层API ``tf.autograph.to_code`` 将函数 ``square_if_positive`` 转换成TensorFlow计算图：

.. literalinclude:: /_static/code/zh/model/autograph/autograph.py

输出：

::

    tf.Tensor(1, shape=(), dtype=int32) tf.Tensor(0, shape=(), dtype=int32)
    def tf__square_if_positive(x):
        do_return = False
        retval_ = ag__.UndefinedReturnValue()
        cond = x > 0

        def get_state():
            return ()

        def set_state(_):
            pass

        def if_true():
            x_1, = x,
            x_1 = x_1 * x_1
            return x_1

        def if_false():
            x = 0
            return x
        x = ag__.if_stmt(cond, if_true, if_false, get_state, set_state)
        do_return = True
        retval_ = x
        cond_1 = ag__.is_undefined_return(retval_)

        def get_state_1():
            return ()

        def set_state_1(_):
            pass

        def if_true_1():
            retval_ = None
            return retval_

        def if_false_1():
            return retval_
        retval_ = ag__.if_stmt(cond_1, if_true_1, if_false_1, get_state_1, set_state_1)
        return retval_

我们注意到，原函数中的Python控制流 ``if...else...`` 被转换为了 ``x = ag__.if_stmt(cond, if_true, if_false, get_state, set_state)`` 这种计算图式的写法。AutoGraph起到了类似编译器的作用，能够帮助我们通过更加自然的Python控制流轻松地构建带有条件/循环的计算图，而无需手动使用TensorFlow的API进行构建。

使用传统的 ``tf.Session`` 
------------------------------------------- 

不过，如果你依然钟情于TensorFlow传统的Graph Execution模式也没有问题。TensorFlow 2.0提供了 ``tf.compat.v1`` 模块以支持TensorFlow 1.X版本的API。同时，只要在编写模型的时候稍加注意，Keras的模型是可以同时兼容Eager Execution模式和Graph Execution模式的。注意，在Graph Execution模式下， ``model(input_tensor)`` 只需运行一次以完成图的建立操作。

例如，通过以下代码，同样可以在MNIST数据集上训练前面所建立的MLP或CNN模型：

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 112-136


关于Graph Execution的更多内容可参见 :doc:`/zh/appendix/static`。

``tf.config``：GPU的使用与分配 *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/beta/guide/using_gpu

指定当前程序使用的GPU
------------------------------------------- 

很多时候的场景是：实验室/公司研究组里有许多学生/研究员都需要使用GPU，但多卡的机器只有一台，这时就需要注意如何分配显卡资源。

命令 ``nvidia-smi`` 可以查看机器上现有的GPU及使用情况（在Windows下，将 ``C:\Program Files\NVIDIA Corporation\NVSMI`` 加入Path环境变量中即可，或Windows 10下可使用任务管理器的“性能”标签查看显卡信息）。

使用环境变量 ``CUDA_VISIBLE_DEVICES`` 可以控制程序所使用的GPU。假设发现四卡的机器上显卡0,1使用中，显卡2,3空闲，Linux终端输入::

    export CUDA_VISIBLE_DEVICES=2,3

或在代码中加入

.. code-block:: python

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

即可指定程序只在显卡2,3上运行。

设置显存使用策略
------------------------------------------- 

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

Graph Execution下，也可以在实例化新的session时传入 ``tf.ConfigPhoto`` 类来进行设置。

模拟多GPU环境
------------------------------------------- 

