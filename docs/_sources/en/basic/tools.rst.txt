Common Modules in TensorFlow
=====================================

.. admonition:: Prerequisite

    * `Python serialization module Pickle <http://www.runoob.com/python3/python3-inputoutput.html>`_ (not required)
    * `Python's special function parameter **kwargs <https://eastlakeside.gitbooks.io/interpy-zh/content/args_kwargs/Usage_kwargs.html>`_ (not required)
    * `Python iterator <https://www.runoob.com/python3/python3-iterator-generator.html>`_ 

Variable saving and restore: ``tf.train.Checkpoint``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/beta/guide/checkpoints

.. admonition:: Warning
    
    Checkpoint only saves the parameters (variables) of the model, not the calculation process of the model, so it is generally used to recover previously trained model parameters when the model source code is available. If you need to export the model (and run it without source code), please refer to :ref:`SavedModel <en_savedmodel>` in the "Deployment" section.

In many scenarios, we want to save the trained parameters (variables) after the model training is complete. By loading models and parameters elsewhere where they need to be used, you can get trained models directly. Probably the first thing that comes to mind is to use ``pickle`` in Python to serialize ``model.variables``. Unfortunately, TensorFlow's variable type ``ResourceVariable`` cannot be serialized.

The good thing is that, TensorFlow provides a powerful variable save and restore class, ``tf.train.Checkpoint``, which can save and restore all objects in TensorFlow that contain Checkpointable State using its ``save()`` and ``restore()`` methods. Specifically, TensorFlow instances including ``tf.keras.optimizer``, ``tf.Variable``, ``tf.keras.Layer`` and ``tf.keras.Model`` can be saved. 

Checkpoint is very easy to use, we start by declaring a Checkpoint.

.. code-block:: python

    checkpoint = tf.train.Checkpoint(model=model)

Here the ``tf.train.Checkpoint()`` accepts a special initialization parameter, a ``**kwargs``. It is a series of key-value pairs. The key names can be taken by your own, and the values are the objects to be saved. For example, if we want to save an instance of ``model`` inheriting ``tf.keras.Model`` and an optimizer ``optimizer`` inheriting ``tf.train.Optimizer``, we could write

.. code-block:: python

    checkpoint = tf.train.Checkpoint(myAwesomeModel=model, myAwesomeOptimizer=optimizer)

Here ``myAwesomeModel`` is an arbitrary key name we take for the ``model`` instance that we want to save. Note that we will also use this key name when recovering variables.

Next, when the model training is complete and needs to be saved, use

.. code-block:: python

    checkpoint.save(save_path_with_prefix)

and things are all set. ``save_path_with_prefix`` is the save directory with prefix.

.. admonition:: Note

    For example, by creating a folder named "save" in the source directory and calling ``checkpoint.save('./save/model.ckpt')``, we can find three files named ``checkpoint``, ``model.ckpt-1.index``, ``model.ckpt-1.data-00000-of-00001`` in the save directory, and these files record variable information. The ``checkpoint.save()`` method can be run several times, and each run will result in a ``.index`` file and a ``.data`` file, with serial numbers added in sequence.

When the variable values of a previously saved instance needs to be reloaded elsewhere, a checkpoint needs to be instantiated again, while keeping the key names consistent. Then call the checkpoint's restore method like this

.. code-block:: python

    model_to_be_restored = MyModel()                                        # the same class of model that need to be restored
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)   # keep the key name to be "myAwesomeModel"
    checkpoint.restore(save_path_with_prefix_and_index)

The model variables can then be recovered. ``save_path_with_prefix_and_index`` is the directory + prefix + number of the previously saved file. For example, calling ``checkpoint.store('./save/model.ckpt-1')`` will load a file with the prefix ``model.ckpt`` and a serial number of 1.

When multiple files are saved, we often want to load the most recent one. You can use ``tf.train.latest_checkpoint(save_path)`` to get the file name of the last checkpoint in the directory. For example, if there are 10 saved files in the save directory from ``model.ckpt-1.index`` to ``model.ckpt-10.index``, ``tf.train.most_checkpoint('./save')`` will return ``./save/model.ckpt-10``.

In general, a typical code framework for saving and restoring variables is as follows

.. code-block:: python

    # train.py: model training stage

    model = MyModel()
    # Instantiate the Checkpoint, specify the model instance to be saved 
    # (you can also add the optimizer if you want to save it)
    checkpoint = tf.train.Checkpoint(myModel=model)     
    # ... (model training code)
    # Save the variable values to a file when the training is finished 
    # (you can also save them regularly during the training)
    checkpoint.save('./save/model.ckpt')               

.. code-block:: python

    # test.py: model inference stage

    model = MyModel()
    # Instantiate the Checkpoint and specify the instance to be recovered.
    checkpoint = tf.train.Checkpoint(myModel=model)            
    # recover the variable values of the model instance 
    checkpoint.restore(tf.train.latest_checkpoint('./save'))   
    # ... (model inference code)

.. admonition:: Note

    ``tf.train.Checkpoint`` is stronger than the ``tf.train.Saver`` commonly used in TensorFlow 1.X, in that it supports "delayed" recovery of variables in eager execution mode. Specifically, when ``checkpoint.store()`` is called, but the variable in the model has not been created, Checkpoint can wait until the variable is created to recover the value. In eager execution mode, the initialization of the layers in the model and the creation of variables is done when the model is first called (the advantage is that the variable shape can be determined automatically based on the input tensor shape, without having to be specified manually). This means that when the model has just been instantiated, there is not even a single variable in it, and it is bound to raise error to recover the variable values if we keep the same way as before. For example, you can try calling the ``save_weight()`` method of ``tf.keras.Model`` in ``train.py`` to save the parameters of the model, and call the ``load_weight()`` method immediately after instantiating the model in ``test.py``. You will get an error. Only if you call the model once can you recover the variable values via ``load_weight()``. If you use ``tf.train.Checkpoint``, you do not need to worry about this. In addition, ``tf.train.Checkpoint`` also supports graph execution mode.

Finally, we provide an example of saving and restore of model variables, based on the :ref:`multi-layer perceptron <en_mlp>` in the previous chapter

.. literalinclude:: /_static/code/zh/tools/save_and_restore/mnist.py
    :emphasize-lines: 20, 30-32, 38-39

After creating the ``save`` folder in the code directory and running the training code, the ``save`` folder will hold model variable data that is saved every 100 batches. Adding ``--mode=test`` to the command line argument and running the code again, the model will be restored using the last saved variable values. Then we can directly obtain an accuracy rate of about 95% on test set.

.. admonition:: Use ``tf.train.CheckpointManager`` to delete old Checkpoints and customize file numbers

    During the training of the model, sometimes we'll have the following needs

    - After a long training period, the program will save a large number of Checkpoints, but we only want to keep the last few Checkpoints.
    - Checkpoint is numbered by default from 1, accruing 1 at a time, but we may want to use another numbering method (e.g. using the current number of batch as the file number).

    We can use TensorFlow's ``tf.train.CheckpointManager`` to satisfy these needs. After instantiating a Checkpoint, we instantiate a CheckpointManager

    .. code-block:: python

        checkpoint = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(checkpoint, directory='./save', checkpoint_name='model.ckpt', max_to_keep=k)

    Here, the ``directory`` parameter is the path to the saved file, ``checkpoint_name`` is the file name prefix (or ``ckpt`` by default if not provided), and ``max_to_keep`` is the number of retained checkpoints.

    When we need to save the model, we can use ``manager.save()`` directly. If we wish to assign our own number to the saved Checkpoint, we can add the ``checkpoint_number`` parameter to ``manager.save()`` like ``manager.save(checkpoint_number=100)``.

    The following code provides an example of using CheckpointManager to keep only the last three Checkpoint files and to use the number of the batch as the file number for the Checkpoint.

    .. literalinclude:: /_static/code/zh/tools/save_and_restore/mnist_manager.py
        :emphasize-lines: 22, 34

Visualization of training process: TensorBoard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/tensorboard/r2/get_started

Sometimes you may want to see how indicators change during model training (e.g. the loss value). Although it can be viewed via command line output, it seems not intuitive enough. And TensorBoard is a tool that can help us visualize the training process.

Real-time monitoring of indicator change
-------------------------------------------

To use TensorBoard, first create a folder (e.g. ``./tensorboard``) in the code directory to hold the TensorBoard log files, then instantiate a logger as follows

.. code-block:: python
    
    summary_writer = tf.summary.create_file_writer('./tensorboard')     # the parameter is the log folder we created

Next, when it is necessary to record the indicators during training, the value of the indicators during training at STEP can be logged by specifying the logger with the WITH statement and running ``tf.summary.scalar(name, tensor, step=batch_index)`` for the indicator (usually scalar) to be logged. The STEP parameters here can be set according to your own needs and can generally be set to the index of batch in the current training process. The overall framework is as follows.

.. code-block:: python

    summary_writer = tf.summary.create_file_writer('./tensorboard')    
    # start model training
    for batch_index in range(num_batches):
        # ... (training code, use variable "loss" to store current loss value)
        with summary_writer.as_default():                               # the logger to be used
            tf.summary.scalar("loss", loss, step=batch_index)
            tf.summary.scalar("MyScalar", my_scalar, step=batch_index)  # you can also add other indicators below

For every run of ``tf.summary.scalar()``, the logger writes a log to the log file. In addition to the simplest scalar (scalar), TensorBoard can also visualize other types of data like image and audio, as detailed in the `TensorBoard documentation <https://www.tensorflow.org/tensorboard/r2/get_started>`_.

When we want to visualize the training process, open the terminal in the code directory (and go to TensorFlow's conda environment if needed) and run::

    tensorboard --logdir=./tensorboard

The visual interface of the TensorBoard can then be accessed by using a browser to access the URL output from the command line program (usually http://name-of-your-computer:6006), as shown in the following figure.

.. figure:: /_static/image/tools/tensorboard.png
    :width: 100%
    :align: center

By default, TensorBoard updates data every 30 seconds. But you can also refresh manually by clicking the refresh button in the top right corner.

The following caveats apply to the use of TensorBoard.

* If retraining is required, the information in the logging folder needs to be deleted and TensorBoard need to be restarted (or you can create a new logging folder and start another TensorBoard process, with the ``-logdir`` parameter set to the newly created folder).
* Log folder path should not contain any special characters.

.. _en_graph_profile:

Visualize Graph and Profile Information
-------------------------------------------

We can also use ``tf.summary.trace_on`` to open Trace during training, where TensorFlow records a lot of information during training, such as the structure of the dataflow graph, and the time spent on each operation. When training is complete, you can use ``tf.summary.trace_export`` to export the recorded results to a file.

.. code-block:: python

    tf.summary.trace_on(graph=True, profiler=True)  # Open Trace option, then the dataflow graph and profiling information can be recorded
    # ... (training code)
    with summary_writer.as_default():
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)    # Save Trace information to a file

After that, we can select "Profile" in TensorBoard to see the time spent on each operation in a timeline. If you have created a dataflow graph using :ref:`tf.function <tffunction>`, you can also click "Graph" to view the graph structure.

.. figure:: /_static/image/tools/profiling.png
    :width: 100%
    :align: center

.. figure:: /_static/image/tools/graph.png
    :width: 100%
    :align: center

Example: visualize the training process of MLP
----------------------------------------------

Finally, we provide an example of the TensorBoard usage based on :ref:`multi-layer perceptron model <en_mlp>` in the previous chapter.

.. literalinclude:: /_static/code/zh/tools/tensorboard/mnist.py
    :emphasize-lines: 12-13, 21-22, 25-26

.. _en_tfdata:

Dataset construction and preprocessing: ``tf.data`` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/beta/guide/data

In many scenarios, we want to use our own datasets to train the model. However, the process of preprocessing and reading raw data files is often cumbersome and even more labor-intensive than the design of the model. For example, in order to read a batch of image files, we may need to struggle with python's various image processing packages (such as ``pillow``), design our own batch generation method, and finally may not run as efficiently as expected. To this end, TensorFlow provides the ``tf.data`` module, which includes a flexible set of dataset building APIs that help us quickly and efficiently build data input pipelines, especially for large-scale scenarios.

Dataset construction
-------------------------------------------

At the heart of ``tf.data`` is the ``tf.data.Dataset`` class, which provides a high-level encapsulation of the TensorFlow dataset. ``tf.data.Dataset`` consists of a series of iteratively accessible elements, each containing one or more tensor. For example, for a dataset consisting of images, each element can be an image tensor with the shape ``width x height x number of channels``, or a tuple consisting of an image tensor and an image label tensor.

The most basic way to build ``tf.data.Dataset`` is to use ``tf.data.Dataset.from_tensor_slices()`` for small amount of data (which can fit into the memory). Specifically, if all the elements of our dataset are stacked together into a large tensor through the 0-th dimension of the tensor (e.g., the training set of the MNIST dataset in the previous section is one large tensor with shape ``[60000, 28, 28, 1]``, representing 60,000 single-channel grayscale image of size 28*28), then we provide such one or more large tensor as input, and can construct the dataset by unstack it on the 0-th dimension of the tensor. In this case, the size of the 0-th dimension is the number of data elements in the dataset. We have an example as follows

.. literalinclude:: /_static/code/zh/tools/tfdata/tutorial.py
    :lines: 1-14
    :emphasize-lines: 11

Output::

    2013 12000
    2014 14000
    2015 15000
    2016 16500
    2017 17500

.. admonition:: Warning

    When multiple tensors are provided as input, the size of 0-th dimension of these tensors must be the same, and the multiple tensors must be spliced as tuples (i.e., using parentheses in Python).

Similarly, we can load the MNIST dataset in the previous chapter.

.. literalinclude:: /_static/code/zh/tools/tfdata/tutorial.py
    :lines: 16-25
    :emphasize-lines: 5

Output

.. figure:: /_static/image/tools/mnist_1.png
    :width: 40%
    :align: center

.. admonition:: Hint

    TensorFlow Datasets provides an out-of-the-box collection of datasets based on ``tf.data.Datasets``. You can view :doc:`TensorFlow Datasets <../appendix/tfds>` for the detailed usage. For example, we can load the MNIST dataset in just two lines of code:

    .. code-block:: python

        import tensorflow_datasets as tfds
        dataset = tfds.load("mnist", split=tfds.Split.TRAIN, as_supervised=True)

For extremely large datasets that cannot be fully loaded into the memory, we can first process the datasets in TFRecord format and then use ``tf.data.TFRocrdDataset()`` to load them. You can refer to :ref:`the TFRecord section <tfrecord>` for details.

Dataset preprocessing
-------------------------------------------

The ``tf.data.Dataset`` class provides us with a variety of dataset preprocessing methods. Some of the most commonly used methods are

- ``Dataset.map(f)``: apply the function ``f`` to each element of the dataset to obtain a new dataset (this part is often combined with ``tf.io`` to read, write and decode files and ``tf.image`` to process images).
- ``Dataset.shuffle(buffer_size)``: shuffle the dataset (set a fixed-size buffer, put the first ``buffer_size`` element in the buffer, and sample randomly from the buffer, replacing the sampled data with subsequent data).
- ``Dataset.batch(batch_size)``: batches the dataset, i.e. for each ``batch_size`` elements, using ``tf.stack()` to merge into one element on dimension 0.

In addition, there are ``Dataset.repeat()`` (repeat elements in the dataset), ``Dataset.reduce()`` (aggregation operation), ``Dataset.take()`` (interception of the first few elements of a dataset), etc. Further description can be found in the `API document <https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/Dataset>`_.

The following example is based on the MNIST data set.

Using ``Dataset.map()`` to rotate all pictures 90 degrees.

.. literalinclude:: /_static/code/zh/tools/tfdata/tutorial.py
    :lines: 27-37
    :emphasize-lines: 1-5

Output:

.. figure:: /_static/image/tools/mnist_1_rot90.png
    :width: 40%
    :align: center

Use ``Dataset.batch()`` to divide the dataset into batches, each with a size of 4.

.. literalinclude:: /_static/code/zh/tools/tfdata/tutorial.py
    :lines: 38-45
    :emphasize-lines: 1

Output:

.. figure:: /_static/image/tools/mnist_batch.png
    :width: 100%
    :align: center

Use ``Dataset.shuffle()`` to shuffle the dataset with the cache size set to 10000, and then set the batch.

.. literalinclude:: /_static/code/zh/tools/tfdata/tutorial.py
    :lines: 47-54
    :emphasize-lines: 1

Output:

.. figure:: /_static/image/tools/mnist_shuffle_1.png
    :width: 100%
    :align: center
    
    The first run

.. figure:: /_static/image/tools/mnist_shuffle_2.png
    :width: 100%
    :align: center
    
    The second run

It can be seen that each time the data is randomly shuffled.

.. admonition:: ``buffer_size`` setting of ``Dataset.shuffle()``

    As an iterator designed for large-scale data, ``tf.data.Dataset`` does not support easy access to the number of its own elements or random access to elements. Therefore, in order to shuffle the data set efficiently, some specific designed methods are needed. ``Dataset.shuffle()`` took the following approach.

    - Set a buffer with fixed size ``buffer_size``.
    - At initialization, the first ``buffer_size`` element of the dataset is moved to the buffer.
    - Each time an element needs to be randomly taken from the dataset, then one element is randomly sampled and taken out from the buffer (so there is an empty space in the buffer), and then one subsequent element in the dataset is taken out and put back into the empty space to maintain the size of the buffer.

    Therefore, the size of the buffer needs to be set reasonably according to the characteristics of the dataset. For example.

    - When ``buffer_size`` is set to 1, it is equivalent to no shuffling at all.
    - When the label order of the dataset is extremely unevenly distributed (e.g., the first half labels of the dataset are 0 and the second half labels are 1 in binary classification), a small buffer size will result in all elements in a batch to have same label, thus affecting the training effect. In general, the size of the buffer can be smaller if the distribution of the dataset is more random, otherwise a larger buffer is required.

.. _en_prefetch:

Increase the efficiency using the parallelization strategy of ``tf.data``
--------------------------------------------------------------------------

..
    https://www.tensorflow.org/guide/data_performance

When training models, we want to make the most of computing resources and reduce CPU/GPU idle time. However, sometimes, the preparation of dataset is very time-consuming, thus we have to spend a lot of time preparing data for training before each batch of training. When we are preparing the data, the GPU can only wait for data with no load, resulting in a waste of computing resources, as shown in the following figure.

.. figure:: /_static/image/tools/datasets_without_pipelining.png
    :width: 100%
    :align: center

    Original training process, GPU can only be idle when preparing data. `Source 1 <https://www.tensorflow.org/guide/data_performance>`_ 。

To tackle this problem, ``tf.data`` provides us with the ``Dataset.prefetch()`` method, which allows us to let the dataset prefetch several elements during training, so that the CPU can prepare data while training in the GPU, improving the efficiency of the training process, as shown below.

.. figure:: /_static/image/tools/datasets_with_pipelining.png
    :width: 100%
    :align: center
    
     The training process with ``Dataset.prefetch()``. The CPU preloads the data while the GPU is training the model, improving training efficiency. `Source 2 <https://www.tensorflow.org/guide/data_performance>`_ 。

The usage of ``Dataset.prefetch()`` is very similar to ``Dataset.batch()`` and ``Dataset.shuffle()`` in the previous section. Continuing with the MNIST dataset example, if you want to preloaded data, you can use the following code

.. code-block:: python

    mnist_dataset = mnist_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

Here the parameter ``buffer_size`` can be set either manually, or set to ``tf.data.experimental.AUTOTUNE`` to let TensorFlow select the appropriate value automatically.

Similarly, ``Dataset.map()`` can also transform data elements in parallel with multiple GPU resources to increase efficiency. Take the MNIST dataset as an example. assumes that the training machine has a 2-core CPU, and we want to take full advantage of the multi-core CPU to perform a parallelized transformation of the data (e.g. the 90-degree rotation function ``rot90`` in the previous section), we can using the following code

.. code-block:: python

    mnist_dataset = mnist_dataset.map(map_func=rot90, num_parallel_calls=2)

The operation process is shown in the following figure.

.. figure:: /_static/image/tools/datasets_parallel_map.png
    :width: 100%
    :align: center

    Parallelization of data conversion is achieved by setting the ``num_parallel_calls`` parameter of ``Dataset.map()``. The top part is unparallelized and the bottom part is 2-core parallel. `Source 3 <https://www.tensorflow.org/guide/data_performance>`_ 。

It is also possible to set ``num_parallel_calls`` to ``tf.data.experimental.AUTOTUNE`` to allow TensorFlow to automatically select the appropriate value.

In addition to this, there are a number of ways to improve dataset processing performance, which can be found in the `TensorFlow documentation <https://www.tensorflow.org/guide/data_performance>`_. The powerful performance of the tf.data parallelization policy is demonstrated in a later example, which can be viewed :ref:`here <en_tfdata_performance>`.

Fetching elements from datasets
-------------------------------------------
After the data is constructed and pre-processed, we need to iterate through it to get the data for training. ``tf.data.Dataset`` is an iteratable Python object, so data can be obtained using the For loop iteratively, namely.

.. code-block:: python

    dataset = tf.data.Dataset.from_tensor_slices((A, B, C, ...))
    for a, b, c, ... in dataset:
        # Operate on tensor a, b, c, etc., e.g. feed into model for training

You can also use ``iter()` to explicitly create a Python iterator and use ``next()` to get the next element, namely.

.. code-block:: python

    dataset = tf.data.Dataset.from_tensor_slices((A, B, C, ...))
    it = iter(dataset)
    a_0, b_0, c_0, ... = next(it)
    a_1, b_1, c_1, ... = next(it)

Keras supports the use of ``tf.data.Dataset`` directly as input. When calling the ``fit()`` and ``evaluate()`` methods of ``tf.keras.Model``, the input data ``x`` in the parameter can be specified as ``Dataset`` with all elements formatted as ``(input data, label data) ``. In this case, the parameter ``y`` (label data) can be ignored. For example, for the MNIST dataset mentioned above, the original Keras training approach is.

.. code-block:: python

    model.fit(x=train_data, y=train_label, epochs=num_epochs, batch_size=batch_size)

After using ``tf.data.Dataset``, we can pass the dataset directly into Keras API.

.. code-block:: python

    model.fit(mnist_dataset, epochs=num_epochs)

Since the dataset have already been divided into batches by the ``Dataset.batch()`` method, we do not need to provide the size of the batch to ``model.fit()``.

.. _en_cats_vs_dogs:

Example: cats_vs_dogs image classification
-------------------------------------------

The following code, using the "Cat and Dog" binary image classification task as an example, demonstrates the complete process of building, training and testing model with ``tf.data`` combined with ``tf.io`` and ``tf.image``. The dataset can be downloaded `here <https://www.floydhub.com/fastai/datasets/cats-vs-dogs>`_. The dataset should be decompressed into the ``data_dir`` directory in the code (here the default setting is ``C:/datasets/cats_vs_dogs``, which can be modified to suit your needs).

.. literalinclude:: /_static/code/zh/tools/tfdata/cats_vs_dogs.py
    :lines: 1-54
    :emphasize-lines: 13-17, 29-36, 54

Use the following code to test the model

.. literalinclude:: /_static/code/zh/tools/tfdata/cats_vs_dogs.py
    :lines: 56-70

.. _en_tfdata_performance:

By performing performance tests on the above examples, we can feel the powerful parallelization performance of ``tf.data``. Through the use of ``prefetch()`` and the addition of the ``num_parallel_calls`` parameter to the ``map()`` process, the model training time can be reduced to half or even less than before. The test results are as follows.

.. figure:: /_static/image/tools/tfdata_performance.jpg
    :width: 100%
    :align: center

    Parallelization performance test for tf.data (vertical axis is time taken per epoch, in seconds)

.. _en_tfrecord:

TFRecord: Dataset format of TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/tutorials/load_data/tfrecord

TFRecord is the dataset storage format in TensorFlow. Once we have organized the datasets into TFRecord format, TensorFlow can read and process them efficiently, helping us to train large-scale models more efficiently.

TFRecord can be understood as a file consisting of a series of serialized ``tf.train.Sample`` elements, each ``tf.train.Sample`` consisting of a dict of several ``tf.train.Feature``. The form is as follows.

::

    # dataset.tfrecords
    [
        {   # example 1 (tf.train.Example)
            'feature_1': tf.train.Feature,
            ...
            'feature_k': tf.train.Feature
        },
        ...
        {   # example N (tf.train.Example)
            'feature_1': tf.train.Feature,
            ...
            'feature_k': tf.train.Feature
        }
    ]


In order to organize the various datasets into TFRecord format, we can do the following steps for each element of the dataset.

- Read the data element into memory.
- Convert the element to `tf.train.example` objects (each `tf.train.example` consists of several ``tf.train.Feature``, so a dictionary of Feature needs to be created first).
- Serialize the ``tf.train.Sample``` object as a string and write it to a TFRecord file with a predefined ``tf.io.TFRecordWriter``.

To read the TFRecord data, follow these steps.

- Obtain a ``tf.data.Dataset`` instance by reading the original TFRecord file (notice that the ``tf.train.Sample`` object in the file has not been deserialized).
- Deserialize the ``tf.train.Sample`` string by ``tf.io.parse_single_example`` function for each serialized ``tf.train.Sample`` string in the dataset through the ``Dataset.map`` method.

In the following part, we show a code example to convert the training set of the :ref:`cats_vs_dogs dataset  <en_cats_vs_dogs>` into a TFRecord file and load this file.

Convert the dataset into a TFRecord file
-------------------------------------------

First, similar to the :ref:`previous section <en_cats_vs_dogs>`, we `download the dataset <https://www.floydhub.com/fastai/datasets/cats-vs-dogs>`_ and extract it to ``data_dir``. We also initialize the list of image filenames and tags for the dataset.

.. literalinclude:: /_static/code/zh/tools/tfrecord/cats_vs_dogs.py
    :lines: 1-12

Then, through the following code, we iteratively read each image, build the ``tf.train.Feature`` dictionary and the ``tf.train.Sample`` object, serialize it and write it to the TFRecord file.

.. literalinclude:: /_static/code/zh/tools/tfrecord/cats_vs_dogs.py
    :lines: 14-22

It is worth noting that ``tf.train.Feature`` supports three data formats.

- ``tf.train.BytesList``: string or binary files (e.g. image). Use ``bytes_list`` parameter to pass through a ``tf.train.BytesList`` object initialized by an array of strings or bytes.
- ``tf.train.FloatList`` : float or double numbers. Use ``float_list`` parameter to pass through a ``tf.train.FloatList`` object initialized by a float or double array.
- ``tf.train.Int64List`` : integers. Use ``int64_list`` parameter to pass through a ``tf.train.Int64List`` object initialized by an array of integers.

If you want to feed in only one element rather than an array, you can pass in an array with only one element.

With the code above, we can get a file sized around 500MB named ``train.tfrecords``.

Read the TFRecord file
-------------------------------------------

We can read the file ``train.tfrecords`` created in the previous section, and decode each serialized ``tf.train.Example`` object with ``Dataset.map`` and ``tf.io.parse_single_example`` .

.. literalinclude:: /_static/code/zh/tools/tfrecord/cats_vs_dogs.py
    :lines: 24-36

The ``feature_description`` is like a "description file" of a dataset, informing the ``tf.io.parse_single_example`` function the properties of each ``tf.train.sample`` element, through a dictionary of key-value pairs. The properties contain which features are available for each ``tf.train.sample`` element, and the type, shape, and other properties of those features. The three input parameters of ``tf.io.FixedLenFeatures``: ``shape``, ``dtype`` and ``default_value`` (optional) are the shape, type and default values for each Feature. Here our data items are single values or strings, so ``shape``` is an empty array.

After running the above code, we get a dataset instance ``dataset``, which is already a ``tf.data.Dataset`` instance that can be used for training! We output an element from this dataset to validate the code

.. literalinclude:: /_static/code/zh/tools/tfrecord/cats_vs_dogs.py
    :lines: 38-43

Output:

.. figure:: /_static/image/tools/tfrecord_cat.png
    :width: 60%
    :align: center

It can be seen that the images and labels are displayed correctly, and the data set is constructed successfully.

.. _tffunction:

Graph Execution mode: ``@tf.function`` *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the default Eager Execution mode gives us flexibility and ease of debugging, in some scenarios, we still want to use the Graph Execution mode (default in in TensorFlow 1.X) to transform the model into an efficient TensorFlow graph model, especially when we want high performance or to deploy models. Therefore, TensorFlow 2 provides us with the ``tf.function`` module, which, in conjunction with the AutoGraph mechanism, makes it easy to run the model in graph execution mode by simply adding a ``@tf.function``` decorator.

Basic usage of ``@tf.function``
-------------------------------------------

..
    https://www.tensorflow.org/beta/guide/autograph
    https://www.tensorflow.org/guide/autograph
    https://www.tensorflow.org/beta/tutorials/eager/tf_function
    https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/
    https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/
    https://pgaleone.eu/tensorflow/tf.function/2019/05/10/dissecting-tf-function-part-3/

In TensorFlow 2, it is recommended to use ``tf.function`` (instead of ``tf.Session`` in 1.X) to implement the graph execution, so that you can convert the model to an easy-to-deploy, high-performance TensorFlow graph model. To use tf.function, you can just simply encapsulate the code within a function, and decorate the function with ``@tf.function`` decorator, as shown in the example below. For an in-depth discussion of the graph execution mode, see :doc:`the appendix <../advanced/static>` .

.. admonition:: Warning

    Not all functions can be decorated by ``@tf.function``! ``@tf.function`` uses static compilation to convert the code within the function into a dataflow graph, so there are restrictions on the statements that can be used within the function (only a subset of the Python language is supported), and the operations within the function need to be able to act as a node in the computational graph. It is recommended to use only native TensorFlow operations within the function, not to use overly complex Python statements, and only include TensorFlow tensors or NumPy arrays in the function arguments. In conclusion, it will be better to build the function according to the idea of a dataflow graph. ``@tf.function`` just gives you a more convenient way to write computational graphs, not a `"silver bullet" <https://en.wikipedia.org/wiki/No_Silver_Bullet>`_ that will accelerate any function. Details are available at `AutoGraph Capabilities and Limitations <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md>`_. You can read this section together with :doc:`the appendix <../advanced/static>` for better understanding. 

.. literalinclude:: /_static/code/zh/model/autograph/main.py
    :emphasize-lines: 11, 18

With 400 batches, the program took 35.5 seconds with ``@tf.function`` and 43.8 seconds without ``@tf.function''. It can be seen that ``@tf.function`` brought some performance improvements. In general, ``@tf.function`` brings greater performance boost when the model is composed of many small operations. But if the model does not have much operations while each operation is time-consuming, the performance gains from ``@tf.function`` will not be significant.

..
    https://www.tensorflow.org/beta/guide/autograph
    Functions can be faster than eager code, for graphs with many small ops. But for graphs with a few expensive ops (like convolutions), you may not see much speedup.

``tf.function`` 内在机制
-------------------------------------------

当被 ``@tf.function`` 修饰的函数第一次被调用的时候，进行以下操作：

- 在即时执行模式关闭的环境下，函数内的代码依次运行。也就是说，每个 ``tf.`` 方法都只是定义了计算节点，而并没有进行任何实质的计算。这与TensorFlow 1.X的图执行模式是一致的；
- 使用AutoGraph将函数中的Python控制流语句转换成TensorFlow计算图中的对应节点（比如说 ``while`` 和 ``for`` 语句转换为 ``tf.while`` ， ``if`` 语句转换为 ``tf.cond`` 等等；
- 基于上面的两步，建立函数内代码的计算图表示（为了保证图的计算顺序，图中还会自动加入一些 ``tf.control_dependencies`` 节点）；
- 运行一次这个计算图；
- 基于函数的名字和输入的函数参数的类型生成一个哈希值，并将建立的计算图缓存到一个哈希表中。

在被 ``@tf.function`` 修饰的函数之后再次被调用的时候，根据函数名和输入的函数参数的类型计算哈希值，检查哈希表中是否已经有了对应计算图的缓存。如果是，则直接使用已缓存的计算图，否则重新按上述步骤建立计算图。

.. hint:: 对于熟悉 TensorFlow 1.X 的开发者，如果想要直接获得 ``tf.function`` 所生成的计算图以进行进一步处理和调试，可以使用被修饰函数的 ``get_concrete_function`` 方法。该方法接受的参数与被修饰函数相同。例如，为了获取前节被 ``@tf.function`` 修饰的函数 ``train_one_step`` 所生成的计算图，可以使用以下代码：

    .. code-block:: python

        graph = train_one_step.get_concrete_function(X, y)

    其中 ``graph`` 即为一个 ``tf.Graph`` 对象。

以下是一个测试题：

.. literalinclude:: /_static/code/zh/model/autograph/quiz.py
    :lines: 1-18

思考一下，上面这段程序的结果是什么？

答案是::

    The function is running in Python
    1
    2
    2
    The function is running in Python
    0.1
    0.2    

当计算 ``f(a)`` 时，由于是第一次调用该函数，TensorFlow进行了以下操作：

- 将函数内的代码依次运行了一遍（因此输出了文本）；
- 构建了计算图，然后运行了一次该计算图（因此输出了1）。这里 ``tf.print(x)`` 可以作为计算图的节点，但Python内置的 ``print`` 则不能被转换成计算图的节点。因此，计算图中只包含了 ``tf.print(x)`` 这一操作；
- 将该计算图缓存到了一个哈希表中（如果之后再有类型为 ``tf.int32`` ，shape为空的张量输入，则重复使用已构建的计算图）。

计算 ``f(b)`` 时，由于b的类型与a相同，所以TensorFlow重复使用了之前已构建的计算图并运行（因此输出了2）。这里由于并没有真正地逐行运行函数中的代码，所以函数第一行的文本输出代码没有运行。计算 ``f(b_)`` 时，TensorFlow自动将numpy的数据结构转换成了TensorFlow中的张量，因此依然能够复用之前已构建的计算图。

计算 ``f(c)`` 时，虽然张量 ``c`` 的shape和 ``a`` 、 ``b`` 均相同，但类型为 ``tf.float32`` ，因此TensorFlow重新运行了函数内代码（从而再次输出了文本）并建立了一个输入为 ``tf.float32`` 类型的计算图。

计算 ``f(d)`` 时，由于 ``d`` 和 ``c`` 的类型相同，所以TensorFlow复用了计算图，同理没有输出文本。

而对于 ``@tf.function`` 对Python内置的整数和浮点数类型的处理方式，我们通过以下示例展现：

.. literalinclude:: /_static/code/zh/model/autograph/quiz.py
    :lines: 18-24

结果为::

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

简而言之，对于Python内置的整数和浮点数类型，只有当值完全一致的时候， ``@tf.function`` 才会复用之前建立的计算图，而并不会自动将Python内置的整数或浮点数等转换成张量。因此，当函数参数包含Python内置整数或浮点数时，需要格外小心。一般而言，应当只在指定超参数等少数场合使用Python内置类型作为被 ``@tf.function`` 修饰的函数的参数。

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

Output:

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

不过，如果你依然钟情于TensorFlow传统的图执行模式也没有问题。TensorFlow 2 提供了 ``tf.compat.v1`` 模块以支持TensorFlow 1.X版本的API。同时，只要在编写模型的时候稍加注意，Keras的模型是可以同时兼容即时执行模式和图执行模式的。注意，在图执行模式下， ``model(input_tensor)`` 只需运行一次以完成图的建立操作。

例如，通过以下代码，同样可以在MNIST数据集上训练前面所建立的MLP或CNN模型：

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 112-136


关于图执行模式的更多内容可参见 :doc:`../appendix/static`。

``tf.TensorArray`` ：TensorFlow 动态数组 *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/api_docs/python/tf/TensorArray

在部分网络结构，尤其是涉及到时间序列的结构中，我们可能需要将一系列张量以数组的方式依次存放起来，以供进一步处理。当然，在即时执行模式下，你可以直接使用一个Python列表（List）存放数组。不过，如果你需要基于计算图的特性（例如使用 ``@tf.function`` 加速模型运行或者使用SavedModel导出模型），就无法使用这种方式了。因此，TensorFlow提供了 ``tf.TensorArray`` ，一种支持计算图特性的TensorFlow动态数组。

其声明的方式为：

- ``arr = tf.TensorArray(dtype, size, dynamic_size=False)`` ：声明一个大小为 ``size`` ，类型为 ``dtype`` 的TensorArray ``arr`` 。如果将 ``dynamic_size`` 参数设置为 ``True`` ，则该数组会自动增长空间。

其读取和写入的方法为：

- ``write(index, value)`` ：将 ``value`` 写入数组的第 ``index`` 个位置；
- ``read(index)`` ：读取数组的第 ``index`` 个值；

除此以外，TensorArray还包括 ``stack()`` 、 ``unstack()`` 等常用操作，可参考 `文档 <https://www.tensorflow.org/api_docs/python/tf/TensorArray>`_ 以了解详情。

请注意，由于需要支持计算图， ``tf.TensorArray`` 的 ``write()`` 方法是不可以忽略左值的！也就是说，在图执行模式下，必须按照以下的形式写入数组：

.. code-block:: python

    arr = arr.write(index, value)

这样才可以正常生成一个计算图操作，并将该操作返回给 ``arr`` 。而不可以写成：

.. code-block:: python

    arr.write(index, value)     # 生成的计算图操作没有左值接收，从而丢失

一个简单的示例如下：

.. literalinclude:: /_static/code/zh/tools/tensorarray/example.py

Output

::
    
    tf.Tensor(0.0, shape=(), dtype=float32) tf.Tensor(1.0, shape=(), dtype=float32) tf.Tensor(2.0, shape=(), dtype=float32)

``tf.config``：GPU的使用与分配 *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/beta/guide/using_gpu

指定当前程序使用的GPU
------------------------------------------- 

很多时候的场景是：实验室/公司研究组里有许多学生/研究员需要共同使用一台多GPU的工作站，而默认情况下TensorFlow会使用其所能够使用的所有GPU，这时就需要合理分配显卡资源。

首先，通过 ``tf.config.list_physical_devices`` ，我们可以获得当前主机上某种特定运算设备类型（如 ``GPU`` 或 ``CPU`` ）的列表，例如，在一台具有4块GPU和一个CPU的工作站上运行以下代码：

.. code-block:: python

    gpus = tf.config.list_physical_devices(device_type='GPU')
    cpus = tf.config.list_physical_devices(device_type='CPU')
    print(gpus, cpus)

Output:

.. code-block:: python

    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), 
     PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), 
     PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), 
     PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU')]     
    [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]

可见，该工作站具有4块GPU：``GPU:0`` 、 ``GPU:1`` 、 ``GPU:2`` 、 ``GPU:3`` ，以及一个CPU ``CPU:0`` 。

然后，通过 ``tf.config.set_visible_devices`` ，可以设置当前程序可见的设备范围（当前程序只会使用自己可见的设备，不可见的设备不会被当前程序使用）。例如，如果在上述4卡的机器中我们需要限定当前程序只使用下标为0、1的两块显卡（``GPU:0`` 和 ``GPU:1``），可以使用以下代码：

.. code-block:: python

    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.set_visible_devices(devices=gpus[0:2], device_type='GPU')

.. tip:: 使用环境变量 ``CUDA_VISIBLE_DEVICES`` 也可以控制程序所使用的GPU。假设发现四卡的机器上显卡0,1使用中，显卡2,3空闲，Linux终端输入::

        export CUDA_VISIBLE_DEVICES=2,3

    或在代码中加入

    .. code-block:: python

        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

    即可指定程序只在显卡2,3上运行。

设置显存使用策略
------------------------------------------- 

默认情况下，TensorFlow将使用几乎所有可用的显存，以避免内存碎片化所带来的性能损失。不过，TensorFlow提供两种显存使用策略，让我们能够更灵活地控制程序的显存使用方式：

- 仅在需要时申请显存空间（程序初始运行时消耗很少的显存，随着程序的运行而动态申请显存）；
- 限制消耗固定大小的显存（程序不会超出限定的显存大小，若超出的报错）。

可以通过 ``tf.config.experimental.set_memory_growth`` 将GPU的显存使用策略设置为“仅在需要时申请显存空间”。以下代码将所有GPU设置为仅在需要时申请显存空间：

.. code-block:: python

    gpus = tf.config.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

以下代码通过 ``tf.config.set_logical_device_configuration`` 选项并传入 ``tf.config.LogicalDeviceConfiguration`` 实例，设置TensorFlow固定消耗 ``GPU:0`` 的1GB显存（其实可以理解为建立了一个显存大小为1GB的“虚拟GPU”）：

.. code-block:: python

    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

.. hint:: TensorFlow 1.X 的 图执行模式 下，可以在实例化新的session时传入 ``tf.compat.v1.ConfigPhoto`` 类来设置TensorFlow使用显存的策略。具体方式是实例化一个 ``tf.ConfigProto`` 类，设置参数，并在创建 ``tf.compat.v1.Session`` 时指定Config参数。以下代码通过 ``allow_growth`` 选项设置TensorFlow仅在需要时申请显存空间：

    .. code-block:: python

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

    以下代码通过 ``per_process_gpu_memory_fraction`` 选项设置TensorFlow固定消耗40%的GPU显存：

    .. code-block:: python

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        tf.compat.v1.Session(config=config)

单GPU模拟多GPU环境
-------------------------------------------

当我们的本地开发环境只有一个GPU，但却需要编写多GPU的程序在工作站上进行训练任务时，TensorFlow为我们提供了一个方便的功能，可以让我们在本地开发环境中建立多个模拟GPU，从而让多GPU的程序调试变得更加方便。以下代码在实体GPU ``GPU:0`` 的基础上建立了两个显存均为2GB的虚拟GPU。

.. code-block:: python

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=2048),
         tf.config.LogicalDeviceConfiguration(memory_limit=2048)])

我们在 :ref:`单机多卡训练 <multi_gpu>` 的代码前加入以上代码，即可让原本为多GPU设计的代码在单GPU环境下运行。当输出设备数量时，程序会输出：

::

    Number of devices: 2

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 191 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>