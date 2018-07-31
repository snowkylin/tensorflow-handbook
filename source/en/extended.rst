TensorFlow Extensions
=====================

This chapter introduces some of the most commonly used TensorFlow extensions. Although these features are not "must", they make the process of model training and calling more convenient.

Prerequisites:

* `Python serialization module Pickle <http://www.runoob.com/python3/python3-inputoutput.html>`_ (not required)
* `Python special function parameters **kwargs <https://eastlakeside.gitbooks.io/interpy-zh/content/args_kwargs/Usage_kwargs.html>`_ (not required)

Checkpoint：Saving and Restoring Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many times, we hope to save the trained parameters (variables) after the model training is completed. By loading the model and parameters elsewhere in the model, you can get the trained model directly. Perhaps the first thing you think of is to store ``model.variables`` with the Python serialization module ``pickle``. But unfortunately, TensorFlow's variable type ``ResourceVariable`` cannot be serialized.

Fortunately, TensorFlow provides a powerful variable save and restore class `tf.train.Checkpoint <https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint>`_ , which can be used by ``save The ()`` and ``restore()`` methods to save and restore all objects in the TensorFlow that contain the Checkpointable State. Specifically, ``tf.train.Optimizer`` implementations, ``tf.Variable``, ``tf.keras.Layer`` implementations or ``tf.keras.Model`` implementations can all be saved. Its use is very simple, we first declare a Checkpoint:

.. code-block:: python

    checkpoint = tf.train.Checkpoint(model=model)

Here the initialization parameter accepted by ``tf.train.Checkpoint()`` is special, it is a ``**kwargs``. Specifically, it is a series of key-value pairs, and the key names can be taken at will, and the values are objects that need to be saved. For example, if we want to save a model instance ``model`` that inherits ``tf.keras.Model`` and an optimizer ``optimizer`` that inherits ``tf.train.Optimizer``, we can write:

.. code-block:: python

    checkpoint = tf.train.Checkpoint(myAwesomeModel=model, myAwesomeOptimizer=optimizer)

Here ``myAwesomeModel`` is any key name we take for the model ``model`` to be saved. Note that we will also use this key name when restoring variables.

Next, when the model training needs to be saved, use:

.. code-block:: python

    checkpoint.save(save_path_with_prefix)

is fine. ``save_path_with_prefix`` is the directory and prefix of the saved file. For example, if you create a folder named "save" in the source code directory and call ``checkpoint.save('./save/model.ckpt')`` once, we can find three files in the directory named ``checkpoint``, ``model.ckpt-1.index``, ``model.ckpt-1.data-00000-of-00001``, which record variable information. The ``checkpoint.save()`` method can be run multiple times. Each time it runs, we will get an .index file and a .data file. The serial numbers are added in order.

When you need to reload previously saved parameters for models elsewhere, you need to instantiate a checkpoint again, while keeping the key names consistent. Then call the restore method of checkpoint. Just like this:

.. code-block:: python

    model_to_be_restored = MyModel()                                        # The same model of the parameter to be restored
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)   # The key name remains as "myAwesomeModel"
    checkpoint.restore(save_path_with_prefix_and_index)

Then the model variables can be restored. ``save_path_with_prefix_and_index`` is the directory + prefix + number of the previously saved file. For example, calling ``checkpoint.restore('./save/model.ckpt-1')`` will load the file with the prefix ``model.ckpt`` and sequence number 1 to restore the model.

When saving multiple files, we often want to load the most recent one. You can use help function ``tf.train.latest_checkpoint(save_path)`` to return the file name of the most recent checkpoint in the directory. For example, if there are 10 saved files from ``model.ckpt-1.index`` to ``model.ckpt-10.index`` in the save directory, ``tf.train.latest_checkpoint('./save') `` then returns ``./save/model.ckpt-10``.

In general, the typical code framework for restoring and saving variables is as follows:

.. code-block:: python

    # train.py - Model training phase

    model = MyModel()
    checkpoint = tf.train.Checkpoint(myModel=model)     # Instantiate Checkpoint, specify the save object as model (if you need to save the optimizer's parameters, you can also add)
    # Model training code
    checkpoint.save('./save/model.ckpt')                # Save the parameters to a file after the model is trained, or save it once in a while in the model training process.

.. code-block:: python

    # test.py - Model use phase

    model = MyModel()
    checkpoint = tf.train.Checkpoint(myModel=model)             # Instantiate Checkpoint, specify the recovery object as model
    checkpoint.restore(tf.train.latest_checkpoint('./save'))    # Restore model parameters from file
    # Model usage code

By the way, ``tf.train.Checkpoint`` is more powerful than the ``tf.train.Saver``, which is commonly used in previous versions, because it supports "delayed" recovery variables under Eager Execution. Specifically, when ``checkpoint.restore()`` is called but the variables in the model have not yet been created, Checkpoint can wait until the variable is created before restoring the value. Under "Eager Execution" mode, the initialization of each layer in the model and the creation of variables are performed when the model is first called (the advantage is that the shape of the variable can be automatically determined based on the input tensor shape, without manual specification). This means that when the model has just been instantiated, there is actually no variable in it. At this time, using the previous method to recover the variable value will definitely cause an error. For example, you can try to save the parameters of model by calling the ``save_weight()`` method of ``tf.keras.Model`` in train.py, and call ``load_weight()`` method immediately after instantiating the model in test.py, it will cause an error. Only after calling the model and then run the ``load_weight()`` method can you get the correct result. It can be seen that ``tf.train.Checkpoint`` can bring us considerable convenience in this case. In addition, ``tf.train.Checkpoint`` also supports the Graph Execution mode.

Finally, an example is provided. The previous chapter's :ref:`multilayer perceptron model <mlp>` shows the preservation and loading of model variables:

.. literalinclude:: ../_static/code/zh/extended/save_and_restore/mnist.py

After the save folder is created in the code directory and the code is run for training, the model variable data stored every 100 batches will be stored in the save folder. Change line 7 to ``model = 'test'`` and run the code again. The model will be restored directly using the last saved variable value and tested on the test set. You can get about an accuracy of about 95% directly.

TensorBoard: Visualization of the Training Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes you want to see how the various parameters change during the model training (such as the value of the loss function). Although it can be viewed through the terminal output, it is sometimes not direct enough. TensorBoard is a tool that helps us visualize the training process.

Currently, TensorBoard support in Eager Execution mode is still in `tf.contrib.summary <https://www.tensorflow.org/api_docs/python/tf/contrib/summary>`_, and there may be more changes in the future. So here are just a simple example. First create a folder (such as ./tensorboard) in the code directory to store the TensorBoard record file, and instantiate a logger in the code:

.. code-block:: python
    
    summary_writer = tf.contrib.summary.create_file_writer('./tensorboard')

Next, put the code of training part in the context of ``summary_writer.as_default()`` and ``tf.contrib.summary.always_record_summaries()`` and run ``tf.contrib.summary.scalar(name, tensor, step=batch_index)`` to the parameters that need to be logged (generally scalar). The "step" parameter here can be set according to your own needs, and can commonly be set to the batch number in the current training process. The overall framework is as follows:

.. code-block:: python

    summary_writer = tf.contrib.summary.create_file_writer('./tensorboard')
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        # Start model training
        for batch_index in range(num_batches):
            # Training code, the current loss value of batch is put into the variable "loss"
            tf.contrib.summary.scalar("loss", loss, step=batch_index)
            tf.contrib.summary.scalar("MyScalar", my_scalar, step=batch_index)  # You can also add other variables

Each time you run ``tf.contrib.summary.scalar()``, the logger writes a record to the log file. In addition to the simplest scalar, TensorBoard can also visualize other types of data (such as images, audio, etc.) as described in the `API document <https://www.tensorflow.org/api_docs/python/tf/ Contrib/summary>`_.

When we want to visualize the training process, open the terminal in the code directory (and enter the TensaFlow conda environment if needed), run::

    tensorboard --logdir=./tensorboard

Then use the browser to access the URL output by the terminal (usually http://computer name:6006), you can access the visual interface of TensorBoard, as shown below:

.. figure:: ../_static/image/extended/tensorboard.png
    :width: 100%
    :align: center

By default, TensorBoard updates data every 30 seconds. However, you can also manually refresh by clicking the refresh button in the upper right corner.

The use of TensorBoard has the following notes:

* If you need to retrain, you need to delete the information in the record folder and restart TensorBoard (or create a new record folder and open TensorBoard, the ``--logdir`` parameter is set to the newly created folder);
* Language of the record folder directory should all be English.

Finally, an example is provided. The previous chapter's :ref:`multilayer perceptron model <mlp>` shows the use of TensorBoard:

.. literalinclude:: ../_static/code/zh/extended/tensorboard/mnist.py

GPU Usage and Allocation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many times the scenario is: there are many students/researchers in the lab/company research group who need to use the GPU, but there is only one multi-card machine. At this time, you need to pay attention to how to allocate graphics resources.

The command ``nvidia-smi`` can view the existing GPU and usage on the machine (in Windows, add ``C:\Program Files\NVIDIA Corporation\NVSMI`` to the environment variable "Path", or in Windows 10 you can view the graphics card information using the Performance tab of the Task Manager).

Use the environment variable ``CUDA_VISIBLE_DEVICES`` to control the GPU used by the program. Assume that the GPUs on a four-card machine is that 0,1 is in use and 2,3 is idle. Then input in Linux terminal ``export CUDA_VISIBLE_DEVICES=2,3`` to specify that the program runs only on GPU 2,3 (please note that this is NOT specified in the program code).

By default, TensorFlow will use almost all of the available graphic memory to avoid performance penalties caused by memory fragmentation. You can set the strategy for using TensorFlow to use graphic memory through the ``tf.ConfigProto`` class. The specific way is to instantiate a ``tf.ConfigProto`` class, set the parameters, and specify the "config" parameter when running ``tf.enable_eager_execution()``. The following code sets TensorFlow with the ``allow_growth`` option to apply for memory space only when needed:

.. code-block:: python

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)

The following code sets TensorFlow to consume 40% of GPU memory by the ``per_process_gpu_memory_fraction`` option:

.. code-block:: python

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    tf.enable_eager_execution(config=config)

Under "Graph Execution", you can also pass the tf.ConfigPhoto class to set up when instantiating a new session.
