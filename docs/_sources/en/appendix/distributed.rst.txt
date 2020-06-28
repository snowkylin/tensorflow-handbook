Distributed training with TensorFlow
============================================

..
    https://www.tensorflow.org/beta/guide/distribute_strategy

When we have a large number of computational resources, we can leverage these computational resources by using a suitable distributed strategy, which can significantly compress the time spent on model training. For different use scenarios, TensorFlow provides us with several distributed strategies in ``tf.distribute.Strategy`` that allow us to train models more efficiently.

.. _en_multi_gpu:

Training on a single machine with multiple GPUs: ``MirroredStrategy``
---------------------------------------------------------------------

..
    https://www.tensorflow.org/beta/tutorials/distribute/keras
    https://juejin.im/post/5ba9d72ff265da0ac849384b
    https://www.codercto.com/a/86644.html

``MirroredStrategy`` is a simple and high-performance, data-parallel, synchronous distributed strategy that supports training on multiple GPUs of the same machine. To use this strategy, we simply instantiate a ``MirroredStrategy`` strategy::

    strategy = tf.distribute.MirroredStrategy()

and place the model construction code in the context of ``strategy.scope()``::

    with strategy.scope():
        # Model construction code

.. admonition:: Tip

    You can specify devices in parameters such as::

        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    
    That is, only GPUs 0 and 1 are specified to participate in the distributed policy.
    
The following code demonstrates using the ``MirroredStrategy`` strategy to train MobileNetV2 using Keras on some of the image datasets in :doc:`TensorFlow Datasets <../appendix/tfds>`.

.. literalinclude:: /_static/code/zh/distributed/multi_gpu.py
    :emphasize-lines: 8-10, 21

In the following test, we used four NVIDIA GeForce GTX 1080 Ti graphics cards on the same machine to do multi-GPU training. The number of epochs is 5 in all cases. when using a single machine with no distributed configuration, although the machine still has four graphics cards, the program just trains directly, with batch size set to 64. When using a distributed training strategy, both total batch size of 64 (batch size of 16 distributed to a single machine) and total batch size of 256 (batch size of 64 distributed to a single machine) were tested.

============  ==============================  ================================  ================================
Dataset       No distributed strategy         Distributed training with 4 gpus  Distributed training with 4 gpus
                                              (batch size 64)                   (batch size 256)
============  ==============================  ================================  ================================
cats_vs_dogs  146s/epoch                      39s/epoch                         29s/epoch
tf_flowers    22s/epoch                       7s/epoch                          5s/epoch
============  ==============================  ================================  ================================

It can be seen that the speed of model training has increased significantly with MirroredStrategy.

.. admonition:: ``MirroredStrategy``` Process

    The steps of MirroredStrategy are as follows.

    - The strategy replicates a complete model on each of the N computing devices before training begins.
    - Each time a batch of data is passed in for training, the data is divided into N copies and passed into N computing devices (i.e. data parallel). 
    - N computing devices use local variables (mirror variables) to calculate the gradient of their data separately.
    - Apply all-reduce operations to efficiently exchange and sum gradient data between computing devices, so that each device eventually has the sum of all devices' gradients.
    - Update local variables (mirror variables) using the results of gradient summation.
    - After all devices have updated their local variables, the next round of training takes place (i.e., this parallel strategy is synchronized).

    By default, the ``MirroredStrategy`` strategy in TensorFlow uses NVIDIA NCCL for All-reduce operations.

Training on multiple machines: ``MultiWorkerMirroredStrategy`` 
--------------------------------------------------------------

..
    https://www.tensorflow.org/beta/tutorials/distribute/multi_worker_with_keras

Multi-machine distributed training in TensorFlow is similar to multi-GPU training in the previous section, just replacing ``MirroredStrategy`` with ``MultiWorkerMirroredStrategy``. However, there are some additional settings that need to be made as communication between multiple computers is involved. Specifically, the environment variable ``TF_CONFIG`` needs to be set, for example::

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ["localhost:20000", "localhost:20001"]
        },
        'task': {'type': 'worker', 'index': 0}
    })

``TF_CONFIG`` consists of two parts, ``cluster`` and ``task``.

- The ``cluster`` describes the structure of the entire multi-machine cluster and the network address (IP + port number) of each machine. The value of ``cluster`` is the same for each machine.
- The ``task`` describes the role of the current machine. For example, ``{'type': 'worker', 'index': 0}`` indicates that the current machine is the 0th worker in ``cluster`` (i.e. ``localhost:20000``). The ``task`` value of each machine needs to be set separately for the current host.

Once the above is set up, just run the training code on all machines one by one. The machine that runs first will wait before it is connected to other machines. When all the machines is connected, they will start training at the same time.

.. admonition:: Hint

    Please pay attention to the firewall settings on each machine, especially the need to open ports for communication with other machines. As in the example above, worker 0 needs to open port 20000 and worker 1 needs to open port 20001.

The training tasks in the following example are the same as in the previous section, except that they have been migrated to a multi-computer training environment. Suppose we have two machines, we first deploy the following program on both machines. The only difference is the ``task`` part, the first machine is set to ``{'type': 'worker', 'index': 0}`` and the second machine is set to ``{'type': 'worker', 'index': 1}``. Next, run the programs on both machines, and when the communication is established, the training process begins automatically.

.. literalinclude:: /_static/code/zh/distributed/multi_worker.py
    :emphasisize-lines: 10-18, 27

In the following tests, we build two separate virtual machine instances with a single NVIDIA Tesla K80 on Google Cloud Platform (see :ref:`the appendix <en_GCP>` for the usage of GCP), and report the training time with one GPU and the training time with two virtual machine instances for distributed training, respectively. The number of epochs is 5. The batch size is set to 64 when using a single machine with a single GPU, and tested with both a total batch size of 64 (batch size 32 when distributed to a single machine) and a total batch size of 128 (batch size 64 when distributed to a single machine) when using two machines with single GPU.

============  ==========================  ====================================  ====================================
Dataset       No distributed strategy     Distributed training with 2 machines  Distributed training with 2 machines
                                          (batch size 64)                       (batch size 128)
============  ==========================  ====================================  ====================================
cats_vs_dogs  1622s                       858s                                  755s
tf_flowers    301s                        152s                                  144s                               
============  ==========================  ====================================  ====================================

It can be seen that the speed of model training has also increased considerably.

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 359 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>


