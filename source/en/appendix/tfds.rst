TensorFlow Datasets: Ready-to-use Datasets
==========================================

`TensorFlow Datasets <https://www.tensorflow.org/datasets/>`_ is an out-of-the-box collection of dozens of commonly used machine learning datasets. The data can be loaded in the ``tf.data.Datasets`` format with only a few lines of code. For the use of ``tf.data.Datasets``, see :ref:`tf.data <en_tfdata>`.

The tool is a standalone Python package that can be installed via::

    pip install tensorflow-datasets

When using it, first import this package with TensorFlow

.. code-block:: python

    import tensorflow as tf
    import tensorflow_datasets as tfds

The most basic way to use this package is to load the required dataset with ``tfds.load`` method. For example, the following three lines of code loads the MNIST, "cat vs dog" and "tf_flowers" classification datasets, respectively.

.. code-block:: python

    dataset = tfds.load("mnist", split=tfds.Split.TRAIN, as_supervised=True)
    dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
    dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)

When a dataset is first loaded, TensorFlow Datasets will automatically download the dataset from the cloud to local, and show the download progress. For example, when loading the MNIST data set, the terminal output prompts the following.

::

    Downloading and preparing dataset mnist (11.06 MiB) to C:\Users\snowkylin\tensorflow_datasets\mnist\3.0.0...
    WARNING:absl:Dataset mnist is hosted on GCS. It will automatically be downloaded to your
    local data directory. If you'd instead prefer to read directly from our public
    GCS bucket (recommended if you're running on GCP), you can instead set
    data_dir=gs://tfds-data/datasets.

    Dl Completed...: 100%|██████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:10<00:00,  2.93s/ file] 
    Dl Completed...: 100%|██████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:10<00:00,  2.73s/ file] 
    Dataset mnist downloaded and prepared to C:\Users\snowkylin\tensorflow_datasets\mnist\3.0.0. Subsequent calls will reuse this data.

The ``tfds.load`` method returns an ``tf.data.Dataset`` object. Some of the important parameters are as follows.

..
    https://www.tensorflow.org/datasets/api_docs/python/tfds/load

- ``as_supervised``: If True, each row element in the dataset is organized into a pair ``(input, label)`` (i.e., "data + label") based on the characteristics of the dataset, otherwise each row element in the dataset is a dictionary with all the features.
- ``split``: Specifies a part of the dataset. If not specified, the entire data set is returned. Usually the datasets will have ``tfds.Split.TRAIN`` (training set) and ``tfds.Split.TEST``` (test set) options.

The currently supported datasets of TensorFlow Datasets can be viewed in the `official documentation <https://www.tensorflow.org/datasets/datasets>`_, or they can also be viewed using ``tfds.list_builders()``.

Once a dataset of the type ``tf.data.Dataset`` is available, we can use ``tf.data`` to perform various pre-processing operation of the dataset. For example.

.. code-block:: python
    
    # Load "tf_flowers" dataset using TessorFlow Datasets
    dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
    # Resize, shuffle and batch dataset
    dataset = dataset.map(lambda img, label: (tf.image.resize(img, [224, 224]) / 255.0, label)) \
        .shuffle(1024) \
        .batch(32)
    # iterate through data
    for images, labels in dataset:
        # do operations with images and labels

Detailed instructions can be found in :ref:`the tf.data section of this handbook <en_tfdata>` . The :doc:`distributed training <../appendix/distributed>` chapter also uses TensorFlow Datasets to load datasets. You can refer to the sample code in these sections for further information on how to use TensorFlow Datasets.

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 199 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>