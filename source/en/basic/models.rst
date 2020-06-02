Model Construction and Training
===============================

.. _linear:

This chapter describes how to build models with Keras and Eager Execution using TensorFlow 2.

- Model construction: ``tf.keras.Model`` and ``tf.keras.layers``
- Loss function of the model: ``tf.keras.losses``
- Optimizer of the model: ``tf.keras.optimizer``
- Evaluation of models: ``tf.keras.metrics``

.. ADMONITION:: Prerequisite

    * `Object-oriented Python programming <http://www.runoob.com/python3/python3-class.html>`_ (define classes and methods, class inheritance, constructor and deconstructor within Python, `use super() functions to call parent class methods <http://www.runoob.com/python/python-func-super.html>`_, `use __call__() methods to call instances <https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014319098638265527beb24f7840aa97de564ccc7f20f6000>`_, etc.).
    * Multilayer perceptron, convolutional neural networks, recurrent neural networks and reinforcement learning (references given before each section).
    * `Python function decorator <https://www.runoob.com/w3cnote/python-func-decorators.html>`_ (not required)

Models and layers
^^^^^^^^^^^^^^^^^
..  https://www.tensorflow.org/programmers_guide/eager

In TensorFlow, it is recommended to build models using Keras (``tf.keras``), a popular high-level neural network API that is simple, fast and flexible. It is officially built-in and fully supported by TensorFlow.

There are two important concepts in Keras: **Model** and **Layer** . The layers encapsulate various computational processes and variables (e.g., fully connected layers, convolutional layers, pooling layers, etc.), while the model connects the layers and encapsulates them as a whole, describing how the input data is passed through the layers and operations to get the output. Keras has built in a number of predefined layers commonly used in deep learning under ``tf.keras.layers``, while also allowing us to customize the layers.

Keras models are presented as classes, and we can define our own models by inheriting the Python class ``tf.keras.Model``. In the inheritance class, we need to rewrite the ``__init__()`` (constructor) and ``call(input)`` (model call) methods, but we can also add custom methods as needed.

.. code-block:: python

    class MyModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            # Add initialization code here, including the layers that will be used in call(). e.g., 
            # layer1 = tf.keras.layers.BuiltInLayer(...)
            # layer2 = MyCustomLayer(...)

        def call(self, input):
            # Add the code for the model call here (process the input and return the output). e.g.,
            # x = layer1(input)
            # output = layer2(x)
            return output

        # add your custom methods here

.. figure:: /_static/image/model/model.png
    :width: 50%
    :align: center

    Keras model class structure

After inheriting ``tf.keras.Model``, we can use several methods and properties of the parent class at the same time. For example, after instantiating the class ``model = Model()``, we can get all the variables in the model directly through the property ``model.variables``, saving us from the trouble of specifying them one by one explicitly.

Then, we can rewrite the simple linear model in the previous chapter ``y_pred = a * X + b`` with Keras model class as follows

.. literalinclude:: /_static/code/zh/model/linear/linear.py

Here, instead of explicitly declaring two variables ``a`` and ``b`` and writing the linear transformation ``y_pred = a * X + b``, we create a model class ``Linear`` that inherits ``tf.keras.Model``. This class instantiates a **fully connected layer** (``tf.keras.layers.Dense```) in the constructor, and calls this layer in the call method, implementing the calculation of the linear transformation. If you need to explicitly declare your own variables and use them for custom operations, or want to understand the inner workings of the Keras layer, see :ref:`Custom Layer <custom_layer>`.

.. admonition:: Fully connection layer in Keras: linear transformation + activation function

    `Fully-connected Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`_ (``tf.keras.layers.Dense``) is one of the most basic and commonly used layers in Keras, which performs a linear transformation and activation :math:`f(AW + b)` on the input matrix :math:`A`. If the activation function is not specified, it is a purely linear transformation :math:`AW + b`. Specifically, for a given input tensor ``input = [match_size, input_dim]`` , the layer first performs a linear transformation on the input tensor ``tf.matmul(input, kernel) + bias`` (``kernel`` and ``bias`` are trainable variables in the layer), and then apply the activation function ``activation`` on each element of the linearly transformed tensor, thereby outputting a two-dimensional tensor with shape ``[match_size, units]``.

    .. figure:: /_static/image/model/dense.png
        :width: 60%
        :align: center

    ``tf.keras.layers.Dense`` contains the following main parameters.

    * ``units``: the dimension of the output tensor.
    * ``activation``: the activation function, corresponding to :math:`f` in :math:`f(AW + b)` (Default: no activation). Commonly used activation functions include ``tf.nn.relu``, ``tf.nn.tanh`` and ``tf.nn.sigmoid``.
    * ``use_bias``: whether to add the bias vector ``bias``, i.e. :math:`b` in :math:`f(AW + b)` (Default: ``True``).
    * ``kernel_initializer``, ``bias_initializer``: initializer of the two variables, the weight matrix ``kernel`` and the bias vector ``bias``. The default is ``tf.glorot_uniform_initializer`` [#glorot]_. Set them to ``tf.zeros_initializer`` means that both variables are initialized to zero tensors.

    This layer contains two trainable variables, the weight matrix ``kernel = [input_dim, units]`` and the bias vector ``bias = [bits]`` [#broadcast]_ , corresponding to :math:`W` and :math:`b` in :math:`f(AW + b)`.

    The fully connected layer is described here with emphasis on mathematical matrix operations. A description of neuron-based modeling can be found :ref:`here <en_neuron>`.

    .. [#glorot] Many layers in Keras use ``tf.glorot_uniform_initializer`` by default to initialize variables, which can be found at https://www.tensorflow.org/api_docs/python/tf/glorot_uniform_initializer.
    .. [#broadcast] You may notice that ``tf.matmul(input, kernel)`` results in a two-dimensional matrix with shape ``[batch_size, units]``. How is this two-dimensional matrix to be added to the one-dimensional bias vector ``bias`` with shape ``[units]``? In fact, here is TensorFlow's Broadcasting mechanism at work. The add operation is equivalent to adding ``bias`` to each row of the two-dimensional matrix. A detailed description of the Broadcasting mechanism can be found at https://www.tensorflow.org/xla/broadcasting.

.. admonition:: Why is the model class override ``call()`` instead of ``__call__()``?

    In Python, a call to an instance of a class ``myClass`` (i.e., ``myClass(params)``) is equivalent to ``myClass.__call__(params)`` (see the ``__call__()`` part of "Prerequisite" at the beginning of this chapter). Then in order to call the model using ``y_pred = model(X)``, it seems that one should override the ``__call__()`` method instead of ``call()``. Why we do the opposite? The reason is that Keras still needs to have some pre-processing and post-processing for the model call, so it is more reasonable to expose a ``call()`` method specifically for overriding. The parent class ``tf.keras.Model`` already contains the definition of ``__call__()``. The ``call()`` method is invoked in ``__call__()`` while some internal operations of the keras are also performed. Therefore, by inheriting the ``tf.keras.Model`` and overriding the ``call()`` method, we can add the code of model call while maintaining the inner structure of Keras.

.. _en_mlp:

Basic example: multi-layer perceptron (MLP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the simplest `multilayer perceptron <https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E5%99%A8>`_ (MLP), or "multilayer fully connected neural network" as an example to introduce the model building process in TensorFlow 2. In this section, we take the following steps

- Acquisition and pre-processing of datasets using ``tf.keras.datasets``
- Model construction using ``tf.keras.Model`` and ``tf.keras.layers``
- Build model training process. Use ``tf.keras.loses`` to calculate loss functions and use ``tf.keras.optimizer`` to optimize models
- Build model evaluation process. Use ``tf.keras.metrics`` to calculate assessment indicators (e.g., accuracy)

.. admonition:: Basic knowledges and principles

    * The `Multi-Layer Neural Network <http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/>`_ section of the UFLDL tutorial.
    * "Neural Networks Part 1 ~ 3" section of the Stanford course `CS231n: Convolutional Neural Networks for Visual Recognition <http://cs231n.github.io/>`_.

Here, we use a multilayer perceptron to tackle the classification task on the MNIST handwritten digit dataset [LeCun1998]_.

.. figure:: /_static/image/model/mnist_0-9.png
    :align: center

    examples of MNIST handwritten digit

Data acquisition and pre-processing with ``tf.keras.datasets``
----------------------------------------------------------

To prepare the data, we first implement a simple ``MNISTLoader`` class to read data from the MNIST dataset. ``tf.keras.datasets`` are used here to simplify the download and loading process of MNIST dataset.

.. literalinclude:: /_static/code/zh/model/utils.py
    :lines: 5-19

.. admonition:: Hint 
    
    ``mnist = tf.keras.datasets.mnist`` will automatically download and load the MNIST data set from the Internet. If a network connection error occurs at runtime, you can download the MNIST dataset ``mnist.npz`` manually from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz or https://s3.amazonaws.com/img-datasets/mnist.npz ,and move it into the ``.keras/dataset`` directory of the user directory (``C:\Users\USERNAME`` for Windows and ``/home/USERNAME`` for Linux).

.. admonition:: Image data representation in TensorFlow

    In TensorFlow, a typical representation of an image data set is a four-dimensional tensor of ``[number of images, width, height, number of color channels]``. In the ``DataLoader`` class above, ``self.train_data`` and ``self.test_data`` were loaded with 60,000 and 10,000 handwritten digit images of size ``28*28``, respectively. Since we are reading a grayscale image here with only one color channel (a regular RGB color image has 3 color channels), we use the ``np.expand_dims()`` function to manually add one dimensional channels at the last dimension for the image data.

.. _en_mlp_model:

Model construction with ``tf.keras.Model`` and ``tf.keras.layers``
-------------------------------------------------------------------------------

The implementation of the multi-layer perceptron is similar to the linear model above, constructed using ``tf.keras.Model`` and ``tf.keras.layers``, except that the number of layers is increased (as the name implies, "multi-layer" perceptron), and a non-linear activation function is introduced (here we use the `ReLU function <https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E6%95%B4%E6%B5%81%E5%87%BD%E6%95%B0>`_ activation function, i.e. ``activation=tf.nn.relu`` below). The model accepts a vector (e.g. here a flattened ``1×784`` handwritten digit image) as input and outputs a 10-dimensional vector representing the probability that this image belongs to 0 to 9 respectively.

.. literalinclude:: /_static/code/zh/model/mnist/mlp.py
    :lines: 4-

.. admonition:: Softmax function

    Here, because we want to output the probabilities that the input images belongs to 0 to 9 respectively, i.e. a 10-dimensional discrete probability distribution, we want this 10-dimensional vector to satisfy at least two conditions.

    * Each element in the vector is between :math:`[0, 1]`.
    * The sum of all elements of the vector is 1.

    To ensure the output of the model to always satisfy both conditions, we normalize the raw output of the model using the `Softmax function <https://zh.wikipedia.org/wiki/Softmax%E5%87%BD%E6%95%B0>`_ (normalized exponential function, ``tf.nn.softmax``). Its mathematical form is :math:`\sigma(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}` . Not only that, the softmax function is able to highlight the largest value in the original vector and suppress other components that are far below the maximum, which is why it is called the softmax function (that is, the smoothed argmax function).

.. figure:: /_static/image/model/mlp.png
    :width: 80%
    :align: center

    MLP model

Model training with ``tf.keras.losses`` and ``tf.keras.optimizer``
-------------------------------------------------------------------------------

To train the model, first we define some hyperparameters of the model used in training process

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 8-10

Then, we instantiate the model and data reading classes, and instantiate an optimizer in ``tf.keras.optimizer`` (the Adam optimizer is used here).

.. code-block:: python

    model = MLP()
    data_loader = MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

The following steps are then iterated.

- A random batch of training data is taken from the DataLoader.
- Feed the data into the model, and obtain the predicted value from the model.
- Calculate the loss function ( ``loss`` ) by comparing the model predicted value with the true value. Here we use the cross-entropy function in ``tf.keras.losses`` as a loss function.
- Calculate the derivative of the loss function on the model variables (gradients).
- The derivative values (gradients) are passed into the optimizer, and use the ``apply_gradients`` method to update the model variables so that the loss value is minimized (see :ref:`previous chapter <en_optimizer>` for details on how to use the optimizer).

The code is as follows

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 93-102

.. admonition:: Cross entropy and ``tf.keras.losses``

    You may notice that, instead of explicitly writing a loss function, we use the ``sparse_categorical_crossentropy`` (cross entropy) function in ``tf.keras.losses``. We pass the model predicted value ``y_pred`` and the real value ``y_true`` into the function as parameters, then this Keras function helps us calculate the loss value.

    Cross-entropy is widely used as a loss function in classification problems. The discrete form is :math:`H(y, \hat{y}) = -\sum_{i=1}^{n}y_i \log(\hat{y_i})`, where :math:`y` is the true probability distribution, :math:`\hat{y}` is the predicted probability distribution, and :math:`n` is the number of categories in the classification task. The closer the predicted probability distribution is to the true distribution, the smaller the value of the cross-entropy, and vice versa. A more specific introduction and its application to machine learning can be found in `this blog post <https://blog.csdn.net/tsyccnh/article/details/79163834>`_.

    In ``tf.keras``, there are two cross-entropy related loss functions ``tf.keras.losses.categorical_crossentropy`` and ``tf.keras.losses.sparse_categorical_crossentropy``. Here "sparse" means that the true label value ``y_true`` can be passed directly into the function as integer. That means,

    .. code-block:: python

        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)

    is equivalent to

    .. code-block:: python

        loss = tf.keras.losses.categorical_crossentropy(
            y_true=tf.one_hot(y, depth=tf.shape(y_pred)[-1]),
            y_pred=y_pred
        )

Model Evaluation with ``tf.keras.metrics``
-------------------------------------------------------------------------------

Finally, we use the test set to evaluate the performance of the model. Here, we use the ``SparseCategoricalAccuracy`` metric in ``tf.keras.metrics`` to evaluate the performance of the model on the test set, which compares the results predicted by the model with the true results, and outputs the proportion of the test data samples that is correctly classified by the model. We do evaluatio iteratively on the test set, feeding the results predicted by the model and the true results into the metric instance each time by the ``update_state()`` method, with two parameters ``y_pred`` and ``y_true`` respectively. The metric instance has internal variables to maintain the values associated with the current evaluation process (e.g., the current cumulative number of samples that has been passed in and the current number of samples that predicts correctly). At the end of the iteration, we use the ``result()`` method to output the final evaluation value (the proportion of the correctly classified samples over the total samples).

In the following code, we instantiate a ``tf.keras.metrics.SparseCategoricalAccuracy`` metric, use a for loop to feed the predicted and true results iteratively, and output the accuracy of the trained model on the test set.

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 104-110

Output::

    test accuracy: 0.947900

It can be noted that we can reach an accuracy rate of around 95% just using such a simple model.

.. _en_neuron:

.. admonition:: The basic unit of a neural network: the neuron [#order]_

    If we take a closer look at the neural network above and study the computational process in detail, for example by taking the k-th computational unit of the second layer, we can get the following schematic

    .. figure:: /_static/image/model/neuron.png
        :width: 80%
        :align: center

    The computational unit :math:`Q_k` has 100 weight parameters :math:`w_{0k}, w_{1k}, \cdots , w_{99k}` and 1 bias parameter :math:`b_k` . The values of :math:`P_0, P_1, \cdots , P_{99}` of all 100 computational units in layer 1 are taken as inputs, summed by weight :math:`w_{ik}` (i.e. :math:`\sum_{i=0}^{99} w_{ik} P_i` ) and biased by :math:`b_k` , then it is fed into the activation function :math:`f` to get the output result.

    In fact, this structure is quite similar to real nerve cells (neurons). Neurons are composed of dendrites, cytosomes and axons. Dendrites receive signals from other neurons as input (one neuron can have thousands or even tens of thousands of dendrites), the cell body integrates the potential signal, and the resulting signal travels through axons to synapses at nerve endings and propagates to the next (or more) neuron.

    .. figure:: /_static/image/model/real_neuron.png
        :width: 80%
        :align: center

        Neural cell pattern diagram (modified from Quasar Jarosz at English Wikipedia [CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0)])

    The computational unit above can be viewed as a mathematical modeling of neuronal structure. In the above example, each computational unit (artificial neuron) in the second layer has 100 weight parameters and 1 bias parameter, while the number of computational units in the second layer is 10, so the total number of participants in this fully connected layer is 100*10 weight parameters and 10 bias parameters. In fact, this is the shape of the two variables ``kernel`` and ``bias`` in this fully connected layer. Upon closer examination, you will see that the introduction to neuron-based modeling here is equivalent to the introduction to matrix-based computing above.

    .. [#order] Actually, there should be the concept of neuronal modeling first, followed by artificial neural networks based on artificial neurons and layer structures. However, since this manual focuses on how to use TensorFlow, the order of introduction is switched.

Convolutional Neural Network (CNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Convolutional Neural Network <https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C>`_ (CNN) is an artificial neural network with a structure similar to the `visual system <https://zh.wikipedia.org/wiki/%E8%A7%86%E8%A7%89%E7%B3%BB%E7%BB%9F>`_ of a human or animal, that contains one or more Convolutional Layer, Pooling Layer and Fully-connected Layer.

.. admonition:: Basic knowledges and principles

    * `Convolutional Neural Network <http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/>`_ in UFLDL Tutorial
    * "Module 2: Convolutional Neural Networks" in Stanford course `CS231n: Convolutional Neural Networks for Visual Recognition <http://cs231n.github.io/>`_
    * `"Convolutional Neural Networks" <https://d2l.ai/chapter_convolutional-neural-networks/index.html>`_ in *Dive into Deep Learning*

Implementing Convolutional Neural Networks with Keras
-------------------------------------------------------

An example implementation of a convolutional neural network is shown below. The code structure is similar to the :ref:`multi-layer perceptron <en_mlp_model>` in the previous section, except that some new convolutional and pooling layers are added. The network structure here is not unique: the layers in the CNN can be added, removed or adjusted for better performance.

.. literalinclude:: /_static/code/zh/model/mnist/cnn.py
    :lines: 4-

.. figure:: /_static/image/model/cnn.png
    :align: center

    CNN structure diagram in the above sample code

Replace the code line ``model = MLP()`` in previous MLP section to ``model = CNN()`` , the output will be as follows::

    test accuracy: 0.988100

A very significant improvement of accuracy can be found compared to MLP in the previous section. In fact, there is still room for further improvements by changing the network structure of the model (e.g. by adding a Dropout layer to prevent overfitting).

Using predefined classical CNN structures in Keras
---------------------------------------------------------------------------

There are some pre-defined classical convolutional neural network structures in ``tf.keras.applications``, such as ``VGG16``, ``VGG19``, ``ResNet`` and ``MobileNet``. We can directly apply these classical convolutional neural network (and load pre-trained weights) without manually defining the CNN structure.

For example, we can use the following code to instantiate a ``MobileNetV2`` network structure.

.. code-block:: python

    model = tf.keras.applications.MobileNetV2()

When the above code is executed, TensorFlow will automatically download the pre-trained weights of the ``MobileNetV2`` network, so Internet connection is required for the first execution of the code. You can also initialize variables randomly by setting the parameter ``weights`` to ``None``. Each network structure has its own specific detailed parameter settings. Some shared common parameters are as follows.

- ``input_shape``: the shape of the input tensor (without the first batch dimension), which mostly defaults to ``224 × 224 × 3``. In general, models have lower bounds on the size of the input tensor, with a minimum length and width of ``32 × 32`` or ``75 × 75``.
- ``include_top``: whether the fully-connected layer is included at the end of the network, which defaults to ``True``.
- ``weights``: pre-trained weights, which default to ``imagenet`` (using pre-trained weights trained on ImageNet dataset). It can be set to ``None`` if you want to randomly initialize the variables.
- ``classes``: the number of classes, which defaults to 1000. If you want to modify this parameter, the ``include_top`` parameter has to be ``True`` and the ``weights`` parameter has to be ``None``.

A detailed description of each network model parameter can be found in the `Keras documentation <https://keras.io/applications/>`_.

.. admonition:: Set learning phase

    For some pre-defined classical models, some of the layers (e.g. ``BatchNormalization``) behave differently on training and testing stage (see `this article <https://zhuanlan.zhihu.com/p/64310188>`_). Therefore, when training this kind of model, you need to set the learning phase manually, telling the model "I am in the training stage of the model". This can be done through

    .. code-block:: python

        tf.keras.backend.set_learning_phase(True)

    or by setting the ``training`` parameter to ``True`` when the model is called.

An example is shown below, using ``MobileNetV2`` network to train on ``tf_flowers`` five-classifying datasets (for the sake of code brevity and efficiency, we use :doc:`TensorFlow Datasets <../appendix/tfds>` and :ref:`tf.data <en_tfdata>` to load and preprocess the data in this example). Also we set ``classes`` to 5, corresponding to the ``tf_flowers`` dataset with 5 kind of labels.

.. literalinclude:: /_static/code/zh/model/cnn/mobilenet.py
    :emphasize-lines: 10, 15  

In later sections (e.g. :doc:`Distributed Training <../appendix/distributed>`), we will also directly use these classicial network structures for training.

.. admonition:: How the Convolutional and Pooling Layers Work

    The Convolutional Layer, represented by ``tf.keras.layers.Conv2D`` in Keras, is a core component of CNN and has a structure similar to the visual cortex of the brain.

    Recall our previously established computational model of :ref:`neurons <en_neuron>` and the fully-connected layer, in which we let each neuron connect to all other neurons in the previous layer. However, this is not the case in the visual cortex. You may have learned in biology class about the concept of **Receptive Field**, where neurons in the visual cortex are not connected to all the neurons in the previous layer, but only sense visual signals in an area and respond only to visual stimuli in the local area.

    For example, the following figure is a 7×7 single-channel image signal input.

    .. figure:: /_static/image/model/conv_image.png
        :align: center

    If we use the MLP model based on fully-connected layers, we need to make each input signal correspond to a weight value. In this case, modeling a neuron requires 7×7=49 weights (50 if we consider the bias) to get an output signal. If there are N neurons in a layer, we need 49N weights and get N output signals.

    In the convolutional layer of CNN, we model a neuron in a convolutional layer like this.

    .. figure:: /_static/image/model/conv_field.png
        :align: center

    The 3×3 red box in the figure represents the receptor field of this neuron. In this case, we only need a 3×3 weight matrix :math:`W = \begin{bmatrix}w_{1, 1} & w_{1, 2} & w_{1, 3} \\w_{2, 1} & w_{2, 2} & w_{2, 3} \\w_{3, 1} & w_{3, 2} & w_{3, 3}\end{bmatrix}`  with an additional bias :math:`b`  to get an output signal. E.g., for the red box shown in the figure, the output is the sum of all elements of matrix :math:`\begin{bmatrix}0 \times w_{1, 1} & 0 \times w_{1, 2} & 0 \times w_{1, 3} \\0 \times w_{2, 1} & 1 \times w_{2, 2} & 0 \times w_{2, 3} \\0 \times w_{3, 1} & 0 \times w_{3, 2} & 2 \times w_{3, 3}\end{bmatrix}` adding bias :math:`b`, noted as :math:`a_{1, 1}` .

    However, the 3×3 range is clearly not enough to handle the entire image, so we use the sliding window approach. Use the same parameter :math:`W` but swipe the red box from left to right in the image, scanning it line by line, calculating a value for each position it slides to. For example, when the red box moves one unit to the right, we calculate the sum of all elements of the matrix :math:`\begin{bmatrix}0 \times w_{1, 1} & 0 \times w_{1, 2} & 0 \times w_{1, 3} \\1 \times w_{2, 1} & 0 \times w_{2, 2} & 1 \times w_{2, 3} \\0 \times w_{3, 1} & 2 \times w_{3, 2} & 1 \times w_{3, 3}\end{bmatrix}` , adding bias :math:`b`, noted as :math:`a_{1, 2}` . Thus, unlike normal neurons that can only output one value, the convolutional neurons here can output a 5×5 matrix :math:`A = \begin{bmatrix}a_{1, 1} & \cdots & a_{1, 5} \\ \vdots & & \vdots \\ a_{5, 1} & \cdots & a_{5, 5}\end{bmatrix}` .

    .. figure:: /_static/image/model/conv_procedure.png
        :align: center

        Diagram of convolution process. A single channel 7×7 image passes through a convolutional layer with a receptor field of 3×3, yielded a 5×5 matrix as result.

    In the following part, we use TensorFlow to verify the results of the above calculation.

    The input image, the weight matrix :math:`W` and the bias term :math:`b` in the above figure are represented as the NumPy array ``image``, ``W``, ``b`` as follows.

    .. literalinclude:: /_static/code/zh/model/cnn/cnn_example.py
        :lines: 4-21

    Then, we build a model with only one convolutional layer, initialized by ``W`` and ``b`` [#sequential]_ ：

    .. literalinclude:: /_static/code/zh/model/cnn/cnn_example.py
        :lines: 23-30

    Finally, feed the image data ``image`` into the model and print the output.

    .. literalinclude:: /_static/code/zh/model/cnn/cnn_example.py
        :lines: 32-33

    The result will be

    ::

        tf.Tensor(
        [[ 6.  5. -2.  1.  2.]
         [ 3.  0.  3.  2. -2.]
         [ 4.  2. -1.  0.  0.]
         [ 2.  1.  2. -1. -3.]
         [ 1.  1.  1.  3.  1.]], shape=(5, 5), dtype=float32)

    You can find out that this result is consistent with the value of the matrix :math:`A` in the figure above.
    
    One more question, the above convolution process assumes that the images only have one channel (e.g. grayscale images), but what if the image is in color (e.g. has three channels of RGB)? Actually, we can prepare a 3×3 weight matrix for each channel, i.e. there are 3×3×3=27 weights in total. Each channel is processed using its own weight matrix, and the output can be summed by adding the values from multiple channels.

    Some readers may notice that, following the method described above, the result after each convolution will be "one pixel shrinked" around. The 7×7 image above, for example, becomes 5×5 after convolution, which sometimes causes problems to the forthcoming layers. Therefore, we can set the padding strategy. In ``tf.keras.layers.Conv2D``, when we set the ``padding`` parameter to ``same``, the missing pixels around it are filled with 0, so that the size of the output matches the input.

    Finally, since we can use the sliding window method to do convolution, can we set a different step size for the slide? The answer is yes. The step size (default is 1) can be set using the ``strides`` parameter of ``tf.keras.layers.Conv2D``. For example, in the above example, if we set the step length to 2, the output will be a 3×3 matrix.

    In fact, there are many forms of convolution, and the above introduction is only one of the simplest one. Further examples of the convolutional approach can be found in `Convolution Arithmetic <https://github.com/vdumoulin/conv_arithmetic>`_.

    The Pooling Layer is much simpler to understand as the process of downsampling an image, outputting the maximum value (MaxPooling), the mean value, or the value generated by other methods for all values in the window for each slide. For example, for a three-channel 16×16 image (i.e., a tensor of ``16*16*3``), a tensor of ``8*8*3`` is obtained after a pooled layer with a receptive field of 2×2 and a slide step of 2.

    .. [#sequential] Here we use the sequential mode to build the model for simplicity, as described :ref:`later <en_sequential_functional>` .

Recurrent Neural Network (RNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recurrent Neural Network (RNN) is a type of neural network suitable for processing sequence data (especially text). It is widely used in language models, text generation and machine translation.

.. admonition:: Basic knowledges and principles

    - `Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs <http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/>`_
    - `Understanding LSTM Networks <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_
    - `"Recurrent Neural Networks" <https://d2l.ai/chapter_recurrent-neural-networks/index.html>`_ in *Dive into Deep Learning*。
    - RNN sequence generation: [Graves2013]_

Here, we use RNN to generate Nietzschean-style text automatically. [#rnn_reference]_

The essence of this task is to predict the probability distribution of an English sentence's successive character. For example, we have the following sentence::

    I am a studen

This sentence (sequence) has a total of 13 characters including spaces. When we read this sequence of 13 characters, we can predict based on our experience, that the next character is "t" with a high probability. Now we want to build a model to do the same thing as our experience, in which we input a sequence of ``seq_length`` one by one, and output the probability distribution of the next character that follows this sentence. Then we can generate text by sampling a character from the probability distribution as a predictive value, then do snowballing to generate the next two characters, the next three characters, etc.

First of all, we implement a simple ``DataLoader`` class to read training corpus (Nietzsche's work) and encode it in characters. Each character is assigned a unique integer number i between 0 and ``num_chars - 1``, in which ``num_chars`` is the number of character types.

.. literalinclude:: /_static/code/zh/model/text_generation/rnn.py
    :lines: 35-53

The model implementation is carried out next. In the constructor (``__init__`` method), we instantiate a ``LSTMCell`` unit and a fully connected layer. in ``call`` method, We first perform a "One Hot" operation on the sequence, i.e., we transform the encoding i of each character in the sequence into a ``num_char`` dimensional vector with bit i being 1 and the rest being 0. The transformed sequence tensor has a shape of ``[seq_length, num_chars]`` . We then initialize the state of the RNN unit. Next, the characters of the sequence is fed into the RNN unit one by one. At moment t, the state of RNN unit ``state`` in the previous time step ``t-1`` and the t-th element of the sequence ``inputs[t, :]`` are fed into the RNN unit, to get the output ``output`` and the RNN unit state in the current time step ``t``. The last output of the RNN unit is taken and transformed through the fully connected layer to ``num_chars`` dimension.

.. figure:: /_static/image/model/rnn_single.jpg
    :width: 50%
    :align: center

    Diagram of ``output, state = self.cell(inputs[:, t, :], state)``

.. figure:: /_static/image/model/rnn.jpg
    :width: 100%
    :align: center

    RNN working process

The code implementation is like this

.. literalinclude:: /_static/code/zh/model/text_generation/rnn.py
    :lines: 7-25

Defining some hyperparameters of the model

.. literalinclude:: /_static/code/zh/model/text_generation/rnn.py
    :lines: 57-60

The training process is very similar to the previous section. Here we just repeat it:

- A random batch of training data is taken from the DataLoader.
- Feed the data into the model, and obtain the predicted value from the model.
- Calculate the loss function ( ``loss`` ) by comparing the model predicted value with the true value. Here we use the cross-entropy function in ``tf.keras.losses`` as a loss function.
- Calculate the derivative of the loss function on the model variables (gradients).
- The derivative values (gradients) are passed into the optimizer, and use the ``apply_gradients`` method to update the model variables so that the loss value is minimized.

.. literalinclude:: /_static/code/zh/model/text_generation/rnn.py
    :lines: 62-73

One thing about the process of text generation requires special attention. Previously, we have been using the ``tf.argmax()`` function, which takes the value corresponding to the maximum probability as the predicted value. For text generation, however, such predictions are too "absolute" and can make the generated text lose its richness. Thus, we use the ``np.random.choice()`` function to sample the resulting probability distribution. In this way, even characters that correspond to a small probability have a chance of being sampled. At the same time, we add a ``temperature`` parameter to control the shape of the distribution, the larger the parameter value, the smoother the distribution (the smaller the difference between the maximum and minimum values), the higher the richness of the generated text; the smaller the parameter value, the steeper the distribution, the lower the richness of the generated text.

.. literalinclude:: /_static/code/zh/model/text_generation/rnn.py
    :lines: 27-32

Through a contineous prediction of characters, we can get the automatically generated text.

.. literalinclude:: /_static/code/zh/model/text_generation/rnn.py
    :lines: 75-83

The generated text is like follows::

    diversity 0.200000:
    conserted and conseive to the conterned to it is a self--and seast and the selfes as a seast the expecience and and and the self--and the sered is a the enderself and the sersed and as a the concertion of the series of the self in the self--and the serse and and the seried enes and seast and the sense and the eadure to the self and the present and as a to the self--and the seligious and the enders

    diversity 0.500000:
    can is reast to as a seligut and the complesed
    has fool which the self as it is a the beasing and us immery and seese for entoured underself of the seless and the sired a mears and everyther to out every sone thes and reapres and seralise as a streed liees of the serse to pease the cersess of the selung the elie one of the were as we and man one were perser has persines and conceity of all self-el

    diversity 1.000000:
    entoles by
    their lisevers de weltaale, arh pesylmered, and so jejurted count have foursies as is
    descinty iamo; to semplization refold, we dancey or theicks-welf--atolitious on his
    such which
    here
    oth idey of pire master, ie gerw their endwit in ids, is an trees constenved mase commars is leed mad decemshime to the mor the elige. the fedies (byun their ope wopperfitious--antile and the it as the f

    diversity 1.200000:
    cain, elvotidue, madehoublesily
    inselfy!--ie the rads incults of to prusely le]enfes patuateded:.--a coud--theiritibaior "nrallysengleswout peessparify oonsgoscess teemind thenry ansken suprerial mus, cigitioum: 4reas. whouph: who
    eved
    arn inneves to sya" natorne. hag open reals whicame oderedte,[fingo is
    zisternethta simalfule dereeg hesls lang-lyes thas quiin turjentimy; periaspedey tomm--whach

.. [#rnn_reference] Here we referenced https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

.. admonition:: The working process of recurrent neural networks

    Recurrent neural network is a kind of neural network designed to process time series data. To understand the working process of RNN, we need to have a timeline in our mind. The RNN unit has an initial state :math:`s_0` at initial time step 0, then at each time step :math:`t`, the RNN unit process the current input :math:`x_t`, modifies its own state :math:`s_t` , and outputs :math:`o_t` .

    The core of RNN is the state :math:`s` , which is a vector of fixed dimensions, regarded as the "memory" of RNN. At the initial moment of :math:`t=0`, :math:`s_0` is given an initial value (usually a zero vector). We then describe the working process of RNN in a recursive way. That is, at the moment :math:`t`, we assume that :math:`s_{t-1}` is known, and focus on how to calculate :math:`s_{t}` based on the input and the previous state.

    - Linear transformation of the input vector :math:`x_t` through the matrix :math:`U`. The result :math:`U x_t` has the same dimension as the state s.
    - Linear transformation of :math:`s_{t-1}` through the matrix :math:`W`. The result :math:`W s_{t-1}` has the same dimension as the state s.
    - The two vectors obtained above are summed and passed through the activation function as the value of the current state :math:`s_t`, i.e. :math:`s_t = f(U x_t + W s_{t-1})`. That is, the value of the current state is the result of non-linear information combination of the previous state and the current input.
    - Linear transformation of the current state :math:`s_t` through the matrix :math:`V` to get the output of the current moment :math:`o_t`.

    .. figure:: /_static/image/model/rnn_cell.jpg
        :align: center

        RNN working process (from http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns )

    We assume the dimension of the input vector :math:`x_t`, the state :math:`s` and the output vector :math:`o_t` are :math:`m`, :math:`n` and :math:`p` respectively, then :math:`U \in \mathbb{R}^{m \times n}`, :math:`W \in \mathbb{R}^{n \times n}`, :math:`V \in \mathbb{R}^{n \times p}`.

    The above is an introduction to the most basic RNN type. In practice, some improved version of RNN are often used, such as LSTM (Long Short-Term Memory Neural Network, which solves the problem of gradient disappearance for longer sequences), GRU, etc.

.. _en_rl:

Deep Reinforcement Learning (DRL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Reinforcement learning <https://zh.wikipedia.org/wiki/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0>`_ (RL) emphasizes how to act based on the environment in order to maximize the intended benefits. With deep learning techniques combined, Deep Reinforcement Learning (DRL) is a powerful tool to solve decision tasks. AlphaGo, which has become widely known in recent years, is a typical application of deep reinforcement learning.

.. admonition:: Note

    You may want to read :doc:`../appendix/rl` in the appendix to get some basic ideas of reinforcement learning.

Here, we use deep reinforcement learning to learn to play CartPole (inverted pendulum). The inverted pendulum is a classic problem in cybernetics. In this task, the bottom of a pole is connected to a cart through an axle, and the pole's center of gravity is above the axle, making it an unstable system. Under the force of gravity, the pole falls down easily. We need to control the cart to move left and right on a horizontal track to keep the pole in vertical balance.

.. only:: html

    .. figure:: /_static/image/model/cartpole.gif
        :width: 500
        :align: center

        CartPole Game

.. only:: latex

    .. figure:: /_static/image/model/cartpole.png
        :width: 500
        :align: center

        CartPole Game

We use the CartPole game environment from `OpenAI's Gym Environment Library <https://gym.openai.com/>`_, which can be installed using ``pip install gym``, the installation steps and tutorials can be found in the `official documentation <https://gym.openai.com/docs/>`_ and `here <https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-4-gym/>`_. The interaction with Gym is very much like a turn-based game. We first get the initial state of the game (such as the initial angle of the pole and the position of the cart), then in each turn t, we need to choose one of the currently feasible actions and send it to Gym to execute (such as pushing the cart to the left or to the right, only one of the two actions can be chosen in each turn). After executing the action, Gym will return the next state after the action is executed and the reward value obtained in the current turn (for example, after we choose to push the cart to the left and execute, the cart position is more to the left and the angle of the pole is more to the right, Gym will return the new angle and position to us. if the pole still doesn't go down on this round, Gym returns us a small positive reward simultaneously). This process can iterate on and on until the game ends (e.g. the pole goes down). In Python, the sample code to use Gym is as follows.

.. code-block:: python

    import gym

    env = gym.make('CartPole-v1')   # Instantiate a game environment with the game name
    state = env.reset()             # Initialize the environment, get the initial state
    while True:
        env.render()                # Render the current frame and draw it to the screen.
        action = model.predict(state)   # Suppose we have a trained model that can predict what action should be performed at this time from the current state
        next_state, reward, done, info = env.step(action)   # Let the environment execute the action, get the next state of the executed action, the reward for the action, whether the game is over and additional information
        if done:                    # Exit loop if game over
            break

Now, our task is to train a model that can predict a good move based on the current state. Roughly speaking, a good move should maximize the sum of the rewards earned throughout the game, which is the goal of reinforcement learning. In the CartPole game, the goal is to make the right moves to keep the pole from falling, i.e. as many rounds of game interaction as possible. In each round, we get a small positive bonus, and the more rounds the higher the cumulative bonus value. Thus, maximizing the sum of the rewards is consistent with our ultimate goal.

The following code shows how to train the model using the Deep Q-Learning method [Mnih2013]_ , a classical Deep Reinforcement Learning algorithm. First, we import TensorFlow, Gym and some common libraries, and define some model hyperparameters.

.. literalinclude:: /_static/code/zh/model/rl/qlearning.py
    :lines: 1-14

We then use ``tf.keras.Model``` to build a Q-network for fitting the Q functions in Q-Learning algorithm. Here we use a simple multilayered fully connected neural network for fitting. The network use the current state as input and outputs the Q-value for each action (2-dimensional for CartPole, i.e. pushing the cart left and right).

.. literalinclude:: /_static/code/zh/model/rl/qlearning.py
    :lines: 16-31

Finally, we implement the Q-learning algorithm in the main program.

.. literalinclude:: /_static/code/zh/model/rl/qlearning.py
    :lines: 34-82

For different tasks (or environments), we need to design different states and adopt appropriate networks to fit the Q function depending on the characteristics of the task. For example, if we consider the classic "Block Breaker" game (`Breakout-v0 <https://gym.openai.com/envs/Breakout-v0/>`_ in the Gym environment library), every action performed (baffle moving to the left, right, or motionless) returns an RGB image of ``210 * 160 * 3`` representing the current screen. In order to design a suitable state representation for this game, we have the following analysis.

* The colour information of the bricks is not very important and the conversion of the image to grayscale does not affect the operation, so the colour information in the state can be removed (i.e. the image can be converted to grayscale).
* Information on the movement of the ball is important. For CartPole, it is difficult for even a human being to judge the direction in which the baffle should move if only a single frame is known (so the direction in which the ball is moving is not known). Therefore, information that characterizes motion direction of the ball should be added to the state. A simple way is to stack the current frame with the previous frames to obtain a state representation of ``210 * 160 * X`` (X being the number of stacked frames).
* The resolution of each frame does not need to be particularly high, as long as the position of the blocks, ball and baffle can be roughly characterized for decision-making purposes, so that the length and width of each frame can be compressed appropriately.

Considering that we need to extract features from the image information, using CNN as a network for fitting Q functions would be more appropriate. Based on the analysis, we can just replace the ``QNetwork`` model class above to a CNN-based model and make some changes for the status, then the same program can be used to play some simple video games like "Block Breaker".

Keras Pipeline *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://medium.com/tensorflow/what-are-symbolic-and-imperative-apis-in-tensorflow-2-0-dfccecb01021
    https://www.tensorflow.org/beta/guide/keras/overview
    https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models

Until now, all the examples are using Keras' Subclassing API and customized training loop. That is, we inherit ``tf.keras.Model`` class to build our new model, while the process of training and evaluating the model is explicitly implemented by us. This approach is flexible and similar to other popular deep learning frameworks (e.g. PyTorch and Chainer), and is the approach recommended in this handbook. In many cases, however, we just need to build a neural network with a relatively simple and typical structure (e.g., MLP and CNN in the above section) and train it using conventional means. For these scenarios, Keras also give us another simpler and more efficient built-in way to build, train and evaluate models.

.. _sequential_functional:

Use Keras Sequential/Functional API to build models
---------------------------------------------------

The most typical and common neural network structure is to stack a bunch of layers in a specific order, so can we just provide a list of layers and have Keras automatically connect them head to tail to form a model? This is exactly what Keras Sequential API does. By providing a list of layers to ``tf.keras.models.Sequential()``, we can quickly construct a ``tf.keras.Model`` model.

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 18-23

However, the sequential network structure is quite limited. Then Keras provides a more powerful functional API to help us build complex models, such as models with multiple inputs/outputs or where parameters are shared. This is done by using the layer as an invocable object and returning the tensor (which is consistent with the usage in the previous section). Then we can build a model by providing the input and output vectors to the ``inputs`` and ``outputs`` parameters of ``tf.keras.Model``, as follows

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 25-30
..
    https://www.tensorflow.org/alpha/guide/keras/functional

Train and evaluate models using the ``compile``, ``fit`` and ``evaluate`` methods of Keras
------------------------------------------------------------------------------------------

When the model has been built, the training process can be configured through the ``compile`` method of ``tf.keras.Model``.

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 84-88

``tf.keras.Model.compile`` accepts three important parameters.

- ``oplimizer``: an optimizer, can be selected from ``tf.keras.optimizers''.
- ``loss``: a loss function, can be selected from ``tf.keras.loses''.
- ``metrics``: a metric, can be selected from ``tf.keras.metrics``.

Then, the ``fit`` method of ``tf.keras.Model`` can be used to actually train the model.

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 89

``tf.keras.model.fit`` accepts five important parameters.

- ``x``: training data.
- ``y``: target data (labels of data).
- ``epochs``: the number of iterations through training data.
- ``batch_size``: the size of the batch.
- ``validation_data``: validation data that can be used to monitor the performance of the model during training.

Keras supports ``tf.data.Dataset`` as data source, detailed in :ref:`tf.data <en_tfdata>`.

Finally, we can use ``tf.keras.Model.evaluate`` to evaluate the trained model, just by providing the test data and labels.

.. literalinclude:: /_static/code/zh/model/mnist/main.py
    :lines: 90

..
    https://www.tensorflow.org/beta/guide/keras/training_and_evaluation

Custom layers, losses and metrics *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Perhaps you will also ask, what if these existing layers do not meet my requirements, and I need to define my own layers? In fact, we can inherit not only ``tf.keras.Model`` to write our own model classes, but also ``tf.keras.layers.Layer`` to write our own layers.

.. _custom_layer:

Custom layers
-------------------------------------------

The custom layer requires inheriting the ``tf.keras.layers.Layer`` class and overriding the ``__init__``, ``build`` and ``call`` methods, as follows.

.. code-block:: python

    class MyLayer(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            # Initialization code

        def build(self, input_shape): # input_shape is a TensorShape object that provides the shape of the input
            # this part of the code will run at the first time you call this layer
            # you can create variables here so that the the shape of the variable is adaptive to the input shape
            # If the shape of the variable can already be fully determined without the infomation of input shape
            # you can also create the variable in the constructor (__init__)
            self.variable_0 = self.add_weight(...)
            self.variable_1 = self.add_weight(...)

        def call(self, inputs):
            # Code for model call (handles inputs and returns outputs)
            return output

For example, we can implement a :ref:`fully-connected layer <en_linear>` on our own with the following code. This code creates two variables in the ``build`` method and uses the created variables in the ``call`` method.

.. literalinclude:: /_static/code/zh/model/custom/linear.py
    :lines: 9-22

When defining a model, we can use our custom layer ``LinearLayer`` just like other pre-defined layers in Keras.

.. literalinclude:: /_static/code/zh/model/custom/linear.py
    :lines: 25-32

Custom loss functions and metrics
-------------------------------------------

..
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/losses/Loss

The custom loss function needs to inherit the ``tf.keras.losses.Loss`` class and override the ``call`` method. The ``call`` method use the real value ``y_true`` and the model predicted value ``y_pred`` as input, and return the customized loss value between the model predicted value and the real value. The following example implements a mean square error loss function.

.. code-block:: python

    class MeanSquaredError(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            return tf.reduce_mean(tf.square(y_pred - y_true))

..
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/metrics/Metric

The custom metrics need to inherit the ``tf.keras.metrics.Metric`` class and override the ``__init__``, ``update_state`` and ``result`` methods. The following example re-implements a simple ``SparseCategoricalAccuracy`` metric class that we used earlier.

.. literalinclude:: /_static/code/zh/model/utils.py
    :lines: 22-34

.. [LeCun1998] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998. http://yann.lecun.com/exdb/mnist/
.. [Graves2013] Graves, Alex. “Generating Sequences With Recurrent Neural Networks.” ArXiv:1308.0850 [Cs], August 4, 2013. http://arxiv.org/abs/1308.0850.
.. [Mnih2013] Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. “Playing Atari with Deep Reinforcement Learning.” ArXiv:1312.5602 [Cs], December 19, 2013. http://arxiv.org/abs/1312.5602.

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 190 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>

