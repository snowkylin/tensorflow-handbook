TensorFlow Models
===================

.. _linear:

This chapter describes how to build a dynamic model with TensorFlow quickly.

Prerequisites:

* `Python OOP <https://www.python-course.eu/python3_object_oriented_programming.php>`_ (definition of classes & methods in Python, class inheritance, construction and destruction functions, `using super() to call methods of the parent class <https://www.python-course.eu/python3_inheritance.php>`_, `using __call__() to call an instance <https://www.python-course.eu/python3_magic_methods.php>`_, etc.)
* Multilayer perceptrons, convolutional neural networks, recurrent neural networks and reinforcement learning (references given before each chapter).

Models & Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
..  https://www.tensorflow.org/programmers_guide/eager

As mentioned in the last chapter, to improve the reusability of codes, we usually implement models as classes and use statements like ``y_pred() = model(X)`` to call the models. The structure of a **model class** is rather simple which basically includes ``__init__()`` (for construction and initialization) and ``call(input)`` (for model invocation) while you can also define your own methods if necessary. [#call]_

.. code-block:: python

    class MyModel(tf.keras.Model):
        def __init__(self):
            super().__init__()     # Use super(MyModel, self).__init__() under Python 2.
            # Add initialization code here (including layers to be used in the "call" method).

        def call(self, inputs):
            # Add model invocation code here (processing inputs and returning outputs).
            return output

Here our model inherits from ``tf.keras.Model``. Keras is an advanced neural network API written in Python and is now supported by and built in official TensorFlow. One benefit of inheriting from ``tf.keras.Model`` is that we will be able to use its several methods and attributes like acquiring all variables in the model through the ``model.variables`` attribute after the class is instantiated, which saves us from indicating variables one by one explicitly.

Meanwhile, we introduce the concept of **layers**. Layers can be regarded as finer components encapsulating the computation procedures and variables compared with models. We can use layers to build up models quickly.

The simple linear model ``y_pred = tf.matmul(X, w) + b`` mentioned in the last chapter can be implemented through model classes:

.. literalinclude:: ../_static/code/en/model/linear/linear.py

Here we didn't explicitly declare two variables ``W`` and ``b`` or write the linear transformation ``y_pred = tf.matmul(X, w) + b``, but instead instantiated a densely-connected layer (``tf.keras.layers.Dense``) in the initialization and called this layer in the "call" method. The densely-connected layer encapsulates the ``output = activation(tf.matmul(input, kernel) + bias)`` linear transformation, the activation function, and two variables ``kernel`` and ``bias``. This densely-connected layer would be equivalent to the aforementioned linear transformation if the activation function is not specified (i.e. ``activation(x) = x``). By the way, the densely-connected layer may be the most frequent in our daily model coding.

Please refer to :ref:`Custom Layers <custom_layer>` if you need to declare variables explicitly and customize operations.

.. [#call] In Python classes, calling instances of the class ``myClass`` by such as ``myClass()`` is equivalent to ``myClass.__call__()``. Here our model inherits from the parent class ``tf.keras.Model``. This class includes the definition of ``__call__()``, calling ``call()`` and also doing some internal keras operations. By inheriting from ``tf.keras.Model`` and overloading its ``call()`` method, we can add the codes for calling while keep the structure of keras, for details please refer to the ``__call__()`` part of the prerequisites in this chapter.

.. _mlp:

A Basic Example: Multilayer Perceptrons (MLP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's begin with implementing a simple `Multilayer Perceptron <https://en.wikipedia.org/wiki/Multilayer_perceptron>`_ as an introduction of writing models in TensorFlow. Here we use the multilayer perceptron to classify the MNIST handwriting digit image dataset.

.. figure:: ../_static/image/model/mnist_0-9.png
    :align: center

    An image example of MNIST handwriting digits.

Before the main course, we implement a simple ``DataLoader`` class for loading the MNIST dataset.

.. literalinclude:: ../_static/code/en/model/mlp/main.py
    :lines: 13-23

The implementation of the multilayer perceptron model class is similar to the aforementioned linear model class except the former has more layers (as its name "multilayer" suggests) and introduces a nonlinear activation function (here `ReLU function <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ is used by the code ``activation=tf.nn.relu`` below). This model receives a vector (like a flattened 1*784 handwriting digit image here) and outputs a 10 dimensional signal representing the probability of this image being 0 to 9, respectively. Here we introduce a "predict" method which predicts the handwriting digits. It chooses the digit with the maximum likelihood as an output.

.. literalinclude:: ../_static/code/en/model/mlp/mlp.py
    :lines: 4-17

Define some hyperparameters for the model:

.. literalinclude:: ../_static/code/en/model/mlp/main.py
    :lines: 8-10

Instantiate the model, the data loader class and the optimizer:

.. code-block:: python

    model = MLP()
    data_loader = DataLoader()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

And iterate the following steps:

- Randomly read a set of training data from DataLoader;
- Feed the data into the model to acquire its predictions;
- Compare the predictions with the correct answers to evaulate the loss function;
- Differentiate the loss function with respect to model parameters;
- Update the model parameters in order to minimize the loss.

The code implementation is as follows:

.. literalinclude:: ../_static/code/en/model/mlp/main.py
    :lines: 32-39

Then, we will use the validation dataset to examine the performance of this model. To be specific, we will compare the predictions with the answers on the validation dateset, and output the ratio of correct predictions:

.. literalinclude:: ../_static/code/en/model/mlp/main.py
    :lines: 41-43

Output::

    test accuracy: 0.947900

Note that we can easily get an accuracy of 95% with even a simple model like this.

Convolutional Neural Networks (CNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Convolutional Neural Network <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_ is an artificial neural network resembled as the animal `visual system <https://en.wikipedia.org/wiki/Visual_system>`_ . It consists of one or more convolutional layers, pooling layers and dense layers. For detailed theories please refer to the `Convolutional Neural Network <https://www.bilibili.com/video/av10590361/?p=21>`_ chapter of the `Machine Learning` course by Professor Li Hongyi of National Taiwan University.

The specific implementation is as follows, which is very similar to MLP except some new convolutional layers and pooling layers.

.. figure:: ../_static/image/model/cnn.png
    :align: center

    Figure of the CNN structure

.. literalinclude:: ../_static/code/en/model/cnn/cnn.py
    :lines: 4-38

By substituting ``model = MLP()`` in the last chapter with ``model = CNN()``, we get the following output::

    test accuracy: 0.988100

We can see that there is a significant improvement of accuracy. In fact, the accuracy can be improved further by altering the network structure of the model (e.g. adding the Dropout layer to avoid overfitting).

Recurrent Neural Networks (RNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Recurrent Neural Network is a kind of neural network suitable for processing sequential data, which is generally used for language modeling, text generation and machine translation, etc. For RNN theories, please refer to:

- `Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs <http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/>`_.
- `Recurrent Neural Network (part 1) <https://www.bilibili.com/video/av10590361/?p=36>`_  and `Recurrent Neural Network (part 2) <https://www.bilibili.com/video/av10590361/?p=37>`_ from the `Machine Learning` course by Professor Li Hongyi of Taiwan University.
- The principles of LSTM: `Understanding LSTM Networks <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_.
- Generation of RNN sequences：[Graves2013]_.

Here we use RNN to generate a piece of Nietzsche style text. [#rnn_reference]_

This task essentially predicts the probability distribution of the following alphabet of a given English text segment. For example, we have the following sentence::

    I am a studen

This sentence (sequence) has 13 characters (including spaces) in total. Based on our experience, we can predict the following alphabet will probably be "t" when we read this sequence. We would like to build a model that receives num_batch sequences consisting of seq_length encoded characters and the input tensor with shape [num_batch, seq_length], and outputs the probability distribution of the following character with the dimension of num_chars (number of characters) and the shape [num_batch, num_chars]. We will sample from the probability distribution of the following character as a prediction and generate the second, third following characters step by step in order to complete the task of text generation.

First, we still implement a simple ``DataLoader`` class to read text and encode it in characters.

.. literalinclude:: ../_static/code/en/model/rnn/rnn.py
    :lines: 31-49

Then we implement the model. We instantiate a common ``BasicLSTMCell`` unit and a dense layer for linear transformation in the ``__init__`` method. We first do an One-Hot operation on the sequence, i.e. transforming the code i into an n dimensional vector with the value 1 on the i-th position and value 0 elsewhere. Here n is the number of characters. The shape of the transformed sequence tensor is [num_batch, seq_length, num_chars]. After that, we will feed the sequences one by one into the RNN unit, i.e. feeding the state of the RNN unit ``state`` and sequences ``inputs[:, t, :]`` of the current time t into the RNN unit, and get the output of the current time ``output`` and the RNN unit state of the next time t+1. We take the last output of the RNN unit, transform it into num_chars dimensional one through a dense layer, and consider it as the model output.

.. figure:: ../_static/image/model/rnn_single.jpg
    :width: 30%
    :align: center

    Figure of ``output, state = self.cell(inputs[:, t, :], state)``

.. figure:: ../_static/image/model/rnn.jpg
    :width: 50%
    :align: center

    Figure of RNN process

The implementation is as follows:

.. literalinclude:: ../_static/code/en/model/rnn/rnn.py
    :lines: 7-21

The training process is almost the same with the previous chapter, which we reiterate here:

- Randomly read a set of training data from DataLoader;
- Feed the data into the model to acquire its predictions;
- Compare the predictions with the correct answers to evaluate the loss function;
- Differentiate the loss function with respect to model parameters;
- Update the model parameters in order to minimize the loss.

.. literalinclude:: ../_static/code/en/model/rnn/rnn.py
    :lines: 59-69

There is one thing we should notice about the process of text generation is that earlier we used the ``tf.argmax()`` function to regard the value with the maximum likelihood as the prediction. However, this method of prediction will be too rigid for text generation which also deprives the richness of the generated text. Thus, we use the ``np.random.choice()`` function for sampling based on the generated probability distribution by which even characters with small likelihood can still be possible to be sampled. Meanwhile we introduce the ``temperature`` parameter to control the shape of the distribution. Larger the value, flatter the distribution (smaller difference between the maximum and the minimum) and richer the generated text. Vice versa.

.. literalinclude:: ../_static/code/en/model/rnn/rnn.py
    :lines: 23-28

We can get the generated text by this step-by-step continuing prediction.

.. literalinclude:: ../_static/code/en/model/rnn/rnn.py
    :lines: 71-78

The generated text is as follows::

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

.. [#rnn_reference] The task and its implementation are referred to https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

Deep Reinforcement Learning (DRL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Reinforcement Learning <https://en.wikipedia.org/wiki/Reinforcement_learning>`_ focuses on how to act in the environment in order to maximize the expected benefits. It becomes more powerful if combined with the deep learning technique. The recently famous AlphaGo is a typical application of the deep reinforcement learning. For its basic knowledge please refer to:

- `Demystifying Deep Reinforcement Learning <https://ai.intel.com/demystifying-deep-reinforcement-learning/>`_.
- [Mnih2013]_.

Here, we use the deep reinforcement learning to learn how to play CartPole. To be simple, our model needs to control the pole to keep it in a straight balance state.

.. only:: html

    .. figure:: ../_static/image/model/cartpole.gif
        :width: 500
        :align: center

        The CartPole game

.. only:: latex

    .. figure:: ../_static/image/model/cartpole.png
        :width: 500
        :align: center

        The CartPole game

We use the CartPole game environment in the `Gym Environment Library powered by OpenAI <https://gym.openai.com/>`_. Please refer to the `official documentation <https://gym.openai.com/docs/>`_ and `here <http://kvfrans.com/simple-algoritms-for-solving-cartpole/>`_ for detailed installation steps and tutorials.

.. code-block:: python

    import gym
    
    env = gym.make('CartPole-v1')       # Instantiate a game environment. The parameter is its name.
    state = env.reset()                 # Initialize the environment and get its initial state.
    while True:
        env.render()                    # Render the current frame.
        action = model.predict(state)   # Assume we have a trained model that can predict the action based on the current state.
        next_state, reward, done, info = env.step(action)   # Let the environment to execute the action, get the next state, the reward of the action, a flag whether the game is done, and extra information.
        if done:                        # Exit if the game is done.
            break

Therefore our task is to train a model that can predict good actions based on the current state. Roughly speaking, a good action should maximize the sum of rewards gained during the whole game process, which is also the target of reinforcement learning.

The following code shows how to use the Deep Q-Learning in deep reinforcement learning to train the model.

.. literalinclude:: ../_static/code/en/model/rl/rl.py

.. _custom_layer:

Custom Layers *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Maybe you want to ask that what if these layers can't satisfy my needs and I need to customize my own layers?

In fact, we can not only inherit from ``tf.keras.Model`` to write your own model class, but also inherit from ``tf.keras.layers.Layer`` to write your own layer.

.. code-block:: python

    class MyLayer(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            # Initialization.

        def build(self, input_shape):     # input_shape is a TensorFlow class object which provides the input shape.
            # Call this part when the layer is firstly used. Variables created here will have self-adapted shapes without specification from users. They can also be created in __init__ part if their shapes are already determined.
            self.variable_0 = self.add_variable(...)
            self.variable_1 = self.add_variable(...)

        def call(self, input):
            # Model calling (process inputs and return outputs).
            return output

For example, if we want to implement a dense layer in :ref:`the first section of this chapter <linear>` with a specified output dimension of 1, we can write as below, creating two variables in the ``build`` method and do operations on them:

.. literalinclude:: ../_static/code/en/model/custom_layer/linear.py
    :lines: 9-21
    
With the same way, we can call our custom layers ``LinearLayer``:

.. literalinclude:: ../_static/code/en/model/custom_layer/linear.py
    :lines: 24-31

Graph Execution Mode *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In fact, these models above will be compatible with both Eager Execution mode and Graph Execution mode if we write them more carefully [#rnn_exception]_. Please notice, ``model(input_tensor)`` is only needed to run once for building a graph under Graph Execution mode.

For example, we can also call the linear model built in :ref:`the first section of this chapter <linear>` and do linear regression by the following codes:

.. literalinclude:: ../_static/code/en/model/custom_layer/linear.py
    :lines: 48-59

.. [#rnn_exception] In addition to the RNN model implemented in this chapter, we get the length of seq_length dynamically under Eager Execution, which enables us to easily control the expanding length of RNN dynamically, which is not supported by Graph Execution. In order to reach the same effect, we need to fix the length of seq_length or use ``tf.nn.dynamic_rnn`` instead (`Documentation <https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn>`_).

.. [LeCun1998] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998. http://yann.lecun.com/exdb/mnist/
.. [Graves2013] Graves, Alex. “Generating Sequences With Recurrent Neural Networks.” ArXiv:1308.0850 [Cs], August 4, 2013. http://arxiv.org/abs/1308.0850.
.. [Mnih2013] Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. “Playing Atari with Deep Reinforcement Learning.” ArXiv:1312.5602 [Cs], December 19, 2013. http://arxiv.org/abs/1312.5602.



