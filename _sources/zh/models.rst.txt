TensorFlow模型
================

.. _linear:

本章介绍如何使用TensorFlow快速搭建动态模型。

前置知识：

* `Python面向对象 <http://www.runoob.com/python3/python3-class.html>`_ （在Python内定义类和方法、类的继承、构造和析构函数，`使用super()函数调用父类方法 <http://www.runoob.com/python/python-func-super.html>`_ ，`使用__call__()方法对实例进行调用 <https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014319098638265527beb24f7840aa97de564ccc7f20f6000>`_ 等）；
* 多层感知机、卷积神经网络、循环神经网络和强化学习（每节之前给出参考资料）。

模型（Model）与层（Layer）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
..  https://www.tensorflow.org/programmers_guide/eager

如上一章所述，为了增强代码的可复用性，我们往往会将模型编写为类，然后在模型调用的地方使用 ``y_pred = model(X)`` 的形式进行调用。 **模型类** 的形式非常简单，主要包含 ``__init__()`` （构造函数，初始化）和 ``call(input)`` （模型调用）两个方法，但也可以根据需要增加自定义的方法。 [#call]_ 

.. code-block:: python

    class MyModel(tf.keras.Model):
        def __init__(self):
            super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
            # 此处添加初始化代码（包含call方法中会用到的层）

        def call(self, inputs):
            # 此处添加模型调用的代码（处理输入并返回输出）
            return output

在这里，我们的模型类继承了 ``tf.keras.Model`` 。Keras是一个用Python编写的高级神经网络API，现已得到TensorFlow的官方支持和内置。继承 ``tf.keras.Model`` 的一个好处在于我们可以使用父类的若干方法和属性，例如在实例化类后可以通过 ``model.variables`` 这一属性直接获得模型中的所有变量，免去我们一个个显式指定变量的麻烦。

同时，我们引入 **“层”（Layer）** 的概念，层可以视为比模型粒度更细的组件单位，将计算流程和变量进行了封装。我们可以使用层来快速搭建模型。

上一章中简单的线性模型 ``y_pred = tf.matmul(X, w) + b`` ，我们可以通过模型类的方式编写如下：

.. literalinclude:: ../_static/code/zh/model/linear/linear.py

这里，我们没有显式地声明 ``w`` 和 ``b`` 两个变量并写出 ``y_pred = tf.matmul(X, w) + b`` 这一线性变换，而是在初始化部分实例化了一个全连接层（ ``tf.keras.layers.Dense`` ），并在call方法中对这个层进行调用。全连接层封装了 ``output = activation(tf.matmul(input, kernel) + bias)`` 这一线性变换+激活函数的计算操作，以及 ``kernel`` 和 ``bias`` 两个变量。当不指定激活函数时（即 ``activation(x) = x`` ），这个全连接层就等价于我们上述的线性变换。顺便一提，全连接层可能是我们编写模型时使用最频繁的层。

如果我们需要显式地声明自己的变量并使用变量进行自定义运算，请参考 :ref:`自定义层 <custom_layer>`。

.. [#call] 在Python类中，对类的实例 ``myClass`` 进行形如 ``myClass()`` 的调用等价于 ``myClass.__call__()`` 。在这里，我们的模型继承了 ``tf.keras.Model`` 这一父类。该父类中包含 ``__call__()`` 的定义，其中调用了 ``call()`` 方法，同时进行了一些keras的内部操作。这里，我们通过继承 ``tf.keras.Model`` 并重载 ``call()`` 方法，即可在保持keras结构的同时加入模型调用的代码。具体请见本章初“前置知识”的 ``__call__()`` 部分。

.. _mlp:

基础示例：多层感知机（MLP）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们从编写一个最简单的 `多层感知机 <https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E5%99%A8>`_ （Multilayer Perceptron, MLP）开始，介绍TensorFlow的模型编写方式。这里，我们使用多层感知机完成MNIST手写体数字图片数据集 [LeCun1998]_ 的分类任务。

.. figure:: ../_static/image/model/mnist_0-9.png
    :align: center

    MNIST手写体数字图片示例

先进行预备工作，实现一个简单的 ``DataLoader`` 类来读取MNIST数据集数据。

.. literalinclude:: ../_static/code/zh/model/mlp/main.py
    :lines: 13-23

多层感知机的模型类实现与上面的线性模型类似，所不同的地方在于层数增加了（顾名思义，“多层”感知机），以及引入了非线性激活函数（这里使用了 `ReLU函数 <https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E6%95%B4%E6%B5%81%E5%87%BD%E6%95%B0>`_ ， 即下方的 ``activation=tf.nn.relu`` ）。该模型输入一个向量（比如这里是拉直的1×784手写体数字图片），输出10维的信号，分别代表这张图片属于0到9的概率。这里我们加入了一个predict方法，对图片对应的数字进行预测。在预测的时候，选择概率最大的数字进行预测输出。

.. literalinclude:: ../_static/code/zh/model/mlp/mlp.py
    :lines: 4-17

定义一些模型超参数：

.. literalinclude:: ../_static/code/zh/model/mlp/main.py
    :lines: 8-10

实例化模型，数据读取类和优化器：

.. code-block:: python

    model = MLP()
    data_loader = DataLoader()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

然后迭代进行以下步骤：

- 从DataLoader中随机取一批训练数据；
- 将这批数据送入模型，计算出模型的预测值；
- 将模型预测值与真实值进行比较，计算损失函数（loss）；
- 计算损失函数关于模型变量的导数；
- 使用优化器更新模型参数以最小化损失函数。

具体代码实现如下：

.. literalinclude:: ../_static/code/zh/model/mlp/main.py
    :lines: 32-39

接下来，我们使用验证集测试模型性能。具体而言，比较验证集上模型预测的结果与真实结果，输出预测正确的样本数占总样本数的比例：

.. literalinclude:: ../_static/code/zh/model/mlp/main.py
    :lines: 41-43

输出结果::

    test accuracy: 0.947900

可以注意到，使用这样简单的模型，已经可以达到95%左右的准确率。

卷积神经网络（CNN）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`卷积神经网络 <https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C>`_ （Convolutional Neural Network, CNN）是一种结构类似于人类或动物的 `视觉系统 <https://zh.wikipedia.org/wiki/%E8%A7%86%E8%A7%89%E7%B3%BB%E7%BB%9F>`_ 的人工神经网络，包含一个或多个卷积层（Convolutional Layer）、池化层（Pooling Layer）和全连接层（Dense Layer）。具体原理建议可以参考台湾大学李宏毅教授的《机器学习》课程的 `Convolutional Neural Network <https://www.bilibili.com/video/av10590361/?p=21>`_ 一章。

具体的实现见下，和MLP很类似，只是新加入了一些卷积层和池化层。

.. figure:: ../_static/image/model/cnn.png
    :align: center

    CNN结构图示

.. literalinclude:: ../_static/code/zh/model/cnn/cnn.py
    :lines: 4-38

将前节的 ``model = MLP()`` 更换成 ``model = CNN()`` ，输出如下::

    test accuracy: 0.988100

可以发现准确率有非常显著的提高。事实上，通过改变模型的网络结构（比如加入Dropout层防止过拟合），准确率还有进一步提升的空间。

循环神经网络（RNN）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

循环神经网络（Recurrent Neural Network, RNN）是一种适宜于处理序列数据的神经网络，被广泛用于语言模型、文本生成、机器翻译等。关于RNN的原理，可以参考：

- `Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs <http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/>`_
- 台湾大学李宏毅教授的《机器学习》课程的 `Recurrent Neural Network (part 1) <https://www.bilibili.com/video/av10590361/?p=36>`_ `Recurrent Neural Network (part 2) <https://www.bilibili.com/video/av10590361/?p=37>`_ 两部分。
- LSTM原理：`Understanding LSTM Networks <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_
- RNN序列生成：[Graves2013]_

这里，我们使用RNN来进行尼采风格文本的自动生成。 [#rnn_reference]_

这个任务的本质其实预测一段英文文本的接续字母的概率分布。比如，我们有以下句子::

    I am a studen

这个句子（序列）一共有13个字符（包含空格）。当我们阅读到这个由13个字符组成的序列后，根据我们的经验，我们可以预测出下一个字符很大概率是“t”。我们希望建立这样一个模型，输入num_batch个由编码后字符组成的，长为seq_length的序列，输入张量形状为[num_batch, seq_length]，输出这些序列接续的下一个字符的概率分布，概率分布的维度为字符种类数num_chars，输出张量形状为[num_batch, num_chars]。我们从下一个字符的概率分布中采样作为预测值，然后滚雪球式地生成下两个字符，下三个字符等等，即可完成文本的生成任务。

首先，还是实现一个简单的 ``DataLoader`` 类来读取文本，并以字符为单位进行编码。

.. literalinclude:: ../_static/code/zh/model/rnn/rnn.py
    :lines: 31-49

接下来进行模型的实现。在 ``__init__`` 方法中我们实例化一个常用的 ``BasicLSTMCell`` 单元，以及一个线性变换用的全连接层，我们首先对序列进行One Hot操作，即将编码i变换为一个n维向量，其第i位为1，其余均为0。这里n为字符种类数num_char。变换后的序列张量形状为[num_batch, seq_length, num_chars]。接下来，我们将序列从头到尾依序送入RNN单元，即将当前时间t的RNN单元状态 ``state`` 和t时刻的序列 ``inputs[:, t, :]`` 送入RNN单元，得到当前时间的输出 ``output`` 和下一个时间t+1的RNN单元状态。取RNN单元最后一次的输出，通过全连接层变换到num_chars维，即作为模型的输出。

.. figure:: ../_static/image/model/rnn_single.jpg
    :width: 30%
    :align: center

    ``output, state = self.cell(inputs[:, t, :], state)`` 图示

.. figure:: ../_static/image/model/rnn.jpg
    :width: 50%
    :align: center

    RNN流程图示

具体实现如下：

.. literalinclude:: ../_static/code/zh/model/rnn/rnn.py
    :lines: 7-21

训练过程与前节基本一致，在此复述：

- 从DataLoader中随机取一批训练数据；
- 将这批数据送入模型，计算出模型的预测值；
- 将模型预测值与真实值进行比较，计算损失函数（loss）；
- 计算损失函数关于模型变量的导数；
- 使用优化器更新模型参数以最小化损失函数。

.. literalinclude:: ../_static/code/zh/model/rnn/rnn.py
    :lines: 59-69

关于文本生成的过程有一点需要特别注意。之前，我们一直使用 ``tf.argmax()`` 函数，将对应概率最大的值作为预测值。然而对于文本生成而言，这样的预测方式过于绝对，会使得生成的文本失去丰富性。于是，我们使用 ``np.random.choice()`` 函数按照生成的概率分布取样。这样，即使是对应概率较小的字符，也有机会被取样到。同时，我们加入一个 ``temperature`` 参数控制分布的形状，参数值越大则分布越平缓（最大值和最小值的差值越小），生成文本的丰富度越高；参数值越小则分布越陡峭，生成文本的丰富度越低。

.. literalinclude:: ../_static/code/zh/model/rnn/rnn.py
    :lines: 23-28

通过这种方式进行“滚雪球”式的连续预测，即可得到生成文本。

.. literalinclude:: ../_static/code/zh/model/rnn/rnn.py
    :lines: 71-78

生成的文本如下::

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

.. [#rnn_reference] 此处的任务及实现参考了 https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

深度强化学习（DRL）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`强化学习 <https://zh.wikipedia.org/wiki/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0>`_ （Reinforcement learning，RL）强调如何基于环境而行动，以取得最大化的预期利益。结合了深度学习技术后的强化学习更是如虎添翼。这两年广为人知的AlphaGo即是深度强化学习的典型应用。深度强化学习的基础知识可参考：

- `Demystifying Deep Reinforcement Learning <https://ai.intel.com/demystifying-deep-reinforcement-learning/>`_ （`中文编译 <https://snowkylin.github.io/rl/2017/01/04/Reinforcement-Learning.html>`_）
- [Mnih2013]_

这里，我们使用深度强化学习玩CartPole（平衡杆）游戏。简单说，我们需要让模型控制杆的左右运动，以让其一直保持竖直平衡状态。

.. only:: html

    .. figure:: ../_static/image/model/cartpole.gif
        :width: 500
        :align: center

        CartPole游戏

.. only:: latex

    .. figure:: ../_static/image/model/cartpole.png
        :width: 500
        :align: center

        CartPole游戏

我们使用 `OpenAI推出的Gym环境库 <https://gym.openai.com/>`_ 中的CartPole游戏环境，具体安装步骤和教程可参考 `官方文档 <https://gym.openai.com/docs/>`_ 和 `这里 <https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-4-gym/>`_ 。Gym的基本调用方法如下：

.. code-block:: python

    import gym
    
    env = gym.make('CartPole-v1')       # 实例化一个游戏环境，参数为游戏名称
    state = env.reset()                 # 初始化环境，获得初始状态
    while True:
        env.render()                    # 对当前帧进行渲染，绘图到屏幕
        action = model.predict(state)   # 假设我们有一个训练好的模型，能够通过当前状态预测出这时应该进行的动作
        next_state, reward, done, info = env.step(action)   # 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
        if done:                        # 如果游戏结束则退出循环
            break

那么，我们的任务就是训练出一个模型，能够根据当前的状态预测出应该进行的一个好的动作。粗略地说，一个好的动作应当能够最大化整个游戏过程中获得的奖励之和，这也是强化学习的目标。

以下代码展示了如何使用深度强化学习中的Deep Q-Learning方法来训练模型。

.. literalinclude:: ../_static/code/zh/model/rl/rl.py

.. _custom_layer:

自定义层 *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可能你还会问，如果现有的这些层无法满足我的要求，我需要定义自己的层怎么办？

事实上，我们不仅可以继承 ``tf.keras.Model`` 编写自己的模型类，也可以继承 ``tf.keras.layers.Layer`` 编写自己的层。

.. code-block:: python

    class MyLayer(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            # 初始化代码

        def build(self, input_shape):     # input_shape 是一个 TensorShape 类型对象，提供输入的形状
            # 在第一次使用该层的时候调用该部分代码，在这里创建变量可以使得变量的形状自适应输入的形状
            # 而不需要使用者额外指定变量形状。
            # 如果已经可以完全确定变量的形状，也可以在__init__部分创建变量
            self.variable_0 = self.add_variable(...)
            self.variable_1 = self.add_variable(...)

        def call(self, input):
            # 模型调用的代码（处理输入并返回输出）
            return output

例如，如果我们要自己实现一个 :ref:`本章第一节 <linear>` 中的全连接层，但指定输出维度为1，可以按如下方式编写，在 ``build`` 方法中创建两个变量，并在 ``call`` 方法中使用创建的变量进行运算：

.. literalinclude:: ../_static/code/zh/model/custom_layer/linear.py
    :lines: 9-21
    
使用相同的方式，可以调用我们自定义的层 ``LinearLayer``：

.. literalinclude:: ../_static/code/zh/model/custom_layer/linear.py
    :lines: 24-31

Graph Execution模式 *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

事实上，只要在编写模型的时候稍加注意，以上的模型都是可以同时兼容Eager Execution模式和Graph Execution模式的 [#rnn_exception]_ 。注意，在Graph Execution模式下， ``model(input_tensor)`` 只需运行一次以完成图的建立操作。

例如，通过以下代码，同样可以调用 :ref:`本章第一节 <linear>` 建立的线性模型并进行线性回归：

.. literalinclude:: ../_static/code/zh/model/custom_layer/linear.py
    :lines: 48-59

.. [#rnn_exception] 除了本章实现的RNN模型以外。在RNN模型的实现中，我们通过Eager Execution动态获取了seq_length的长度，使得我们可以方便地动态控制RNN的展开长度。然而Graph Execution不支持这一点，为了达到相同的效果，我们需要固定seq_length的长度，或者使用 ``tf.nn.dynamic_rnn`` （ `文档 <https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn>`_ ）。

.. [LeCun1998] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998. http://yann.lecun.com/exdb/mnist/
.. [Graves2013] Graves, Alex. “Generating Sequences With Recurrent Neural Networks.” ArXiv:1308.0850 [Cs], August 4, 2013. http://arxiv.org/abs/1308.0850.
.. [Mnih2013] Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. “Playing Atari with Deep Reinforcement Learning.” ArXiv:1312.5602 [Cs], December 19, 2013. http://arxiv.org/abs/1312.5602.



