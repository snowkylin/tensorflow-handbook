TensorFlow Quantum: 混合量子-经典机器学习 *
============================================================

我们身边的经典计算机利用比特位和逻辑门进行二进制运算。在物理硬件上，这种运算主要是通过半导体的特殊导电性质实现的。经过几十年的发展，我们已经可以在一片小小的半导体芯片上集成上亿个晶体管，从而实现高性能的经典计算。

而量子计算（Quantum Computing）旨在利用具有量子特性（例如量子态叠加和量子纠缠）的“量子比特位”和“量子逻辑门”进行计算。这种新的计算模式可以在搜索和大数分解等重要领域达成指数级的加速，让当前无法实现的一些超大规模运算成为可能，从而可能在未来深远地改变世界。在物理硬件上，这类量子运算也可以通过一些具有量子特性的结构（例如超导约瑟夫森结）实现。

不幸的是，尽管量子计算的理论已经有了比较深入的发展，可在物理硬件上，我们目前仍然造不出一台超越经典计算机的通用量子计算机 [#f0]_ 。IBM和谷歌等业界巨头在通用量子计算机的物理构建上已经取得了一些成绩，但无论是量子比特的个数还是在退相干问题的解决上，都还远无法达到实用的层级。

以上是量子计算的基本背景，接下来我们讨论量子机器学习。量子机器学习的一种最直接的思路是使用量子计算加速传统的机器学习任务，例如量子版本的PCA、SVM和K-Means算法，然而这些算法目前都尚未达到可实用的程度。我们在本章讨论的量子机器学习则采用另一种思路，即构建参数化的量子线路（Parameterized Quantum Circuits, PQCs）。PQC可以作为深度学习模型中的层而被使用，如果我们在普通深度学习模型的基础上加入PQC，即称为混合量子-经典机器学习（Hybrid Quantum-Classical Machine Learning）。这种混合模型尤其适合于量子数据集（Quantum Data）上的任务。而TensorFlow Quantum正是帮助我们构建这种混合量子-经典机器学习模型的利器。接下来，我们会对量子计算的若干基本概念进行简介，然后介绍使用TensorFlow Quantum和谷歌的量子计算库Cirq构建PQC、将PQC嵌入Keras模型、并在量子数据集上训练混合模型的流程。

..
    https://www.tensorflow.org/quantum
    https://mp.weixin.qq.com/s?__biz=MzU1OTMyNDcxMQ==&mid=2247487901&idx=2&sn=bf00bbc09e5e1f415d1809d6333b5d5b&chksm=fc185ad5cb6fd3c3e7f77e9ccfa77b1aae083ab033b43711e84ee7f09b4ea7b0c4dbad5bfdfb&mpshare=1&scene=23&srcid=&sharer_sharetime=1585490090816&sharer_shareid=b6f86ab8b392c4d4036aa6a1d3b82824#rd
    https://www.youtube.com/watch?v=-o9AhIz1uvo
    https://medium.com/mdr-inc/tensorflow-quantum-basic-tutorial-explained-with-supplementary-2f69011036c0


量子计算基本概念
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本节将简述量子计算的一些基本概念，包括量子比特、量子门、量子线路等。

.. admonition:: 推荐阅读

    如果你希望更深入地了解量子力学以及量子计算的基本原理，建议可以从以下两本书入手：

    -  `吴飚 <http://www.phy.pku.edu.cn/~wubiao/>`_ ， `简明量子力学 <https://item.jd.com/12852996.html>`_ ，北京大学出版社，2020（简洁明快的量子力学入门教程）
    - Hidary, Jack D. `Quantum Computing: An Applied Approach <http://link.springer.com/10.1007/978-3-030-23922-0>`_ . Cham: Springer International Publishing, 2019. https://doi.org/10.1007/978-3-030-23922-0. （注重代码实操的量子计算教程，GitHub上有配套源码： https://github.com/JackHidary/quantumcomputingbook ）


量子比特
-------------------------------------------

在二进制的经典计算机中，我们用比特（Bit，也称“位”）作为信息存储的基本单位，一个比特只有0或者1两种状态。而在量子计算机中，我们使用量子比特（Quantum Bit, Qubit，也称“量子位”）进行信息的表示。量子比特也有两种基本状态 :math:`\ket{0}` 和 :math:`\ket{1}` 。不过量子比特除了可以处于这两种基本状态以外，还可以处于两者之间的叠加态（Superposition State），即 :math:`\ket{\psi} = a \ket{0} + b \ket{1}` （其中a和b是复数， :math:`|a|^2 + |b|^2 = 1` ）。例如， :math:`\ket{\psi_0} = \frac{1}{\sqrt{2}} \ket{0} + \frac{1}{\sqrt{2}} \ket{1}` 和 :math:`\ket{\psi_1} = \frac{1}{\sqrt{2}} \ket{0} - \frac{1}{\sqrt{2}} \ket{1}` 都是合法的量子态。我们也可以使用向量化的语言来表示量子比特的状态。如果我们令 :math:`\ket{0} = \begin{bmatrix}1 \\ 0\end{bmatrix}` 、 :math:`\ket{1} = \begin{bmatrix}0 \\ 1\end{bmatrix}`，则 :math:`\ket{\psi} = \begin{bmatrix}a \\ b\end{bmatrix}`、:math:`\ket{\psi_0} = \begin{bmatrix}\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}}\end{bmatrix}`、:math:`\ket{\psi_1} = \begin{bmatrix}\frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}}\end{bmatrix}`。

同时，我们可以用布洛赫球面（Bloch Sphere）来形象地展示单个量子比特的状态。球面的最顶端为 :math:`\ket{0}` ，最底端为 :math:`\ket{1}` ，而从原点到球面上任何一点的单位向量都可以是一个量子比特的状态。

.. figure:: /_static/image/quantum/bloch_sphere.png
    :width: 30%
    :align: center

    布洛赫球面（ `Bloch Sphere <https://en.wikipedia.org/wiki/Bloch_sphere>`_ ）。其中Z轴正负方向的量子态分别为基本态 :math:`\ket{0}` 和 :math:`\ket{1}` ，X轴正负方向的量子态分别为 :math:`\frac{1}{\sqrt{2}} \ket{0} + \frac{1}{\sqrt{2}} \ket{1}` 和 :math:`\frac{1}{\sqrt{2}} \ket{0} - \frac{1}{\sqrt{2}} \ket{1}` ，Y轴正负方向的量子态分别为 :math:`\frac{1}{\sqrt{2}} \ket{0} + \frac{i}{\sqrt{2}} \ket{1}` 和 :math:`\frac{1}{\sqrt{2}} \ket{0} - \frac{i}{\sqrt{2}} \ket{1}` 。`图示来源 <https://en.wikipedia.org/wiki/File:Bloch_sphere.svg>`_ 

值得特别注意的是，尽管量子比特 :math:`\ket{\psi} = a \ket{0} + b \ket{1}` 可能的状态相当之多，但一旦我们对其进行观测，则其状态会立即坍缩 [#f1]_ 到 :math:`\ket{0}` 和 :math:`\ket{1}` 这两个基本状态中的一个，其概率分别为 :math:`|a|^2` 和  :math:`|b|^2` 。

量子逻辑门
-------------------------------------------

在二进制的经典计算机中，我们有AND（与）、OR（或）、NOT（非）等逻辑门，对输入的比特状态进行变换并输出。在量子计算机中，我们同样有量子逻辑门（Quantum Logic Gate，或简称“量子门”），对量子状态进行变换并输出。如果我们使用向量化的语言来表述量子状态，则量子逻辑门可以看作是一个对状态向量进行变换的矩阵。

例如，量子非门可以表述为 :math:`X = \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}` ，于是当我们将量子非门作用于基本态 :math:`\ket{0} = \begin{bmatrix}1 \\ 0\end{bmatrix}` 时，我们得到 :math:`X\ket{0} = \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix} \begin{bmatrix}1 \\ 0\end{bmatrix} = \begin{bmatrix}0 \\ 1\end{bmatrix} = \ket{1}`。量子门也可以作用在叠加态，例如 :math:`X\ket{\psi_0} = \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix} \begin{bmatrix}\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}}\end{bmatrix} = \begin{bmatrix}\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}}\end{bmatrix} = \ket{\psi_0}` （这说明量子非门没能改变量子态 :math:`\ket{\psi_0} = \frac{1}{\sqrt{2}} \ket{0} + \frac{1}{\sqrt{2}} \ket{1}` 的状态。事实上，量子非门 :math:`X` 相当于在布洛赫球面上将量子态绕X轴旋转180度。而 :math:`\ket{\psi_0}` 就在X轴上，所以没有变化）。量子与门和或门 [#f2]_ 由于涉及到多个量子位而稍显复杂，但同样可以通过尺寸更大的矩阵实现。

可能有些读者已经想到了，既然单个量子比特的状态不止 :math:`\ket{0}` 和 :math:`\ket{1}` 两种，那么量子逻辑门作为作为对量子比特的变换，其实也完全可以不局限于与或非。事实上，只要满足一定条件的矩阵 [#f3]_ 都可以作为量子逻辑门。例如，将量子态在布洛赫球面上绕X、Y、Z轴旋转的变换 :math:`Rx(\theta)` 、:math:`Ry(\theta)` 、:math:`Rz(\theta)` （其中 :math:`\theta` 是旋转角度，当 :math:`\theta=180^\circ` 时记为 :math:`X` 、:math:`Y` 、:math:`Z` ）都是量子逻辑门。另外，有一个量子逻辑门“阿达马门”（Hadamard Gate） :math:`H = \frac{1}{\sqrt{2}} \begin{bmatrix}1 & 1 \\ 1 & -1\end{bmatrix}` 可以将量子状态从基本态转换为叠加态，在很多量子计算的场景中占据了重要地位。

量子线路
-------------------------------------------

..
    https://www.overleaf.com/read/brpwwxrqbvhh
    http://physics.unm.edu/CQuIC/Qcircuit/Qtutorial.pdf

当我们将量子比特以及量子逻辑门按顺序标记在一条或多条平行的线条上时，就构成了量子线路（Quantum Circuit，或称量子电路）。例如，对于我们在上一节讨论的，使用量子非门 :math:`X` 对基本态 :math:`\ket{0}` 进行变换的过程，我们可以写出量子线路如下：

.. figure:: /_static/image/quantum/X_circuit.png
    :width: 30%
    :align: center

    一个简单的量子线路

在量子线路中，每条横线代表一个量子比特。上图中最左边的 :math:`\ket{0}` 代表量子比特的初始态。中间的X方块代表量子非门 :math:`X` ，最右边的表盘符号代表测量操作。这个线路的意义是“对初始状态为 :math:`\ket{0}` 的量子比特执行量子非门 :math:`X` 操作，并测量变换后的量子比特状态”。根据我们在前节的讨论，变换后的量子比特状态为基本态 :math:`\ket{1}` ，因此我们可以期待该量子线路最后的测量结果始终为1。

接下来，我们考虑将上图中量子线路的量子非门 :math:`X` 换为阿达马门 :math:`H` ：

.. figure:: /_static/image/quantum/H_circuit.png
    :width: 30%
    :align: center

    将量子非门 :math:`X` 换为阿达马门 :math:`H` 后的量子线路

阿达马门对应的矩阵表示为 :math:`H = \frac{1}{\sqrt{2}} \begin{bmatrix}1 & 1 \\ 1 & -1\end{bmatrix}` ，于是我们可以计算出变换后的量子态为 :math:`H\ket{0} = \frac{1}{\sqrt{2}} \begin{bmatrix}1 & 1 \\ 1 & -1\end{bmatrix}\begin{bmatrix}1 \\ 0\end{bmatrix} = \begin{bmatrix}\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}}\end{bmatrix} = \frac{1}{\sqrt{2}} \ket{0} + \frac{1}{\sqrt{2}} \ket{1}` 。这是一个 :math:`\ket{0}` 和 :math:`\ket{1}` 的叠加态，在观测后会坍缩到基本态，其概率分别为 :math:`|\frac{1}{\sqrt{2}}|^2 = \frac{1}{2}` 。也就是说，这个量子线路的观测结果类似于扔硬币。假若观测20次，则大约10次的结果是 :math:`\ket{0}` ，10次的结果是 :math:`\ket{1}` 。

实例：使用Cirq建立简单的量子线路
-------------------------------------------

`Cirq <https://cirq.readthedocs.io/>`_ 是谷歌主导的开源量子计算库，可以帮助我们方便地建立量子线路并模拟测量结果（我们在下一节介绍TensorFlow Quantum的时候还会用到它）。Cirq是一个Python库，可以使用 ``pip install cirq`` 进行安装。以下代码实现了上节所建立的两个简单的量子线路，并分别进行了20次的模拟测量。

.. literalinclude:: /_static/code/zh/appendix/quantum/basic.py

结果如下：

::

    0: ───X───M───
    0=11111111111111111111
    0: ───H───M───
    0=00100111001111101100

可见第一个量子线路的测量结果始终为1，第二个量子态的20次测量结果中有9次是0，11次是1（如果你多运行几次，会发现0和1出现的概率趋近于 :math:`\frac{1}{2}` ）。可见结果符合我们在上节中的分析。

混合量子-经典机器学习
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本节介绍混合量子-经典机器学习的基本概念，以及使用 TensorFlow Quantum 建立此类模型的方法。

在混合量子-经典机器学习过程中，我们使用量子数据集训练混合量子-经典模型。混合量子-经典模型的前半部分是量子模型（即参数化的量子线路）。量子模型接受量子数据集作为输入，对输入使用量子门进行变换，然后通过测量转换为经典数据。测量后的经典数据输入经典模型，并使用常规的损失函数计算模型的损失值。最后，基于损失函数的值计算模型参数的梯度并更新模型参数。这一过程不仅包括经典模型的参数，也包括量子模型的参数。具体流程如下图所示。

.. figure:: /_static/image/quantum/pipeline.png
    :width: 60%
    :align: center

    经典机器学习（上图）与混合量子-经典机器学习（下图）的流程对比

TensorFlow Quantum 即是一个与 TensorFlow Keras 结合紧密的，可快速建立混合量子-经典机器学习模型的开源库，可以使用 ``pip install tensorflow-quantum`` 进行安装。

后文示例均默认使用以下代码导入 TensorFlow、TensorFlow Quantum和Cirq：

.. code-block:: python

    import tensorflow as tf
    import tensorflow_quantum as tfq
    import cirq

.. admonition:: 推荐阅读

    Broughton, Michael, Guillaume Verdon, Trevor McCourt, Antonio J. Martinez, Jae Hyeon Yoo, Sergei V. Isakov, Philip Massey, et al. “ `TensorFlow Quantum: A Software Framework for Quantum Machine Learning. <http://arxiv.org/abs/2003.02989>`_ ” ArXiv:2003.02989 [Cond-Mat, Physics:Quant-Ph], March 5, 2020. （TensorFlow Quantum 白皮书）

量子数据集与带参数的量子门
-------------------------------------------

以有监督学习为例，经典数据集由经典数据和标签组成。经典数据中的每一项是一个由不同特征组成的向量。我们可以将经典数据集写作 :math:`(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)` ，其中 :math:`x_i = (x_{i,1}, \cdots, x_{i,K})` 。量子数据集同样由数据和标签组成，而数据中的每一项是一个量子态。以前节单量子比特的量子态为例，我们可以将每一项数据写作 :math:`x_i = a_i \ket{0} + b_i \ket{1}` 。在具体实现上，我们可以通过量子线路来生成量子数据。也就是说，每一项数据 :math:`x_i` 都对应着一个量子线路。例如，我们可以通过以下代码，使用Cirq生成一组量子数据：

.. code-block:: python

    q = cirq.GridQubit(0, 0)
    q_data = []
    for i in range(100):
        x_i = cirq.Circuit(
            cirq.rx(np.random.rand() * np.pi)(q)
        )
        q_data.append(x_i)

在这一过程中，我们使用了一个带参数的量子门 ``cirq.rx(angle)(q)`` 。和之前我们使用的量子门 ``cirq.X(q)`` , ``cirq.H(q)`` 不同的是，这个量子门多了一个参数 ``angle`` ，表示将量子比特 ``q`` 绕布洛赫球面的x轴旋转 ``angle`` 角度（弧度制）。以上代码生成了100项量子数据，每项数据是从基本态 :math:`\ket{0}` 开始绕布洛赫球面的x轴随机旋转 :math:`[0, \pi]` 弧度所变换而来的量子态。量子数据集在不少量子相关的领域（如化学、材料科学、生物学和药物发现等）都有应用。

当我们要将量子数据集作为Keras的输入时，可以使用TensorFlow Quantum的 ``convert_to_tensor`` 方法，将量子数据集转换为张量：

.. code-block:: python

    q_data = tfq.convert_to_tensor(q_data)

值得注意的是，当使用量子数据集作为Keras模型的训练数据时，Keras模型的输入类型（ ``dtype`` ）需要为 ``tf.dtypes.string`` 。

参数化的量子线路（PQC）
-------------------------------------------

当我们在建立量子线路时使用了带参数的量子门，且该参数可以自由调整时，我们就称这样的量子线路为参数化的量子线路。Cirq支持结合SymPy这一Python下的符号运算库实现参数化的量子线路，示例如下

.. code-block:: python

    import sympy
    theta = sympy.Symbol('theta')
    q_model = cirq.Circuit(cirq.rx(theta)(q))

在上面的代码中，我们建立了如下图所示的量子线路。该量子线路可以将任意输入量子态 :math:`\ket{\psi}` 绕布洛赫球面的x轴逆时针旋转 :math:`\theta` 度，其中 :math:`\theta` 是使用 ``sympy.Symbol`` 声明的符号变量（即参数）。

.. figure:: /_static/image/quantum/pqc.png
    :width: 30%
    :align: center

    参数化的量子线路示例

将参数化的量子线路嵌入机器学习模型
-------------------------------------------

通过TensorFlow Quantum，我们可以轻松地将参数化的量子线路以Keras层的方式嵌入Keras模型。例如对于前节建立的参数化的量子线路 ``q_model`` ，我们可以使用 ``tfq.layers.PQC`` 将其直接作为一个Keras层使用

.. code-block:: python

    q_layer = tfq.layers.PQC(q_model, cirq.Z(q))
    expectation_output = q_layer(q_data_input)

``tfq.layers.PQC`` 的第一个参数为使用Cirq建立的参数化的量子线路，第二个参数为测量方式，此处使用 ``cirq.Z(q)`` 在布洛赫球面的Z轴进行测量。

以上代码也可直接写作：

.. code-block:: python

    expectation_output = tfq.layers.PQC(q_model, cirq.Z(q))(q_data_input)

实例：对量子数据集进行二分类
-------------------------------------------

在以下代码中，我们首先建立了一个量子数据集，其中一半的数据项为基本态 :math:`\ket{0}` 绕布洛赫球面的x轴逆时针旋转 :math:`\frac{\pi}{2}` 弧度（即 :math:`\frac{1}{\sqrt{2}} \ket{0} - \frac{i}{\sqrt{2}} \ket{1}` ），另一半则为 :math:`\frac{3\pi}{2}` 弧度（即 :math:`\frac{1}{\sqrt{2}} \ket{0} + \frac{i}{\sqrt{2}} \ket{1}` ）。所有的数据均加入了绕x,y轴方向旋转的，标准差为 :math:`\frac{\pi}{4}` 的高斯噪声。对于这个量子数据集，如果不加变换而直接测量，则所有数据都会和抛硬币一样等概率随机坍缩到基本态 :math:`\ket{0}` 和 :math:`\ket{1}` ，从而无法区分。

为了区分这两类数据，我们接下来建立了一个量子模型，这个模型将单比特量子态绕布洛赫球面的x轴逆时针旋转 :math:`\theta` 弧度。变换过后量子态数据的测量值送入“全连接层+softmax”的经典机器学习模型，并使用交叉熵作为损失函数。模型训练过程会自动同时调整量子模型中 :math:`\theta` 的值和全连接层的权值，使得整个混合量子-经典机器学习模型的准确度较高。

.. literalinclude:: /_static/code/zh/appendix/quantum/binary_classification.py

输出：

::

    ...
    200/200 [==============================] - 0s 165us/sample - loss: 0.1586 - sparse_categorical_accuracy: 0.9500
    [array([-1.5279944], dtype=float32)]

可见，通过训练，模型在训练集上可以达到95%的准确率， :math:`\theta = -1.5279944 \approx -\frac{\pi}{2} = -1.5707963...` 。而当 :math:`\theta = -\frac{\pi}{2}` 时，恰好可以使得两种类型的数据分别接近基本态 :math:`\ket{0}` 和 :math:`\ket{1}` ，从而达到最易区分的状态。

.. [#f0] 此手册的行文时间为公元2020年，如果你来自未来，请理解作者的时代局限性。
.. [#f1] “坍缩”一词多用于量子观测的哥本哈根诠释，除此以外还有多世界理论等。此处使用“坍缩”一词仅是方便表述。
.. [#f2] 其实更常见的基础二元量子门是“量子控制非门”（CNOT）和“量子交换门”（SWAP）。
.. [#f3] 这种矩阵被称之为“幺正矩阵”或“酉矩阵”。

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 201 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>