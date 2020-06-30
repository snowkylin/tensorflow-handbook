TensorFlow Quantum: Hybrid Quantum-classical Machine Learning *
===============================================================

The classic computer around us uses bits and logic gates for binary operations. In physical hardware, such arithmetic is primarily achieved by the special conductive properties of semiconductors. After decades of development, we have been able to integrate hundreds of millions of transistors on a tiny semiconductor chip, enabling high-performance classical computing.

Quantum Computing, on the other hand, aims to use "quantum bits" and "quantum logic gates" with quantum properties such as superposition and entanglement to perform calculations. This new computing paradigm could achieve exponential acceleration in important areas such as search and large number decomposition, making possible some of the hyperscale computing that is not currently possible, potentially changing the world profoundly in the future. On physical hardware, such quantum computing can also be implemented by some structures with quantum properties (e.g., superconducting Josephson junctions).

Unfortunately, although the theory of quantum computing has been developed in depth, in terms of physical hardware, we are still unable to build a general quantum computer [#f0]_ that surpasses the classical computer. IBM and Google have made some achievements in the physical construction of general quantum computers, but neither the number of quantum bits nor the solution of decoherence problems are yet to reach the practical level.

The above is the basic background of quantum computing, and next we discuss quantum machine learning. One of the most straightforward ways of thinking about quantum machine learning is to use quantum computing to accelerate traditional machine learning tasks, such as quantum versions of PCA, SVM, and K-Means algorithms, yet none of these algorithms have yet reached a practical level. The quantum machine learning we discuss in this chapter takes a different line of thinking, which is to construct Parameterized Quantum Circuits (PQCs). PQCs can be used as layers in a deep learning model, which is called Hybrid Quantum-Classical Machine Learning (HQC) if we add PQCs to the ordinary deep learning model. This hybrid model is particularly suitable for tasks on quantum datasets. TensorFlow Quantum helps us build this kind of hybrid quantum-classical machine learning model. Next, we will provide an introduction to several basic concepts of quantum computing, and then describe the process of building a PQC using TensorFlow Quantum and Google's quantum computing library Cirq, embedding the PQC into a Keras model, and training a hybrid model on a quantum dataset.

..
    https://www.tensorflow.org/quantum
    https://mp.weixin.qq.com/s?__biz=MzU1OTMyNDcxMQ==&mid=2247487901&idx=2&sn=bf00bbc09e5e1f415d1809d6333b5d5b&chksm=fc185ad5cb6fd3c3e7f77e9ccfa77b1aae083ab033b43711e84ee7f09b4ea7b0c4dbad5bfdfb&mpshare=1&scene=23&srcid=&sharer_sharetime=1585490090816&sharer_shareid=b6f86ab8b392c4d4036aa6a1d3b82824#rd
    https://www.youtube.com/watch?v=-o9AhIz1uvo
    https://medium.com/mdr-inc/tensorflow-quantum-basic-tutorial-explained-with-supplementary-2f69011036c0


Basic concepts of quantum computing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section will briefly describe some basic concepts of quantum computing, including quantum bits, quantum gates, quantum circuits, etc.

.. admonition:: recommended reading

    If you want a deeper understanding of quantum mechanics and the fundamentals of quantum computing, it is recommended to start with the following two books.

    - Griffiths, D., & Schroeter, D. (2018). `Introduction to Quantum Mechanics <https://www.cambridge.org/core/books/introduction-to-quantum-mechanics/990799CA07A83FC5312402AF6860311E>`_ . Cambridge: Cambridge University Press. doi:10.1017/9781316995433
    - Hidary, Jack D. `Quantum Computing: An Applied Approach <http://link.springer.com/10.1007/978-3-030-23922-0>`_ . Cham: Springer International Publishing, 2019. https://doi.org/10.1007/978-3-030-23922-0. (Tutorial on Quantum Computing with a focus on code-based practice, source code available on GitHub: https://github.com/JackHidary/quantumcomputingbook)


Quantum bit
-------------------------------------------

In classical binary computers, we use bits as the basic unit of information storage, and a bit has only two states, 0 or 1. In a quantum computer, we use Quantum Bits (Qubits) to represent information. Quantum bits also have two pure states :math:`\ket{0}` and :math:`\ket{1}`. However, in addition to these two pure states, a quantum bit can also be in a superposition state between them, i.e. :math:`\ket{\psi} = a \ket{0} + b \ket{1}` (where a and b are complex number, :math:`|a|^2 + |b|^2 = 1`). For example, :math:`\ket{\psi_0} = \frac{1}{\sqrt{2}} \ket{0} + \frac{1}{\sqrt{2}} \ket{1}` and :math:`\ket{\psi_1} = \frac{1}{\sqrt{2}} \ket{0} - \frac{1}{\sqrt{2}} \ket{1}` are both vaild quantum states. We can also use a vector to represent the state of a quantum bit. If we let :math:`\ket{0} = \begin{bmatrix}1 \\0\end{bmatrix}`, :math:`\ket{1} = \begin{bmatrix}0 \\1\end{bmatrix}`, then :math:`\ket{\psi} = \begin{bmatrix}a \\ b\end{bmatrix}`, :math:`\ket{\psi_0} = \begin{bmatrix}\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}`, :math:`\ket{\psi_1} = \begin{bmatrix}\frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}}\end{bmatrix}`.

We can also use the Bloch Sphere to graphically demonstrate the state of a single quantum bit. The topmost part of the sphere is :math:`\ket{0}` and the bottommost part is :math:`\ket{1}`. The unit vector from the origin to any point on the sphere can be a state of quantum bit.

.. figure:: /_static/image/quantum/bloch_sphere.png
    :width: 30%
    :align: center

    `Bloch Sphere <https://en.wikipedia.org/wiki/Bloch_sphere>`_. `Source of illustration <https://en.wikipedia.org/wiki/File:Bloch_sphere.svg>`_ 

It is worth noting in particular that although quantum bits :math:`\ket{\psi} = a \ket{0} + b \ket{1}` have quite a few possible states, once we observe them, their states immediately collapse [#f1]_ into one of two pure states of :math:`\ket{0}` and :math:`\ket{1}` with probabilities of :math:`|a|^2` and :math:`|b|^2`, respectively.

Quantum logic gate
-------------------------------------------

In binary classical computers, we have logic gates such as AND, OR and NOT that transform the input bit state. In quantum computers, we also have Quantum Logic Gates (or "quantum gates" for short) that transform quantum states. If we use vectors to represent quantum states, the quantum logic gate can be seen as a matrix that transforms the state vectors.

For example, the quantum NOT gate can be expressed as :math:`X = \begin{bmatrix}0 & 1 \\1 & 0\end{bmatrix}`, so when we act the quantum NOT gate on the pure state :math:`\ket{0} = \begin{bmatrix}1 \\0\end{bmatrix}`, we get :math:`X\ket{0} = \begin{bmatrix}0 & 1 \\1 & 0\end{bmatrix} \begin{bmatrix}1 \\0\end{bmatrix} = \begin{bmatrix}0 \\1\end{bmatrix}`.  In fact, quantum NOT gates :math:`X` are equivalent to rotating a quantum state 180 degrees around the X axis on a Bloch sphere. and :math:`\ket{\psi_0}` is on the X-axis, so no change). Quantum AND gates and OR gates [#f2]_ are slightly more complex due to the multiple quantum bits involved, but are equally achievable with matrices of greater size.

It may have occurred to some readers that since there are more states of a single quantum bit than :math:`\ket{0}` and :math:`\ket{1}`, then quantum logic gates as transformations of quantum bits can in fact be completely unrestricted to AND, OR and NOT. In fact, any matrix [#f3]_ that meets certain conditions can serve as a quantum logic gate. For example, transforms that rotate quantum states around the X, Y and Z axes on the Bloch sphere :math:`Rx(\theta)`, :math:`Ry(\theta)`, :math:`Rz(\theta)` (where :math:`\theta` is the angle of rotation and when :math:`\theta=180^\circ` they are noted as :math:`X`, math:`Y`, math:`Z`) are quantum logic gates. In addition, there is a quantum logic gate "Hadamard Gate" :math:`H = \frac{1}{\sqrt{2}} \begin{bmatrix}1 & 1 \\1 & -1\end{bmatrix}` that can convert quantum states from pure to superposition states, which occupies an important place in many scenarios of quantum computing.

Quantum circuit
-------------------------------------------

..
    https://www.overleaf.com/read/brpwwxrqbvhh
    http://physics.unm.edu/CQuIC/Qcircuit/Qtutorial.pdf

When we mark quantum bits, as well as quantum logic gates, sequentially on one or more parallel lines, they constitute a quantum circuit. For example, for the process we discussed in the previous section, using quantum NOT gate :math:`X` to transform the pure state :math:`\ket{0}`, we can write the quantum circuit as follows.

.. figure:: /_static/image/quantum/X_circuit.png
    :width: 30%
    :align: center

    A simple quantum circuit.

In a quantum circuit, each horizontal line represents one quantum bit. The leftmost :math:`\ket{0}` in the above diagram represents the initial state of a quantum bit. The X square in the middle represents the quantum NOT gate :math:`X` and the dial symbol on the right represents the measurement operation. The meaning of this line is "to perform quantum NOT gate :math:`X` operations on a quantum bit whose initial state is :math:`\ket{0}` and measure the transformed quantum bit state". According to our discussion in the previous section, the transformed quantum bit state is the pure state :math:`\ket{1}`, so we can expect the final measurement of this quantum circuit to be always 1.

Next, we consider replacing the quantum NOT gate :math:`X` of the quantum circuit in the above figure with the Hadamard gate :math:`H`.

.. figure:: /_static/image/quantum/H_circuit.png
    :width: 30%
    :align: center

    Quantum line after replacing quantum NOT gate :math:`X` with Hadamard gate :math:`H`

The matrix corresponding to the Hadamard gate is expressed as :math:`H = \frac{1}{\sqrt{2}} \begin{bmatrix}1 & 1 \\1 & -1 \end{bmatrix}`, so we can calculate the transformed quantum state as  :math:`H\ket{0} = \frac{1}{\sqrt{2}} \begin{bmatrix}1 & 1 \\ 1 & -1\end{bmatrix}\begin{bmatrix}1 \\ 0\end{bmatrix} = \begin{bmatrix}\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}}\end{bmatrix} = \frac{1}{\sqrt{2}} \ket{0} + \frac{1}{\sqrt{2}} \ket{1}` . This is a superposition state of :math:`\ket{0}` and :math:`\ket{1}` that collapses to a pure state after observation with probabilities of :math:`|\frac{1}{\sqrt{2}}|^2 = \frac{1}{2}`. That is, the observation of this quantum circuit is similar to a coin toss. If 20 observations are made, the result for about 10 times are :math:`\ket{0}` and the result for 10 times are :math:`\ket{1}`.

Example: Create a simple circuit circuit using Cirq
---------------------------------------------------

`Cirq <https://cirq.readthedocs.io/>`_ is a Google-led open source quantum computing library that helps us easily build quantum circuits and simulate measurements (we'll use it again in the next section about TensorFlow Quantum). Cirq is a Python library that can be installed using ``pip install cirq``. The following code implements the two simple quantum circuits established in the previous section, with 20 simulated measurements each.

.. literalinclude:: /_static/code/zh/appendix/quantum/basic.py

The results are as follows.

::

    0: ───X───M───
    0=11111111111111111111
    0: ───H───M───
    0=00100111001111101100

It can be seen that the first measurement of the quantum circuit is always 1, and the second quantum state has 9 out of 20 measurements of 0 and 11 of 1 (if you run it a few more times, you will find that the probability of 0 and 1 appearing is close to :math:`\frac{1}{2}`). The results can be seen to be consistent with our analysis in the previous section.

Hybrid Quantum - Classical Machine Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section introduces the basic concepts of hybrid quantum-classical machine learning and methods for building such models using TensorFlow Quantum.

In hybrid quantum-classical machine learning, we train hybrid quantum-classical models using quantum datasets. The first half of the hybrid quantum-classical model is the quantum model (i.e., the parameterized quantum circuit). The quantum model accepts the quantum dataset as input, transforms the input using quantum gates, and then transforms it into classical data by measurement. The measured classical data is fed into the classical model and the loss value of the model is calculated using the regular loss function. Finally, the gradient of the model parameters is calculated and updated based on the value of the loss function. This process includes not only the parameters of the classical model, but also parameters of the quantum model. The process is shown in the figure below.

.. figure:: /_static/image/quantum/pipeline.png
    :width: 60%
    :align: center

    Classical machine learning (above) vs. hybrid quantum-classical machine learning (below) process

TensorFlow Quantum is an open source library that is tightly integrated with TensorFlow Keras to quickly build hybrid quantum-classical machine learning models and can be installed using ``pip install tensorflow-quantum``.

At the beginning, use the following code to import TensorFlow, TensorFlow Quantum and Cirq

.. code-block:: python

    import tensorflow as tf
    import tensorflow_quantum as tfq
    import cirq

.. admonition:: recommended reading

    Broughton, Michael, Guillaume Verdon, Trevor McCourt, Antonio J. Martinez, Jae Hyeon Yoo, Sergei V. Isakov, Philip Massey, et al. " `TensorFlow Quantum: A Software Framework for Quantum Machine Learning. <http://arxiv.org/abs/2003.02989>`_ " ArXiv:2003.02989 [Cond-Mat, Physics:Quant-Ph], March 5, 2020. (TensorFlow Quantum White Paper)

Quantum datasets and quantum gates with parameters
---------------------------------------------------

Using supervised learning as an example, the classical dataset consists of classical data and labels. Each item in the classical data is a vector composed of features. We can write the classical dataset as :math:`(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)` , where :math:`x_i = (x_{i,1}, \cdots, x_{i,K})` . A quantum dataset is also made up of data and labels, but each item in the data is a quantum state. Use the state of a single quantum bit in the previous section as example, we can write each data :math:`x_i = a_i \ket{0} + b_i \ket{1}` . In terms of implementation, we can generate quantum data through quantum circuits. That is, each data :math:`x_i` corresponds to a quantum circuit. For example, we can use Cirq to generate a set of quantum data using the following code.

.. code-block:: python

    q = cirq.GridQubit(0, 0)
    q_data = []
    for i in range(100):
        x_i = cirq.Circuit(
            cirq.rx(np.random.rand() * np.pi)(q)
        )
        q_data.append(x_i)

In this process, we use a quantum gate ``cirq.rx(angle)(q)`` with parameters. Unlike the quantum gate ``cirq.X(q)`` and ``cirq.H(q)`` which we used earlier, this quantum gate has an additional parameter, ``angle`` , which represents the angle (radiances) of the rotation of the quantum bit ``q`` around the x-axis of the Bloch sphere. The above code generates 100 quantum data items, each of which is randomly rotated :math:`[0, \pi]` around the x-axis of the Bloch sphere starting from the pure state :math:`\ket{0}`. Quantum datasets have applications in quite a few quantum-related fields such as chemistry, materials science, biology and drug discovery.

When we want to use the quantum data set as input to Keras, we can use the ``convert_to_tensor`` method of TensorFlow Quantum to convert the quantum dataset to a tensor.

.. code-block:: python

    q_data = tfq.convert_to_tensor(q_data)

It is worth noting that when using quantum datasets as training data for the Keras model, the input type (``dtype``) for the Keras model needs to be set to ``tf.dtypes.string``.

Parametric quantum circuit (PQC)
---------------------------------------------------

When we use a quantum gate with a parameter when building a quantum circuit and the parameter is freely adjustable, we call such a quantum circuit a parametric quantum circuit. Cirq supports parametric quantum circuits in combination with SymPy, a symbolic arithmetic library under Python, for example

.. code-block:: python

    import sympy
    theta = sympy.Symbol('theta')
    q_model = cirq.Circuit(cirq.rx(theta)(q))

In the code above, we built the quantum circuit shown in the following figure. The quantum circuit can rotate any input quantum state :math:`\ket{\psi}` counterclockwise around the x-axis of the Bloch sphere for :math:`\theta` radians, where :math:`\theta` is the symbolic variable (i.e. parameter) declared using ``sympy.Symbol``.

.. figure:: /_static/image/quantum/pqc.png
    :width: 30%
    :align: center

    Example of a parametric quantum circuit

Embedding parametric quantum circuits into machine learning models
-------------------------------------------------------------------

With TensorFlow Quantum, we can easily embed parametric quantum circuits into the Keras model as a Keras layer. For example, for the parameterized quantum circuit ``q_model`` created in the previous section, we can use ``tfq.layers.PQC`` as a Keras layer directly.

.. code-block:: python

    q_layer = tfq.layers.PQC(q_model, cirq.Z(q))
    expectation_output = q_layer(q_data_input)

The first parameter of ``tfq.layers.PQC`` is a parameterized quantum circuit established using Cirq and the second parameter is the measurement method, which is measured here using ``cirq.Z(q)`` on the Z axis of the Bloch sphere.

The above code can also be written directly as

.. code-block:: python

    expectation_output = tfq.players.PQC(q_model, cirq.Z(q))(q_data_input)

Example: binary classification of quantum datasets
---------------------------------------------------

In the following code, we first build a quantum dataset where half of the data items are pure state :math:`\ket{0}` rotating counterclockwise around the x-axis of the Bloch sphere :math:`\frac{\pi}{2}` radians (i.e. :math:`\frac{1}{\sqrt{2}} \ket{0} - \frac{i}{\sqrt{2}} \ket{1}`) and the other half are :math:`\frac{3\pi}{2}` radians (i.e. :math:`\frac{1}{\sqrt{2}} \ket{0} + \frac{i}{\sqrt{2}} \ket{1}`). All data were added Gaussian noise rotated around the x and y axis with a standard deviation of :math:`\frac{\pi}{4}`. For this quantum dataset, if measured directly without transformation, all the data would be randomly collapsed to the pure states :math:`\ket{0}` and :math:`\ket{1}` with the same probability as a coin toss, making it indistinguishable.

To distinguish between these two types of data, we next build a quantum model that rotates the single-bit quantum state counterclockwise around the x-axis of the Bloch sphere :math:`\theta` radians. The measured values of the transformed quantum data are fed into the classical machine learning model with a fully Connected layer and softmax function, using cross-entropy as a loss function. The model training process automatically adjusts both the value of :math:`\theta` in the quantum model and the weights in the fully connection layer, resulting in higher accuracy of the entire hybrid quantum-classical machine learning model.

.. literalinclude:: /_static/code/zh/appendix/quantum/binary_classification.py

Output:

::

    ..
    200/200 [==========================================] - 0s 165us/sample - loss: 0.1586 - sparse_categorical_accuracy: 0.9500
    [array([-1.5279944], dtype=float32)]

It can be seen that the model can achieve 95% accuracy on the training set after training, and :math:`\theta = -1.5279944 \approx -\frac{\pi}{2} = -1.5707963...` . When :math:`\theta = -\frac{\pi}{2}` , it happens that the two class of data items are close to the pure states :math:`\ket{0}` and :math:`\ket{1}`, respectively, so that they can be most easily distinguishable.

.. [#f0] This handbook is written in 2020 AD, so if you are from the future, please understand the limitations of the author's time.
.. [#f1] The term "collapse" is mostly used in the Copenhagen interpretation of quantum observations. There are also other interpretations like multiverse theory. The word "collapse" is used here for convenience only.
.. [#f2] Actually the more common binary quantum gates are "Quantum Controlled NOT Gate" (CNOT) and the "Quantum Swap Gate" (SWAP).
.. [#f3] This matrix is known as `unitary matrix <https://en.wikipedia.org/wiki/Unitary_matrix>`_ .

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 361 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>