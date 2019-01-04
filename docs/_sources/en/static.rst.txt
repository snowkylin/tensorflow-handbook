Appendix: Static TensorFlow
======================================

TensorFlow 1+1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Essentially, TensorFlow is a symbolic computational framework (based on computational graph). Here is a "Hello World" example of computing 1+1.

.. literalinclude:: ../_static/code/en/basic/graph/1plus1.py      

Output::
    
    2

The program above is capable of computing 1+1 only, the following program, however, shows how to use TensorFlow to compute the sum of any two numbers through the parameter ``feed_dict=`` of ``tf.placeholder()`` and ``sess.run()``:

.. literalinclude:: ../_static/code/en/basic/graph/aplusb.py      

Terminal::

    >>> a = 2
    >>> b = 3
    a + b = 5

**Variable** is a special type of tensor, which is built using ``tf.get_variable()``. Just like variables in common progamming language, a ``Variable`` should be initialized before used and its value can be modified during computation in the computational graph. The following example shows how to create a ``Variable``, initialize its value to 0, and increment by one.

.. literalinclude:: ../_static/code/en/basic/graph/variable.py

Output::

    1.0
    2.0
    3.0
    4.0
    5.0

The following code is equivalent to the code shown above. It specifies the initializer upon declaring variables and initializes all variables at once by ``tf.global_variables_initializer()``, which is used more often in practical projects:

.. literalinclude:: ../_static/code/en/basic/graph/variable_with_initializer.py

Matrix and tensor calculation is the basic operation in scientific computation (including Machine Learning). The program shown below is to demonstrate how to calculate the product of the two matrices :math:`\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}` and :math:`\begin{bmatrix} 1 & 1 \\ 1 & 1 \\ 1 & 1 \end{bmatrix}`:

.. literalinclude:: ../_static/code/en/basic/graph/AmatmulB.py

Output::

    [[3. 3.]
     [3. 3.]]

Placeholders and Variables are also allowed to be vector, matrix and even higher dimentional tensor.

A Basic Example: Linear Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike previous NumPy and Eager Execution mode, TensorFlow's Graph Execution mode uses **symbolic programming** for numerical operations. First, we need to abstract the computational processes into a Dataflow Graph, and represent the inputs, operations and outputs with symbolized nodes. Then, we continually send the data to the input nodes, let the data be calculated and flow along the dataflow graph, and finally reach the specific output nodes we want. The following code shows how to accomplish the same task as the code does in previous section based on TensorFlow's symbolic programming approach, where ``tf.placeholder()`` can be regarded as a kind of "symbolic input node", using ``tf.get_variable()`` to define the parameters of the model (the tensor of the Variable type can be assigned using ``tf.assign()``), and ``sess.run(output_node, feed_dict={input_node: data})`` can be thought of as a process which sends data to the input node, calculates along the dataflow graph and reach the output node and eventually return values.

.. literalinclude:: ../_static/code/en/basic/example/tensorflow_manual_grad.py
    :lines: 9-

In the two examples above, we manually calculated the partial derivatives of the loss function with regard to each parameter. But when both the model and the loss function become very complicated (especially in deep learning models), the workload of manual derivation is unacceptable. TensorFlow provides an **automatic derivation mechanism** that eliminates the hassle of manually calculating derivatives, using TensorFlow's derivation function ``tf.gradients(ys, xs)`` to compute the partial derivatives of the loss function with regard to a and b. Thus, the two lines of code in the previous section for calculating derivatives manually,

.. literalinclude:: ../_static/code/en/basic/example/tensorflow_manual_grad.py
    :lines: 21-23

could be replaced by

.. code-block:: python

    grad_a, grad_b = tf.gradients(loss, [a, b])

and the result won't change.

Moreover，TensorFlow has many kinds of **optimizer**, which can complete derivation and gradient update together at the same time. The code in the previous section,

.. literalinclude:: ../_static/code/en/basic/example/tensorflow_manual_grad.py
    :lines: 21-31

could be replaced by

.. code-block:: python

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_)
    grad = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grad)

Here, we first instantiate a gradient descent optimizer ``tf.train.GradientDescentOptimizer()`` in TensorFlow and set the learning rate. Then use its ``compute_gradients(loss)`` method to find the gradients of ``loss`` with regard to all variables (parameters). Finally, through the method ``apply_gradients(grad)``, the variables (parameters) are updated according to the previously calculated gradients.

These three lines of code are equivalent to the following line of code:

.. code-block:: python

    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_).minimize(loss)

The simplified code is as follows:

.. literalinclude:: ../_static/code/en/basic/example/tensorflow_autograd.py
    :lines: 9-29
