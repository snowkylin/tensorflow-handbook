TensorFlow Basic
======================

.. 
    https://www.datacamp.com/community/tutorials/tensorflow-tutorial
    
    As the name suggests, TensorFlow is a procedure which makes tensors flow. The tensor, like a multidimensional array, is a generalization of the vector (one dimensional) and the matrix (two dimensional), while the flows of tensors are based on the Dataflow Graph, also called the Computation Graph. A typical TensorFlow program consists of the following parts:

    1. Define a Dataflow Graph (usually called a 'model' in deep learning), which consists of large numbers of variables (called 'trainable parameters');
    2. Repeat the following steps:

    1. Convert the training data into tensors and input them into the Dataflow Graph for calculation (forward propagation);
    #. Evaluate the loss function and compute its partial derivatives for each variable (backward propagation);
    #. Use gradient descent or other optimizers to update variables in order to reduce the value of the loss function (i.e. training parameters).
    
	After enough times (and time) for repetition in step 2, the loss function will decrease to a very small value, indicating the completion of the model training.

    Before introducing a variety of concepts in TensorFlow such as Tensor, Dataflow Graph, Variable, Optimizer and so on, we give an example first in this handbook so as to provide readers with an intuitive comprehension.

This chapter describes basic operations in TensorFlow.

Prerequisites:

* `Basic Python operations <https://docs.python.org/3/tutorial/>`_ (assignment, branch & loop statement, library import)
* `'With' statement in Python <https://docs.python.org/3/reference/compound_stmts.html#the-with-statement>`_
* `NumPy <https://docs.scipy.org/doc/numpy/user/quickstart.html>`_ , a common library for scientific computation, important for TensorFlow
* `Vectors <https://en.wikipedia.org/wiki/Euclidean_vector>`_ & `Matrices <https://en.wikipedia.org/wiki/Matrix_(mathematics)>`_ operations (matrix addition & subtraction, matrix multiplication with vectors & matrices, matrix transpose, etc., Quiz: :math:`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = ?`)
* `Derivatives of functions <https://en.wikipedia.org/wiki/Derivative>`_ , `derivatives of multivariable functions <https://en.wikipedia.org/wiki/Partial_derivative>`_ (Quiz: :math:`f(x, y) = x^2 + xy + y^2, \frac{\partial f}{\partial x} = ?, \frac{\partial f}{\partial y} = ?`)
* `Linear regression <https://en.wikipedia.org/wiki/Linear_regression>`_;
* `Gradient descent <https://en.wikipedia.org/wiki/Gradient_descent>`_ that searches local minima of a function

TensorFlow 1+1
^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow can be simply regarded as a library of scientific calculation (like Numpy in Python). Here we calculate :math:`1+1` and :math:`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}` as our first example.

.. literalinclude:: ../_static/code/en/basic/eager/1plus1.py  

Output::
    
    tf.Tensor(2, shape=(), dtype=int32)
    tf.Tensor(
    [[19 22]
    [43 50]], shape=(2, 2), dtype=int32)

The code above declares four **tensors** named ``a``, ``b``, ``A`` and ``B``. It also invokes two **operations** ``tf.add()`` and ``tf.matmul()`` which respectively do addition and matrix multiplication on tensors. Operation results are immediately stored in the tensors ``c`` and ``C``. **Shape** and **dtype** are two major attributes of a tensor. Here ``a``, ``b`` and ``c`` are scalars with null shape and int32 dtype, while ``A``, ``B``, ``C`` are 2-by-2 matrices with ``(2, 2)`` shape and int32 dtype.

In machine learning, it's common to differentiate functions. TensorFlow provides us with the powerful **Automatic Differentiation Mechanism** for differentiation. The following codes show how to utilize ``tf.GradientTape()`` to get the slope of :math:`y(x) = x^2` at :math:`x = 3`.

.. literalinclude:: ../_static/code/en/basic/eager/grad.py  
    :lines: 1-8

Output::
    
    [array([9.], dtype=float32), array([6.], dtype=float32)]

Here ``x`` is a **variable** initialized to 3, declared by ``tf.get_variable()``. Like common tensors, variables also have shape and dtype attributes, but require an initialization. We can assign an initializer to ``tf.get_variable()`` by setting the ``Initializer`` parameter. Here we use ``tf.constant_initializer(3.)`` to initialize the variable ``x`` to ``3.`` with a float32 dtype. [#f0]_. An important difference between variables and common tensors is that a function can be differentiated by variables, not by tensors, using the automatic differentiation mechanism by default. Therefore variables are usually used as parameters defined in machine learning models. ``tf.GraidentTape()`` is a recorder of automatic differentiation which records all variables and steps of calculation automatically. In the previous example, the variable ``x`` and the calculation step ``y = tf.square(x)`` are recorded automatically, thus the derivative of the tensor ``y`` with respect to ``x`` can be obtained through ``y_grad = tape.gradient(y, x)``.

In machine learning, calculating the derivatives of a multivariable function, a vector or a matrix is a more common case, which is a piece cake for TensorFlow. The following codes show how to utilize ``tf.GradientTape()`` to differentiate :math:`L(w, b) = \|Xw + b - y\|^2` with respect to :math:`w` and :math:`b` at :math:`w = (1, 2)^T, b = 1`.

.. literalinclude:: ../_static/code/en/basic/eager/grad.py  
    :lines: 10-17

Output::

    [62.5, array([[35.],
       [50.]], dtype=float32), array([15.], dtype=float32)]

Here the operation ``tf.square()`` squares every element in the input tensor without altering its shape. The operation ``tf.reduce_sum()`` outputs the sum of all elements in the input tensor with a null shape (the dimensions of the summation can be indicated by the ``axis`` parameter, while all elements are summed up if not specified). TensorFlow contains a large number of tensor operation APIs including mathematical operations, tensor shape operations (like ``tf.reshape()``), slicing and concatenation (like ``tf.concat()``), etc. You can heck TensorFlow official API documentation [#f3]_ for further information.

As we can see from the output, TensorFlow helps us figure out that

.. math::

    L((1, 2)^T, 1) &= 62.5
    
    \frac{\partial L(w, b)}{\partial w} |_{w = (1, 2)^T, b = 1} &= \begin{bmatrix} 35 \\ 50\end{bmatrix}
    
    \frac{\partial L(w, b)}{\partial b} |_{w = (1, 2)^T, b = 1} &= 15

..
    By combining the automatic differentiation mechanism above with an **optimizer**, we can evaluate the extrema of a function. Here we use linear regression as an example (Evaluating :math:`\min_{w, b} L = (Xw + b - y)^2` essentially, :ref:`The next paragraph <linear-regression>` reveals the principles):

    .. literalinclude:: ../_static/code/en/basic/eager/regression.py  

.. _linear-regression:

A plain example: Linear regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's consider a practical problem. The house prices of a city between 2013 and 2017 are given by the following table:

======  =====  =====  =====  =====  =====
Year    2013   2014   2015   2016   2017
Price   12000  14000  15000  16500  17500
======  =====  =====  =====  =====  =====

Now we want to do linear regression on the given data, i.e. using the linear model :math:`y = ax + b` to fit the data, where ``a`` and ``b`` are unknown parameters.

First, we define and normalize the data.

.. literalinclude:: ../_static/code/en/basic/example/numpy.py
    :lines: 1-7

Then, we use gradient descent to evaluate these two parameters ``a`` and ``b`` in the linear model [#f1]_.

Recalling from the fundamentals of machine learning, for searching local minima of a multivariable function :math:`f(x)`, we use `gradient descent <https://en.wikipedia.org/wiki/Gradient_descent>`_ which taking the following steps:

* Initialize the argument to :math:`x_0` and have :math:`k=0`
* Iterate the following steps repeatedly till the convergence criteria is met:

    * Find the gradient of the function :math:`f(x)` with respect to the parameter :math:`\nabla f(x_k)`
    * Update the parameter :math:`x_{k+1} = x_{k} - \gamma \nabla f(x_k)` where :math:`\gamma` is the learning rate (like the step size of the gradient descent)
    * :math:`k \leftarrow k+1` 

Next we focus on how to implement gradient descent in order to solve the linear regression :math:`\min_{a, b} L(a, b) = \sum_{i=1}^n(ax_i + b - y_i)^2`.

NumPy
-----------------------

The implementation of machine learning models is not a patent of TensorFlow. In fact, even most common scientific calculators or tools can solve simple models. Here, we use Numpy, a general library for scientific computation, to implement gradient descent. Numpy supports multidimensional arrays to represent vectors, matrices and tensors with more dimensions. Meanwhile, it also supports lots of operations on multidimensional arrays (e.g. ``np.dot()`` calculates the inner products and ``np.sum()`` adds up all the elements). In this way Numpy is somewhat like MATLAB. In the following codes, we evaluate the partial derivatives of loss function with respect to the parameters ``a`` and ``b`` manually [#f2]_, and then iterate by gradient descent to acquire the value of ``a`` and ``b`` eventually.

.. literalinclude:: ../_static/code/en/basic/example/numpy.py
    :lines: 9-

However, you may have noticed that there are several pain points using common libraries for scientific computation to implement machine learning models:

- It's often inevitable to differentiate functions manually. Simple ones may be fine, however the more complex ones (especially commonly appeared in deep learning models) are another story. Manual differentiation may be painful, even infeasible in the latter cases.
- It's also often inevitable to update parameters based on the gradients manually. Manual update is still easy here because the gradient descent is a rather basic method while it's not going to be easy anymore if we apply a more complex approach to update parameters (like Adam or Adagrad).

However, the appearance of TensorFlow eliminates these pain points to a large extent, granting users convenience for implementing machine learning models.

TensorFlow
--------------------------------------------------------

The **Eager Execution Mode** of TensorFlow [#f4]_ have very similar operations as the above-mentioned Numpy. In addition, it also provides us with a series of critical functions for deep learning such as faster operation speed (need support from GPU), automatic differentiation and optimizers, etc. We will show how to do linear regression using Tensorflow. You may notice that its code structure is similar to the one of Numpy. Here we delegates TensorFlow to do two important jobs:

* Using ``tape.gradient(ys, xs)`` to get the gradients automatically;
* Using ``optimizer.apply_gradients(grads_and_vars)`` to update parameters automatically.

.. literalinclude:: ../_static/code/en/basic/example/tensorflow_eager_autograd.py
    :lines: 12-29

Here, we use the aforementioned approach to calculate the partial derivatives of the loss function with respect to each parameter, while we also use ``tf.train.GradientDescentOptimizer(learning_rate=1e-3)`` to declare an **optimizer** for graident descent with a learning rate of 1e-3. The optimizer can help us update parameters based on the result of differentiation in order to minimize a specific loss function by calling its ``apply_gradients()`` interface.

Note that, for calling ``optimizer.apply_gradients()`` to update model parameters, we need to provide it with parameters ``grads_and_vars``, i.e. the variables to be updated (like ``variables`` in the aforementioned codes). To be specific, a Python list has to be passed, whose every element is a (partial derivative with respect to a variable, this variable) pair. For instance, ``[(grad_w, w), (grad_b, b)]`` is passed here. By executing ``grads = tape.gradient(loss, variables)`` we get partial derivatives of the loss function with respect to each variable recorded in ``tape``, i.e. ``grads = [grad_w, grad_b]``. Then we use ``zip()`` in Python to pair the elements in ``grads = [grad_w, grad_b]`` and  ``vars = [w, b]`` together respectively so as to get the required parameters.

In practice, we usually build much more complex models rather the linear model ``y_pred = tf.matmul(X, w) + b`` here which can be simply written in a single line. Therefore, we often write a model class and call it by ``y_pred = model(X)`` when needed. :doc:`The following chapter <models>` elaborates writing model classes.

..
    Chapter Summary
    ^^^^^^^^^^^^^^^^^^^^^^^


.. [#f0] We can add a decimal point after an integer to make it become a floating point number in Python. E.g. ``3.`` represents the floating point number ``3.0``.
.. [#f3] Mainly refer to `Tensor Transformations <https://www.tensorflow.org/versions/r1.9/api_guides/python/array_ops>`_ and `Math <https://www.tensorflow.org/versions/r1.9/api_guides/python/math_ops>`_. Note that the tensor operation API of TensorFlow is very similar to Numpy, thus one can get started on TensorFlow rather quickly if knowing about the latter.
.. [#f1] In fact there is an analytic solution for the linear regression. We use gradient descent here just for showing you how TensorFlow works.
.. [#f2] The loss function here is the mean square error :math:`L(x) = \frac{1}{2} \sum_{i=1}^5 (ax_i + b - y_i)^2` whose partial derivatives with respect to ``a`` and ``b`` are :math:`\frac{\partial L}{\partial a} = \sum_{i=1}^5 (ax_i + b - y) x_i` and :math:`\frac{\partial L}{\partial b} = \sum_{i=1}^5 (ax_i + b - y)`.
.. [#f4] The opposite of Eager Execution is Graph Execution that TensorFlow adopts before version 1.8 in Mar 2018. This handbook is mainly written for Eager Execution aiming at fast iterative development, however the basic usage of Graph Execution is also attached in the appendices in case of reference.

..  
    Tensors (Variables, Constants and Placeholders)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Sessions and Computation Graphs
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Automatic Differentiation and Optimizers
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Scope of variables
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ..  https://tensorflow.google.cn/versions/master/api_docs/python/tf/variable_scope

    Save, Restore and Persistence
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
