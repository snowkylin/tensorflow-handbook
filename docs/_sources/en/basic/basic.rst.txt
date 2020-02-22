TensorFlow Basic
================

This chapter describes basic operations in TensorFlow.

Prerequisites:

* `Basic Python operations <https://docs.python.org/3/tutorial/>`_ (assignments, branch & loop statements, importing libraries)
* The `'With' statement in Python <https://docs.python.org/3/reference/compound_stmts.html#the-with-statement>`_
* `NumPy <https://docs.scipy.org/doc/numpy/user/quickstart.html>`_ , a commonly used Python library for scientific computing. TensorFlow 2.X is integrated closely with NumPy.
* `Vectors <https://en.wikipedia.org/wiki/Euclidean_vector>`_ & `Matrices <https://en.wikipedia.org/wiki/Matrix_(mathematics)>`_ operations (matrix addition & subtraction, matrix multiplication with vectors & matrices, matrix transpose, etc., Quiz: :math:`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = ?`)
* `Derivatives of functions <https://en.wikipedia.org/wiki/Derivative>`_ , `derivatives of multivariable functions <https://en.wikipedia.org/wiki/Partial_derivative>`_ (Quiz: :math:`f(x, y) = x^2 + xy + y^2, \frac{\partial f}{\partial x} = ?, \frac{\partial f}{\partial y} = ?`)
* `Linear regression <https://en.wikipedia.org/wiki/Linear_regression>`_;
* `Gradient descent <https://en.wikipedia.org/wiki/Gradient_descent>`_ that searches local minima of a function.

TensorFlow 1+1
^^^^^^^^^^^^^^

In the beginning, we can simply regard TensorFlow as a library for scientific computing (like Numpy in Python).

First, let us import TensorFlow:

.. code-block:: python

    import tensorflow as tf

.. admonition:: Warning

    This handbook is based on the Eager Execution mode of TensorFlow. In TensorFlow 1.X, you MUST run ``tf.enable_eager_execution()`` after importing it to enable Eager Execution mode. In TensorFlow 2.X, the Eager Execution is default thus you do not need to run ``tf.enable_eager_execution()``. (However, if you want to disable it, you should run ``tf.compat.v1.disable_eager_execution()``.)

TensorFlow uses **tensors** as its basic elements of data. Tensors in TensorFlow are conceptually equal to multidimensional arrays. We can use them to describe scalars, vectors, matrices and so on. Here are some examples:

.. literalinclude:: /_static/code/en/basic/eager/1plus1.py  
    :lines: 3-11

A tensor have three important attributes: shape, data type and value. You can use the ``shape`` 、 ``dtype`` attribute and the ``numpy()`` method to fetch them. For example:

.. literalinclude:: /_static/code/en/basic/eager/1plus1.py  
    :lines: 13-17

.. admonition:: Tip

    Most of the TensorFlow API functions will infer the data type automatically from the input (``tf.float32`` in most cases). However, you can add the parameter ``dtype`` to assign the data type manually. For example, ``zero_vector = tf.zeros(shape=(2), dtype=tf.int32)`` will return a tensor with all elements in type of ``tf.int32``.

    The ``numpy()`` method of a tensor is to return a NumPy array whose value is equal to the value of the tensor.

There are lots of **operations** in TensorFlow so that we can obtain new tensors as the result of operations between given tensors. For example:

.. literalinclude:: /_static/code/en/basic/eager/1plus1.py  
    :lines: 19-20

After the operations, the value of ``C`` and ``D`` are::
    
    tf.Tensor(
    [[ 6.  8.]
     [10. 12.]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[19. 22.]
     [43. 50.]], shape=(2, 2), dtype=float32)

So we can see that we have successfully used ``tf.add()`` to compute :math:`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}`, and have used ``tf.matmul()`` to compute :math:`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\43 & 50 \end{bmatrix}`.

Automatic differentiation mechanism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In machine learning, we often need to compute derivatives of functions. TensorFlow provides the powerful **Automatic differentiation mechanism** for computing derivatives. The following codes show how to use ``tf.GradientTape()`` to computer the derivative of the function :math:`y(x) = x^2` at :math:`x = 3`:

.. literalinclude:: /_static/code/en/basic/eager/grad.py  
    :lines: 1-7

Output::
    
    [array([9.], dtype=float32), array([6.], dtype=float32)]

Here ``x`` is a **variable** initialized to 3, declared by ``tf.Variable()``. Same as an ordinary tensor, a variable also has three attributes: shape, data type and value. An initialization is required before using a variable, which can be specified by the parameters ``initial_value`` in ``tf.Variable()``. Here ``x`` is initialized to ``3.`` [#f0]_. One significant difference between the variables and the tensors is the former can be used to differentiate by the automatic differentiation mechanism of TensorFlow by default, which is often used to define parameters of ML models.

``tf.GradientTape()`` is an automatic differentiation recorder, in which variables and calculation steps are automatically recorded. In the previous example, the variable ``x`` and the step ``y = tf.square(x)`` were recorded automatically, thus the derivative of the tensor ``y`` with respect to the variable ``x`` can be obtained by ``y_grad = tape.gradient(y, x)``.

The more common case in machine learning is partial differentiation of multivariable functions as well as differentiation of vectors and matrices. TensorFlow can handle these as well. The following codes show how to obtain the partial derivative of the function :math:`L(w, b) = \|Xw + b - y\|^2` for :math:`w, b` respectively by ``tf.GradientTape()`` where :math:`X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix},  y = \begin{bmatrix} 1 \\ 2\end{bmatrix}`.

.. literalinclude:: /_static/code/en/basic/eager/grad.py  
    :lines: 9-16

Output::

    [62.5, array([[35.],
       [50.]], dtype=float32), array([15.], dtype=float32)]

``tf.square()`` here squared each element of the input tensor without altering its shape. ``tf.reduce_sum()`` summed up all the elements of the input tensor, outputing a scalar tensor with a none shape (the dimensions for sum can be specified by the parameter ``axis``, without which all elements will be summed up by default). There are a large number of tensor operation APIs in TensorFlow, including mathematical operations, tensor shape operations (e.g., ``tf.reshape()``), slicing and concatenation (e.g., ``tf.concat()``), etc. Further information can be acquired by viewing the TensorFlow official API documentaion [#f3]_.

From the output we can see TensorFlow has helped us obtained that

.. math::

    L((1, 2)^T, 1) &= 62.5
    
    \frac{\partial L(w, b)}{\partial w} |_{w = (1, 2)^T, b = 1} &= \begin{bmatrix} 35 \\ 50\end{bmatrix}
    
    \frac{\partial L(w, b)}{\partial b} |_{w = (1, 2)^T, b = 1} &= 15

..
    以上的自动求导机制结合 **优化器** ，可以计算函数的极值。这里以线性回归示例（本质是求 :math:`\min_{w, b} L = (Xw + b - y)^2` ，具体原理见 :ref:`后节 <linear-regression>` ）：

    .. literalinclude:: /_static/code/zh/basic/eager/regression.py  

.. _linear-regression:

A basic example: Linear regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Basics
    
    * UFLDL Tutorial, `Linear Regression <http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/>`_.

Consider a practical problem. The estate price of a city between 2013 and 2017 are listed below:

======  =====  =====  =====  =====  =====
Year    2013   2014   2015   2016   2017
Price   12000  14000  15000  16500  17500
======  =====  =====  =====  =====  =====

Now we wish to perform a linear regression on this data, that is, use the linar model :math:`y = ax + b` to fit the data above, where ``a`` and ``b`` are parameters yet to be determined.

First we define the data and conduct basic normalization.

.. literalinclude:: /_static/code/zh/basic/example/numpy_manual_grad.py
    :lines: 1-7

In the following steps we use gradient descent to find the parameters ``a`` and ``b`` in the linear model [#f1]_.

Recall the basic knowledge of machine learning, to find a local minimum of a multivariable function :math:`f(x)`, the process of `gradient descent <https://en.wikipedia.org/wiki/Gradient_descent>`_ is as follows:

* Initialize the independent variable to :math:`x_0`, :math:`k=0`.
* Iterate the following steps until the convergence criterion is met:

    * Find the gradient :math:`\nabla f(x_k)`  of the function :math:`f(x)` with respect to the independent variable.
    * Update the independent variable: :math:`x_{k+1} = x_{k} - \gamma \nabla f(x_k)` where :math:`\gamma` is the learning rate (i.e. the "stride" in one gradient descent).
    * :math:`k \leftarrow k+1`.

Next, we consider how to programme to implement the gradient descent method to find the solution of the linear regression :math:`\min_{a, b} L(a, b) = \sum_{i=1}^n(ax_i + b - y_i)^2`. 

Linear regression under numPy
-----------------------------

Implementations of ML models are not preserved for TensorFlow. In fact, simple models can be solved even by using regular scientific computing libraries. Here we use Numpy, the common scientific computing library to implement gradient descent. NumPy provides support for multidimensional arrays, which can represent vectors, matrices and even higher dimensional tensors. Meanwhile, it also provides many functions that support operations on multidimensional arrays (e.g. the following ``np.dot()`` evaluates the dot product and ``np.sum()`` gets the sum). NumPy and MATLAB are similar in this regard. In the following codes, we will find the partial derivative of the loss function with respect to the parameters ``a`` and ``b`` manually [#f2]_ and use gradient descent iteratively to obtain the values of ``a`` and ``b`` eventually.

.. literalinclude:: /_static/code/en/basic/example/numpy_manual_grad.py
    :lines: 9-

However, you may have already noticed that there are two pain points for implementing ML models when using conventional scientific computing libraries:

- You have to find the partial derivatives with respect to parameters by yourself often. It may be easy for simple functions, but the process would be very painful or even impossible once the functions become complex.
- You have to update the parameters according to the result of the derivative by yourself frequently. Here we used gradient descent, the most fundamental approach, thus it was not hard updating parameters. However, the process would have been very complicated if you use more advanced approaches updating parameters (e.g., Adam or Adagrad).

The emergence of DL frameworks such as TensorFlow has largely solved these problems and has brought considerable convenience for implementing ML models.

.. _optimizer:

Linear regression under TensorFlow
----------------------------------

TensorFlow **Eager Execution Mode** [#f4]_ is quite similar with how NumPy worked above, while it provides a series of features which are rather crucial for deep learning, such as faster computation (GPU support), automatic differentiation, optimizers, etc. The following shows how to use TensorFlow to compute linear regression. You can notice that the structure of the program is very similar with the previous implemention with NumPy. Here TensorFlow helps us accomplished two crucial tasks:

* Using ``tape.gradient(ys, xs)`` to compute the gradient automatically
* Using ``optimizer.apply_gradients(grads_and_vars)`` to update model parameters automatically

.. literalinclude:: /_static/code/en/basic/example/tensorflow_eager_autograd.py
    :lines: 10-29

Here we used the approach mentioned before to compute the partial derivative of the loss function with respect to parameters. Meanwhile, we declared a gradient descent **optimizer** whose learning rate was 1e-3 by ``tf.keras.optimizers.SGD(learning_rate=1e-3)``. The optimizer can help us update model parameters based on the calculated derivative result, thereby minimizing a certain loss function. Specifically, you should call the method ``apply_gradients()`` for doing so.

Notice here we needed to provide the parameter ``grads_and_vars``, which were the variables to be updated (like ``variables`` in the codes above) and the partial derivatives of the loss function with respect to them (like ``grads`` in the codes above), to the method ``optimizer.apply_gradients()`` that updated model paramters. Specifically, you need to pass in a Python list here whose elements are ``(the partial derivative for the variable, the variable)`` pairs, e.g., ``[(grad_a, a), (grad_b, b)]`` in this case. By ``grads = tape.gradient(loss, variables)`` we found the partial derivatives of ``loss`` with respect to each variable in ``variables = [a, b]`` recorded in tape, which are ``grads = [grad_a, grad_b]``. Then we used the ``zip()`` function in Python to assemble ``grads = [grad_a, grad_b]`` and ``variables = [a, b]`` together to get the parameters we needed.

.. admonition:: Python ``zip()`` function

    The ``zip()`` function is a built-in function of Python. It would be confounding to describe it with natural language, but it will be much more accessible by giving an example: If ``a = [1, 3, 5]`` and ``b = [2, 4, 6]``, then ``zip(a, b) = [(1, 2), (3, 4), ..., (5, 6)]``. In other words, it "takes iterable objects as parameters, packs their corresponding elements into tuples and returns a list of these tuples". In Python 3, the ``zip()`` function returns an object, which needs to be converted into a list by calling ``list()``.

    .. figure:: /_static/image/basic/zip.jpg
        :width: 60%
        :align: center

        Python ``zip()`` function diagram

In practical applications, the models we code are usually much more complicated than the linear model ``y_pred = a * X + b`` (whose paramters are ``variables = [a, b]``) which can be written in a single line. Therefore we will often create and instantiate a model class ``model = Model()``, then use ``y_pred = model(X)`` to call it and use ``model.variables`` to acquire model parameters. Refer to :doc:`chapter "TensorFlow Models" <models>` for writing model classes.

..
    Summary of this chapter
    ^^^^^^^^^^^^^^^^^^^^^^^

.. [#f0] In Python an integer can be defined in float type by adding a period after it. E.g., ``3.`` means the float ``3.0``.
.. [#f3] Refer to `Tensor Transformations <https://www.tensorflow.org/versions/r1.9/api_guides/python/array_ops>`_ and `Math <https://www.tensorflow.org/versions/r1.9/api_guides/python/math_ops>`_. Notice that tensor operations in TensorFlow are quite similar in form with the popular Python scientific computing library NumPy. You can get started quickly if you have already known about the latter.
.. [#f1] In fact, there has already been an analytical solution of linear regression. We used gradient descent here only for demonstrating how TensorFlow works.
.. [#f2] The loss function here is the mean squared error :math:`L(x) = \frac{1}{2} \sum_{i=1}^5 (ax_i + b - y_i)^2`, whose partial derivatives with respect to the parameters ``a`` and ``b`` are :math:`\frac{\partial L}{\partial a} = \sum_{i=1}^5 (ax_i + b - y) x_i` and :math:`\frac{\partial L}{\partial b} = \sum_{i=1}^5 (ax_i + b - y)`.
.. [#f4] The opposite of the Eager Execution mode is the Graph Execution mode, which is the primary mode of TensorFlow before version 1.8 published in March 2018. In this handbook we focus on the Eager Execution mode for rapid iterative development, but we will get to the Graph Execution mode in the appendix for readers in need.

..  
    Tensors (variables, constants and placeholders)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Sessions and graphs
    ^^^^^^^^^^^^^^^^^^^

    Automatic differentiation and optimizers
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Scopes of variables
    ^^^^^^^^^^^^^^^^^^^
    ..  https://tensorflow.google.cn/versions/master/api_docs/python/tf/variable_scope

    Saving, recovery and persistance
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^