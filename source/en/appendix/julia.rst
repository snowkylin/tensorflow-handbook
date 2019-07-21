TensorFlow in Julia
=============================

Introduction of TensorFlow.jl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow.jl is Tensorflow 's Julia version, made by `malmaud <https://github.com/malmaud/>`_ 's packaging for the original Tensorflow.

As a wrapper for Tensorflow, TensorFlow.jl and the Python version of TensorFlow have similar APIs and support for GPU acceleration.

Why use julia for Tensorflow development?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although Julia has no effect on the operation of Tensorflow itself, TensorFlow.jl does have      advantages.

As a modern programming language for numerical computing, Julia has a series of advanced grammatical features. Excellent Just-In-Time allows you to extract data at high speed and process the output of Tensorflow. Thanks to Julia's grammar design, writing expressions are also More flexible and natural.

In this chapter we will introduce Tensorflow to Julia based on TensorFlow.jl 0.12. You can refer to the latest `TensorFlow.jl documentation <https://malmaud.github.io/TensorFlow.jl/stable/tutorial.html>`_.

Try TensorFlow.jl in docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have docker installed, it is very convenient to use the docker image pre-installed with TensorFlow.jl.

Execute ``docker run -it malmaud/julia:tf`` on the command line, 
then you can get a Julia REPL environment with TensorFlow.jl installed.
( If you don't want to open Julia directly, you can open a bash terminal by executing ``docker run -it malmaud/julia:tf /bin/bash``. 
To execute the julia code files you need, you can use dcoker's directory mapping.)

Install TensorFlow.jl in the julia package manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute ``julia`` on the command line to enter the Julia REPL environment, then execute the following command to install TensorFlow.jl

.. code-block:: julia

    using pkg
    Pkg.add("TensorFlow")


Basic Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: julia

    using TensorFlow

    # Define a Session
    sess = TensorFlow.Session()

    # Define a constant and a variable
    x = TensorFlow.constant([1])
    y = TensorFlow.Variable([2])

    # Define a calculation
    w = x + y

    # Performing the calculation process
    run(sess, TensorFlow.global_variables_initializer())
    res = run(sess, w)

    # Output result
    println(res)


MNIST number classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example comes from `TensorFlow.jl documentation <https://malmaud.github.io/TensorFlow.jl/stable/tutorial.html>`_,
can compare python version of API.

.. code-block:: julia

    # Load data using mnist_loader.jl in the example
    include(Pkg.dir("TensorFlow", "examples", "mnist_loader.jl"))
    loader = DataLoader()

    # Define a Session
    using TensorFlow
    sess = Session()


    # Building a softmax regression model
    x = placeholder(Float32)
    y_ = placeholder(Float32)
    W = Variable(zeros(Float32, 784, 10))
    b = Variable(zeros(Float32, 10))

    run(sess, global_variables_initializer())

    # Predicted Class and Loss Function
    y = nn.softmax(x*W + b)
    cross_entropy = reduce_mean(-reduce_sum(y_ .* log(y), axis=[2]))

    # Train the model
    train_step = train.minimize(train.GradientDescentOptimizer(.00001), cross_entropy)
    for i in 1:1000
        batch = next_batch(loader, 100)
        run(sess, train_step, Dict(x=>batch[1], y_=>batch[2]))
    end

    # Output results and evaluate models
    correct_prediction = indmax(y, 2) .== indmax(y_, 2)
    accuracy=reduce_mean(cast(correct_prediction, Float32))
    testx, testy = load_test_set()

    println(run(sess, accuracy, Dict(x=>testx, y_=>testy)))