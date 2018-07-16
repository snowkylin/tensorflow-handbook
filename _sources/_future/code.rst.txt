TensorFlow源码结构分析 *
==============================
..  https://tensorflow.google.cn/versions/master/extend/
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/BUILD
    https://groups.google.com/a/tensorflow.org/forum/?utm_medium=email&utm_source=footer#!msg/discuss/Fp0PEvZpSP0/-5u1WcKkAgAJ
        
    core/ contains the main C++ code and runtimes.

    core/ops/ contains the "signatures" of the operations
    core/kernels/ contains the "implementations" of the operations (including CPU and CUDA kernels)
    core/framework/ contains the main abstract graph computation and other useful libraries
    core/platform/ contains code that abstracts away the platform and other imported libraries (protobuf, etc)

    TensorFlow relies heavily on the Eigen library for both CPU and GPU calculations.  Though some GPU kernels are implemented directly with CUDA code.

    bazel builds certain C++ code using gcc/clang, and certain CUDA code (files with extension .cu.cc) with nvcc.

    python/ops/ contain the core python interface
    python/kernel_tests/ contain the unit tests and lots of example code
    python/framework/ contains the python abstractions of graph, etc, a lot of which get serialized down to proto and/or get passed to swigged session calls.
    python/platform/ is similar to the C++ platform, adding lightweight wrappers for python I/O, unit testing, etc.

    contrib/*/ directories generally mimic the root tensorflow path (i.e., they have core/ops/, etc)

    https://www.zhihu.com/question/41667903/answer/123150582

    源码我推荐几个python目录下非常值得看的基础类定义：
      framework/Ops.py：定义了Tensor、Graph、Opreator类等
      Ops/Variables.py：定义了Variable类

    https://becominghuman.ai/understanding-tensorflow-source-code-rnn-cells-55464036fc07


实例：添加一个新运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
..  https://tensorflow.google.cn/versions/master/extend/adding_an_op