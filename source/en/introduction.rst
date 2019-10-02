TensorFlow Overview
===================

当我们在说“我想要学习一个深度学习框架”，或者“我想学习TensorFlow”、“我想学习TensorFlow 2.0”的时候，我们究竟想要学到什么？事实上，对于不同群体，可能会有相当不同的预期。

For students and researchers: To build and train models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果你是一个初学机器学习/深度学习的学生，你可能已经啃完了Andrew Ng的机器学习公开课或者斯坦福的 `UFIDL Tutorial <http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial>`_ ，亦或是正在上学校里的深度学习课程。你可能也已经了解了链式求导法则和梯度下降法，知道了若干种损失函数，并且对卷积神经网络（CNN）、循环神经网络（RNN）和强化学习的理论也有了一些大致的认识。然而——你依然不知道这些模型在计算机中具体要如何实现。这时，你希望能有一个程序库，帮助你把书本上的公式和算法运用于实践。

具体而言，以最常见的有监督学习（supervised learning）为例。假设你已经掌握了一个模型 :math:`\hat{y} = f(x, \theta)` （x、y为输入和输出， :math:`\theta` 为模型参数），确定了一个损失函数 :math:`L(y, \hat{y})` ，并获得了一批数据 :math:`X`  和相对应的标签 :math:`Y` 。这时，你会希望有一个程序库，帮助你实现下列事情：

- 用计算机程序表示出向量、矩阵和张量等数学概念，并方便地进行运算；
- 方便地建立模型 :math:`f(x, \theta)` 和损失函数 :math:`L(y, \hat{y}) = L(y, f(x, \theta))` 。给定输入 :math:`x_0 \in X` ，对应的标签 :math:`y_0 \in Y` 和当前迭代轮的参数值 :math:`\theta_0` ，能够方便地计算出模型预测值 :math:`\hat{y_0} = f(x_0, \theta_0)` ，并计算损失函数的值 :math:`L_0 = L(y_0, \hat{y_0}) = L(y_0, f(x_0, \theta_0))` ；
- 自动将损失函数 :math:`L` 求已知 :math:`x_0`、:math:`y_0`、:math:`\theta_0` 时对模型参数 :math:`\theta` 的偏导数值，即计算 :math:`\theta_0' = \frac{\partial L}{\partial \theta} |_{x = x_0, y = y_0, \theta = \theta_0}` ，无需人工推导求导结果（这意味着，这个程序库需要支持某种意义上的“符号计算”，表现在能够记录下运算的全过程，这样才能根据链式法则进行反向求导）；
- 根据所求出的偏导数 :math:`\theta_0'` 的值，方便地调用一些优化方法更新当前迭代轮的模型参数 :math:`\theta_0` ，得到下一迭代轮的模型参数 :math:`\theta_1` （比如梯度下降法， :math:`\theta_1 = \theta_0 - \alpha \theta_0'` ， :math:`\alpha` 为学习率）。

更抽象一些地说，这个你所希望的程序库需要能做到：

- 数学概念和运算的程序化表达；
- 对任意可导函数 :math:`f(x)` ，求在自变量 :math:`x = x_0` 给定时的梯度 :math:`\nabla f | _{x = x_0}` （“符号计算”的能力）。

For developers and engineers: To invoke and deploy models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果你是一位在IT行业沉淀多年的开发者或者工程师，你可能已经对大学期间学到的数学知识不再熟悉（“多元函数……求偏微分？那是什么东西？”）。然而，你可能希望在你的产品中加入一些与人工智能相关的功能，抑或需要将已有的深度学习模型部署到各种场景中。具体而言，包括：

* 如何导出训练好的模型？
* 如何在本机使用已有的预训练模型？
* 如何在服务器、移动端、嵌入式设备甚至网页上高效运行模型？
* ……

What can TensorFlow do for us?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow可以为以上的这些需求提供完整的解决方案。具体而言，TensorFlow包含以下特性：

* **模型的建立与调试：** 使用动态图模式Eager Execution和著名的神经网络高层API框架Keras，结合可视化工具TensorBoard，简易、快速地建立和调试模型；
* **模型的训练：** 支持CPU/单GPU/单机多卡GPU/多机集群/TPU训练模型，充分利用海量数据和计算资源进行高效训练；
* **模型的部署：** 通过TensorFlow Serving、TensorFlow Lite、TensorFlow.js等组件，使TensorFlow模型能够无缝地部署到服务器、移动端、嵌入式端和网页端等多种使用场景；
* **预训练模型调用：** 通过TensorFlow Hub和Tensor2Tensor，可以方便地调用预训练完毕的已有成熟模型。

