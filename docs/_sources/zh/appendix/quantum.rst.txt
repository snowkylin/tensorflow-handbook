TensorFlow Quantum: 混合量子-经典机器学习 *
============================================================

我们身边的经典计算机利用比特位和逻辑门进行二进制运算。在物理硬件上，这种运算主要是通过半导体的特殊导电性质实现的。经过几十年的发展，我们已经可以在一片小小的半导体芯片上集成上亿个晶体管，从而实现高性能的经典计算。

而量子计算（Quantum Computing）旨在利用具有量子特性（例如量子态叠加和量子纠缠）的“量子比特位”和“量子逻辑门”进行计算。这种新的计算模式可以在搜索和大数分解等重要领域达成指数级的加速，让当前无法实现的一些超大规模运算成为可能，从而可能在未来深远地改变世界。在物理硬件上，这类量子运算也可以通过一些具有量子特性的结构（例如超导约瑟夫森结）实现。

不幸的是，尽管量子计算的理论已经有了比较深入的发展，可在物理硬件上，我们目前仍然造不出一台超越经典计算机的通用量子计算机 [#f0]_ 。IBM和谷歌等业界巨头在通用量子计算机的物理构建上已经取得了一些成绩，但无论是量子比特的个数还是在退相干问题的解决上，都还远无法达到实用的层级。

以上是量子计算的基本背景，接下来我们讨论量子机器学习。量子机器学习的一种最直接的思路是使用量子计算加速传统的机器学习任务，例如量子版本的PCA、SVM和K-Means算法，然而这些算法目前都尚未达到可实用的程度。我们在本章讨论的量子机器学习则采用另一种思路，即构建参数化的量子电路（Parameterized Quantum Circuits, PQCs）。PQC可以作为深度学习模型中的层而被使用，如果我们在普通深度学习模型的基础上加入PQC，即称为混合量子-经典机器学习（Hybrid Quantum-Classical Machine Learning）。这种混合模型尤其适合于量子数据集（Quantum Data）上的任务。而TensorFlow Quantum正是帮助我们构建这种混合量子-经典机器学习模型的利器。接下来，我们会对量子计算的若干基本概念进行简介，然后介绍使用TensorFlow Quantum构建PQC、将PQC嵌入Keras模型、并在量子数据集上训练混合模型的流程。

..
    https://www.tensorflow.org/quantum
    https://mp.weixin.qq.com/s?__biz=MzU1OTMyNDcxMQ==&mid=2247487901&idx=2&sn=bf00bbc09e5e1f415d1809d6333b5d5b&chksm=fc185ad5cb6fd3c3e7f77e9ccfa77b1aae083ab033b43711e84ee7f09b4ea7b0c4dbad5bfdfb&mpshare=1&scene=23&srcid=&sharer_sharetime=1585490090816&sharer_shareid=b6f86ab8b392c4d4036aa6a1d3b82824#rd


量子计算基本概念
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本节将简述量子计算的一些基本概念，包括量子比特、量子门、量子电路等。

.. admonition:: 推荐阅读

    如果你希望更深入地了解量子力学以及量子计算的基本原理，建议可以从以下两本书入手：

    -  `吴飚 <http://www.phy.pku.edu.cn/~wubiao/>`_ ，简明量子力学（简洁明快的量子力学入门教程，即将由北京大学出版社出版，可先行阅读在线版本 http://www.phy.pku.edu.cn/~wubiao/pop_qm_pkupress.pdf ）
    - Hidary, Jack D. `Quantum Computing: An Applied Approach <http://link.springer.com/10.1007/978-3-030-23922-0>`_ . Cham: Springer International Publishing, 2019. https://doi.org/10.1007/978-3-030-23922-0. （注重代码实操的量子计算教程，GitHub上有配套源码： https://github.com/JackHidary/quantumcomputingbook ）


量子比特
-------------------------------------------

在二进制的经典计算机中，我们用比特（Bit，也称“位”）作为信息存储的基本单位，一个比特只有0或者1两种状态。而在量子计算机中，我们使用量子比特（Quantum Bit, Qubit，也称“量子位”）进行信息的表示。量子比特也有两种基本状态 :math:`\ket{0}` 和 :math:`\ket{1}` 。不过量子比特除了可以处于这两种基本状态以外，还可以处于两者之间的叠加态（Superposition State），即 :math:`\ket{\psi} = a \ket{0} + b \ket{1}` （其中a和b是复数， :math:`|a|^2 + |b|^2 = 1` ）。例如， :math:`\ket{\psi_0} = \frac{1}{\sqrt{2}} \ket{0} + \frac{1}{\sqrt{2}} \ket{1}` 和 :math:`\ket{\psi_1} = \frac{1}{\sqrt{2}} \ket{0} - \frac{1}{\sqrt{2}} \ket{1}` 都是合法的量子态。我们也可以使用向量化的语言来表示量子比特的状态。如果我们令 :math:`\ket{0} = \begin{bmatrix}1 \\ 0\end{bmatrix}` 、 :math:`\ket{1} = \begin{bmatrix}0 \\ 1\end{bmatrix}`，则 :math:`\ket{\psi} = \begin{bmatrix}a \\ b\end{bmatrix}`、:math:`\ket{\psi_0} = \begin{bmatrix}\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}}\end{bmatrix}`、:math:`\ket{\psi_1} = \begin{bmatrix}\frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}}\end{bmatrix}`。

同时，我们可以用布洛赫球面（Bloch Sphere）来形象地展示单个量子比特的状态。球面的最顶端为 :math:`\ket{0}` ，最底端为 :math:`\ket{1}` ，而从原点到球面上任何一点的单位向量都可以是一个量子比特的状态。

.. figure:: /_static/image/quantum/bloch_sphere.png
    :width: 30%
    :align: center

    布洛赫球面（ `Bloch Sphere <https://en.wikipedia.org/wiki/Bloch_sphere>`_ ）。其中Z轴正负方向的量子态分别为基本态 :math:`\ket{0}` 和 :math:`\ket{1}` ，X轴正负方向的量子态分别为 :math:`\frac{1}{\sqrt{2}} \ket{0} + \frac{1}{\sqrt{2}} \ket{1}` 和 :math:`\frac{1}{\sqrt{2}} \ket{0} - \frac{1}{\sqrt{2}} \ket{1}` ，Y轴正负方向的量子态分别为 :math:`\frac{1}{\sqrt{2}} \ket{0} + \frac{i}{\sqrt{2}} \ket{1}` 和 :math:`\frac{1}{\sqrt{2}} \ket{0} - \frac{i}{\sqrt{2}} \ket{1}` 。`图示来源 <https://en.wikipedia.org/wiki/File:Bloch_sphere.svg>`_ 

值得一提的是，尽管量子比特 :math:`\ket{\psi} = a \ket{0} + b \ket{1}` 可能的状态相当之多，但一旦我们对其进行观测，则其状态会立即坍缩 [#f1]_ 到 :math:`\ket{0}` 和 :math:`\ket{1}` 这两个基本状态中的一个，其概率分别为 :math:`|a|^2` 和  :math:`|b|^2` 。

量子逻辑门
-------------------------------------------

在二进制的经典计算机中，我们有AND（与）、OR（或）、NOT（非）等逻辑门，对输入的比特状态进行变换并输出。在量子计算机中，我们同样有量子逻辑门（Quantum Logic Gate，或简称“量子门”），对量子状态进行变换并输出。如果我们使用向量化的语言来表述量子状态，则量子逻辑门可以看作是一个对状态向量进行变换的矩阵。

例如，量子非门可以表述为 :math:`X = \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}` ，于是当我们将量子非门作用于基本态 :math:`\ket{0} = \begin{bmatrix}1 \\ 0\end{bmatrix}` 时，我们得到 :math:`X\ket{0} = \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix} \begin{bmatrix}1 \\ 0\end{bmatrix} = \begin{bmatrix}0 \\ 1\end{bmatrix} = \ket{1}`。量子门也可以作用在叠加态，例如 :math:`X\ket{\psi_0} = \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix} \begin{bmatrix}\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}}\end{bmatrix} = \begin{bmatrix}\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}}\end{bmatrix} = \ket{\psi_0}` （这说明量子非门没能改变量子态 :math:`\ket{\psi_0} = \frac{1}{\sqrt{2}} \ket{0} + \frac{1}{\sqrt{2}} \ket{1}` 的状态。事实上，量子非门 :math:`X` 相当于在布洛赫球面上将量子态绕X轴旋转180度。而 :math:`\ket{\psi_0}` 就在X轴上，所以没有变化）。量子与门和或门 [#f2]_ 由于涉及到多个量子位而稍显复杂，但同样可以通过尺寸更大的矩阵实现。

可能有些读者已经想到了，既然单个量子比特的状态不止 :math:`\ket{0}` 和 :math:`\ket{1}` 两种，那么量子逻辑门作为作为对量子比特的变换，其实也完全可以不局限于与或非。事实上，只要满足一定条件的矩阵都可以作为量子逻辑门。例如，将量子态在布洛赫球面上绕X、Y、Z轴旋转的变换 :math:`Rx(\theta)` 、:math:`Ry(\theta)` 、:math:`Rz(\theta)` （其中 :math:`\theta` 是旋转角度，当 :math:`\theta=180^\circ` 时记为 :math:`X` 、:math:`Y` 、:math:`Z` ）都是量子逻辑门。另外，有一个量子逻辑门“阿达马门”（Hadamard Gate） :math:`H = \frac{1}{\sqrt{2}} \begin{bmatrix}1 & 1 \\ 1 & -1\end{bmatrix}` 可以将量子状态从基本态转换为叠加态，在很多量子计算的场景中占据了重要地位。

量子电路
-------------------------------------------

混合量子-经典机器学习
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: 推荐阅读

    Broughton, Michael, Guillaume Verdon, Trevor McCourt, Antonio J. Martinez, Jae Hyeon Yoo, Sergei V. Isakov, Philip Massey, et al. “ `TensorFlow Quantum: A Software Framework for Quantum Machine Learning. <http://arxiv.org/abs/2003.02989>`_ ” ArXiv:2003.02989 [Cond-Mat, Physics:Quant-Ph], March 5, 2020. （TensorFlow Quantum 白皮书）



参数化的量子电路（PQC）
-------------------------------------------

将参数化的量子电路嵌入机器学习模型
-------------------------------------------

量子数据集
-------------------------------------------

实例：对量子数据集进行二分类
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [#f0] 此手册的行文时间为公元2020年，如果你来自未来，请理解作者的时代局限性。
.. [#f1] “坍缩”一词多用于量子观测的哥本哈根诠释，除此以外还有多世界理论等。此处使用“坍缩”一词仅是方便表述。
.. [#f2] 其实更多讨论的基础二元量子门是“量子选择非门”（CNOT）和“交换门”（SWAP）。

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