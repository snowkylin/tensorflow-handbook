TensorFlow Quantum: 混合量子-经典机器学习 *
============================================================

我们身边的经典计算机利用比特位和逻辑门进行二进制运算。在物理硬件上，这种运算主要是通过半导体的特殊导电性质实现的。经过几十年的发展，我们已经可以在一片小小的半导体芯片上集成上亿个晶体管，从而实现高性能的经典计算。

而量子计算（Quantum Computing）旨在利用具有量子特性（例如量子态叠加和量子纠缠）的“量子比特位”和“量子逻辑门”进行计算。这种新的计算模式可以在搜索和大数分解等重要领域达成指数级的加速，让当前无法实现的一些超大规模运算成为可能，从而可能在未来深远地改变世界。在物理硬件上，这类量子运算也可以通过一些具有量子特性的结构（例如超导约瑟夫森结）实现。

然而不幸的是，尽管量子计算的理论已经有了比较深入的发展，可在物理硬件上，我们目前仍然造不出一台超越经典计算机的通用量子计算机 [#f0]_ 。IBM和谷歌等业界巨头在通用量子计算机的物理构建上已经取得了一些成绩，但无论是量子比特的个数还是在退相干问题的解决上，都还远无法达到实用的层级。

以上是量子计算的基本背景，接下来我们讨论量子机器学习。量子机器学习的一种最直接的思路是使用量子计算加速传统的机器学习任务，例如量子版本的PCA、SVM和K-Means算法，然而这些算法目前都尚未达到可实用的程度。我们在本章讨论的量子机器学习则采用另一种思路，即构建参数化的量子电路（Parameterized Quantum Circuits, PQCs）。PQC可以作为深度学习模型中的层而被使用，如果我们在普通深度学习模型的基础上加入PQC，即称为混合量子-经典机器学习（Hybrid Quantum-Classical Machine Learning）。这种混合模型尤其适合于量子数据集（Quantum Data）上的任务。而TensorFlow Quantum正是帮助我们构建这种混合量子-经典机器学习模型的利器。接下来，我们会对量子计算的若干基本概念进行简介，然后介绍使用TensorFlow Quantum构建PQC、将PQC嵌入Keras模型、并在量子数据集上训练混合模型的流程。

..
    https://www.tensorflow.org/quantum
    https://mp.weixin.qq.com/s?__biz=MzU1OTMyNDcxMQ==&mid=2247487901&idx=2&sn=bf00bbc09e5e1f415d1809d6333b5d5b&chksm=fc185ad5cb6fd3c3e7f77e9ccfa77b1aae083ab033b43711e84ee7f09b4ea7b0c4dbad5bfdfb&mpshare=1&scene=23&srcid=&sharer_sharetime=1585490090816&sharer_shareid=b6f86ab8b392c4d4036aa6a1d3b82824#rd


量子计算基本概念
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本节将简述量子计算的一些基本概念，包括量子位、量子门、量子电路等。

.. admonition:: 推荐阅读

    如果你希望更深入地了解量子力学以及量子计算的基本原理，建议可以从以下两本书开始：

    -  `吴飚 <http://www.phy.pku.edu.cn/~wubiao/>`_ ，简明量子力学（简洁明快的量子力学入门教程，即将由北京大学出版社出版，可先行阅读在线版本 http://www.phy.pku.edu.cn/~wubiao/pop_qm_pkupress.pdf ）
    - Hidary, Jack D. `Quantum Computing: An Applied Approach <http://link.springer.com/10.1007/978-3-030-23922-0>`_ . Cham: Springer International Publishing, 2019. https://doi.org/10.1007/978-3-030-23922-0. （注重代码实操的量子计算教程，GitHub上有配套源码： https://github.com/JackHidary/quantumcomputingbook ）


量子位
-------------------------------------------

量子门
-------------------------------------------

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