前言
======

2018年3月30日，Google在加州山景城举行了第二届TensorFlow Dev Summit开发者峰会，并宣布正式发布TensorFlow 1.8版本。作为国内的第一批机器学习领域的 `Google Developers Expert <https://developers.google.cn/community/experts>`_ （GDE），笔者有幸获得Google的资助亲临峰会现场，见证了这一具有里程碑式意义的新版本发布。众多新功能的加入和支持展示了TensorFlow的雄心壮志，即时执行（Eager Execution）模式（或称“动态图模式”）在这一版本中终于正式加入，并成为了入门TensorFlow的官方推荐模式。

    The easiest way to get started with TensorFlow is using Eager Execution.
    
    —— https://www.tensorflow.org/get_started/

在此之前，TensorFlow所基于的传统图执行模式与会话机制（Graph Execution and Session，或称“静态图模式”）的弊端，如入门门槛高、调试困难、灵活性差、无法使用Python原生控制语句等早已被开发者诟病许久。一些新的基于动态图机制的深度学习框架（如PyTorch）也横空出世，并以其易用性和快速开发的特性而占据了一席之地。尤其是在学术研究等需要快速迭代模型的领域，PyTorch等新兴深度学习框架已经成为主流。笔者所在的数十人的机器学习实验室中，竟只有笔者一人“守旧”地使用TensorFlow。然而，直到目前，市面上相关的TensorFlow相关的中文技术书籍及资料仍然基于传统的计算图与会话机制，让不少初学者（尤其是刚学过机器学习课程的大学生）望而却步。由此，在TensorFlow正式支持即时执行模式之际，有必要出现一本全新的技术手册，帮助初学者及需要快速迭代模型的研究者，以一个全新的角度快速入门TensorFlow。

同时，本手册还有第二个任务。市面上与TensorFlow相关的中文技术书籍大部分都以深度学习为主线，而将TensorFlow作为这些深度学习模型的实现方式。这样固然有体系完整的优点，然而对于已经对机器学习或深度学习理论有所了解，希望侧重于学习TensorFlow本身的读者而言，就显得不够友好。于是，笔者希望编写一本手册，以尽量精简的篇幅展示TensorFlow作为一个计算框架的主要特性，并弥补官方手册的不足，力图能让已经有一定机器学习/深度学习知识及编程能力的读者迅速上手TensorFlow，并在实际编程过程中可以随时查阅并解决实际问题。

本手册的主要特征有：

* 主要基于TensorFlow最新的Eager Execution（动态图）模式，以便于模型的快速迭代开发。但依然会包含传统的Graph Execution模式，代码上尽可能兼容两者；
* 定位以技术手册为主，编排以TensorFlow的各项概念和功能为核心，力求能够让TensorFlow开发者快速查阅。各章相对独立，不一定需要按顺序阅读；
* 代码实现均进行仔细推敲，力图简洁高效和表意清晰。模型实现均统一使用 `TensorFlow官方文档 <https://www.tensorflow.org/programmers_guide/eager#build_a_model>`_ 最新提出的继承 ``tf.keras.Model`` 和 ``tf.keras.layers.Layer`` 的方式，保证代码的高度可复用性。每个完整项目的代码总行数均不过百行，让读者可以快速理解并举一反三；
* 注重详略，少即是多，不追求巨细靡遗和面面俱到，不在正文中进行大篇幅的细节论述。

本书的适用群体
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本书适用于以下读者：

* 已有一定机器学习/深度学习基础，希望将所学理论知识使用TensorFlow进行具体实现的学生和研究者；
* 曾使用或正在使用TensorFlow 1.X版本或其他深度学习框架（比如PyTorch），希望了解TensorFlow 2.0新特性的开发者；
* 希望将已有的TensorFlow模型应用于业界的开发者或工程师。

.. hint:: 本书不是一本机器学习/深度学习原理入门手册。若希望进行机器学习/深度学习理论的入门学习，可参考 :doc:`附录 <appendix/recommended_books>` 中提供的一些入门资料。

如何使用本书
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于已有一定机器学习/深度学习基础，着重于模型建立与训练的学生和研究者，建议顺序阅读本书的“基础”部分。对于希望将TensorFlow模型部署到实际环境中的开发者和工程师，则建议重点阅读本书的“部署”部分。部分章节的开头提供了“前置知识”部分，方便读者查漏补缺。部分周边补充知识以可折叠信息框的方式给出，可随时使用页面最上方的“折叠全部注释”按钮折叠所有信息框的内容。

在整本手册中，带“*”的部分均为选读。

本书的所有示例代码可至 https://github.com/snowkylin/tensorflow-handbook/tree/master/source/_static/code 获得。其中 ``zh`` 目录下是含中文注释的代码， ``en`` 目录下是含英文版注释的代码。在使用时，建议将代码根目录加入到 ``PYTHONPATH`` 环境变量，或者使用合适的IDE（如PyCharm）打开代码根目录，从而使得代码间的相互调用（形如 ``import zh.XXX`` 的代码）能够顺利运行。

致谢
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本手册的暂定名称《简单粗暴TensorFlow》是向我的好友兼同学Chris Wu编写的《简单粗暴 :math:`\text{\LaTeX}` 》（https://github.com/wklchris/Note-by-LaTeX）致敬。该手册清晰精炼，是 :math:`\text{\LaTeX}` 领域不可多得的中文资料，也是我在编写这一技术文档时所学习的对象。本手册最初是在我的好友Ji-An Li所组织的深度学习研讨小组中，由我作为预备知识的讲义而编写和使用。好友们的才学卓著与无私分享的品格也是编写此拙作的重要助力。

本手册的TensorFlow.js和TensorFlow Lite章节分别由Huan Li和Jinpeng Zhu两位在JavaScript和Android领域有丰富履历的GDE和前GDE撰写，同时，Huan撰写了TensorFlow for Swift和TPU部分的介绍，Jinpeng提供了TensorFlow Hub的介绍。来自豆瓣阅读的王子阳也提供了关于Node.js和阿里云的部分示例代码和说明。在此特别表示感谢。

本手册的英文版由我的好友Zida Jin和Ming翻译，并由Ji-An Li和笔者审校（仍在进行中）。三位朋友牺牲了自己的大量宝贵时间翻译和校对本手册，同时Ji-An Li亦对本手册的教学内容和代码细节提供了诸多宝贵意见。我谨向好友们为本手册的辛勤付出致以衷心的感谢。

衷心感谢 Google 中国开发者关系团队和 TensorFlow 工程团队的成员及前成员们对本手册的编写所提供的帮助。其中，开发者关系团队的 Luke Cheng 在本手册编写过程中，全程提供思路启发和鼓励，且提供本手册在线版本的域名 `tf.wiki <https://tf.wiki>`_ ；开发者关系团队的 Soonson Kwon、Lily Chen、Wei Duan、Rui Li、Pryce Mu，TensorFlow 产品经理 Mike Liang，以及 TensorFlow 社群维护的小伙伴们为本手册宣传及推广提供了大力支持；TensorFlow 工程团队的 Tiezhen Wang 在本手册的工程细节方面提供了诸多建议和补充；TensorFlow 工程团队的 Shuangfeng Li 等其他工程师们为本手册提供了专业的审阅意见。同时感谢 TensorFlow 工程总监 Rajat Monga 和 Google AI 负责人 Jeff Dean 在社交媒体上对本手册的推荐与关注。

关于本手册的意见和建议，欢迎在 https://github.com/snowkylin/tensorflow-handbook/issues 提交。这是一个开源项目，您的宝贵意见将促进本手册的持续更新。

|

Google Developers Expert in Machine Learning

Xihan Li ( `snowkylin <https://snowkylin.github.io/>`_ )

2019年8月于燕园
