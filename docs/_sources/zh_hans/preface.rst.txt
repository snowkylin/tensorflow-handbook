前言
======

2018年3月30日，Google在加州山景城举行了第二届TensorFlow Dev Summit开发者峰会，并宣布正式发布TensorFlow 1.8版本。作为国内首批机器学习领域的谷歌开发者专家（ `Google Developers Expert <https://developers.google.cn/community/experts>`_ , GDE），我有幸获得Google的资助亲临峰会现场，见证了这一具有里程碑式意义的新版本发布。众多新功能的加入和支持展示了TensorFlow的雄心壮志，已经酝酿许久的即时执行模式（Eager Execution，或称“动态图模式”）在这一版本中也终于正式得到支持。

在此之前，TensorFlow 所基于的传统的图执行模式与会话机制（Graph Execution and Session，或称“静态图模式”）的弊端，如入门门槛高、调试困难、灵活性差、无法使用 Python 原生控制语句等，已被开发者诟病许久。一些新的基于即时执行模式的深度学习框架（如 PyTorch）也横空出世，并以其易用性和快速开发的特性而占据了一席之地。尤其是在学术研究等需要快速迭代模型的领域，PyTorch 等新兴深度学习框架已成为主流。我所在的数十人的机器学习实验室中，竟只有我一人“守旧”地使用 TensorFlow。而市面上 TensorFlow 相关的中文技术书以及资料也仍然基于传统的计算图与会话机制，这让不少初学者，尤其是刚学过机器学习课程的大学生望而却步。

因此，在 TensorFlow 正式支持即时运行模式之际，我认为有必要出现一本全新的入门手册，帮助初学者及需要快速迭代模型的研究者，以“即时执行”的视角快速入门 TensorFlow。这也是我编写本手册的初衷。本手册自2018年春开始编写，并在2018年8月在 GitHub 发布了第一个中英双语版本，很快得到了国内外不少开发者的关注。尤其是 TensorFlow 工程总监 Rajat Monga 、 Google AI 负责人 Jeff Dean 和 TensorFlow 官方在社交媒体上对本手册给予了推荐与关注，这给了我很大的鼓舞。同时，我作为谷歌开发者专家，多次受到谷歌开发者社群（ `Google Developers Group <https://developers.google.cn/community/gdg>`_ , GDG）的邀请，在 GDG DevFest、TensorFlow Day 和 Women Techmakers 等活动中使用本手册进行线下的 TensorFlow Codelab 教学，获得了较好的反响，也得到了不少反馈和建议。这些都促进了本手册的更新和质量改进。

在2019年3月的第三届TensorFlow Dev Summit开发者峰会上，我再次受邀来到谷歌的硅谷总部，见证了 TensorFlow 2.0 alpha 的发布。此时的 TensorFlow 已经形成了一个拥有庞大版图的生态系统。TensorFlow Lite、TensorFlow.js、TensorFlow for Swift、TPU 等各种组件日益成熟，同时 TensorFlow 2 加入了提升易用性的诸多新特性（例如以 ``tf.keras`` 为核心的统一高层API、使用 ``tf.function`` 构建图模型、默认使用即时执行模式等）。这使得本手册的大幅扩充更新提上日程。GDE 社群的两位 JavaScript 和 Android 领域的资深专家李卓桓和朱金鹏加入了本手册的编写，使得本手册增加了诸多面向业界的 TensorFlow 模块详解与实例。同时，我在谷歌开发者大使（Developer Advocate） Paige Bailey 的邀请下申请并成功加入了 Google Summer of Code 2019 活动。作为全世界 20 位由 Google TensorFlow 项目资助的学生开发者之一，我在 2019 年的暑期基于 TensorFlow 2.0 Beta 版本，对本手册进行了大幅扩充和可读性上的改进。这使得本手册从 2018 年发布的小型入门指南逐渐成长为一本内容全面的 TensorFlow 技术手册和开发指导。

2019年10月1日，TensorFlow 2.0 正式版发布，同时本手册也开始了在 TensorFlow 官方公众微信号（TensorFlow_official）上的长篇连载。在连载过程中，我收到了大量的读者提问和意见反馈。为读者答疑的同时，我也修订了手册中的较多细节。受到新冠疫情的影响，2020年3月的第四届 TensorFlow Dev Summit 开发者峰会在线上直播举行。我根据峰会的内容为手册增添了部分内容，特别是介绍了 TensorFlow Quantum 这一混合量子-经典机器学习库的基本使用方式。我在研究生期间旁听过量子力学，还做过量子计算和机器学习结合的专题报告。TensorFlow Quantum 的推出着实让我感到兴奋，并迫不及待地希望介绍给广大读者。2020年4月，为了适应新冠疫情期间的线上教学需求，我接受 TensorFlow User Group （TFUG）和谷歌开发者社群的邀请，依托本手册在 TensorFlow 官方公众微信号上开展了“机器学习Study Jam”线上教学活动，并启用了手册留言版 https://discuss.tf.wiki 进行教学互动答疑。此次教学也同样有不少学习者为本手册提供了重要的改进意见。

由于我的研究方向是强化学习，所以在本手册的附录中加入了“强化学习简介”一章，对强化学习进行了更细致的介绍。和绝大多数强化学习教程一开始就介绍马尔可夫决策过程和各种概念不同，我从纯动态规划出发，结合具体算例来介绍强化学习，试图让强化学习和动态规划的关系更清晰，以及对程序员更友好。这个视角相对比较特立独行，如果您发现了谬误之处，也请多加指正。

本手册的主要特征有：

* 主要基于 TensorFlow 2 最新的即时执行模式，以便于模型的快速迭代开发。同时使用 ``tf.function`` 实现图执行模式；
* 定位以技术手册为主，编排以 TensorFlow 2 的各项概念和功能为核心，力求能够让 TensorFlow 开发者快速查阅。各章相对独立，不一定需要按顺序阅读；
* 代码实现均进行仔细推敲，力图简洁高效和表意清晰。模型实现均统一使用继承 ``tf.keras.Model`` 和 ``tf.keras.layers.Layer`` 的方式，保证代码的高度可复用性。每个完整项目的代码总行数均不超过一百行，让读者可以快速理解并举一反三；
* 注重详略，少即是多，不追求巨细无遗和面面俱到，不在正文中进行大篇幅的细节论述。

本书的适用群体
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本书适用于以下读者：

* 已有一定机器学习或深度学习基础，希望将所学理论知识使用 TensorFlow 进行具体实现的学生和研究者；
* 曾使用或正在使用 TensorFlow 1.X 版本或其他深度学习框架（比如PyTorch），希望了解和学习 TensorFlow 2 新特性的开发者；
* 希望将已有的 TensorFlow 模型应用于业界的开发者或工程师。

.. hint:: 本书不是一本机器学习或深度学习原理入门手册。若希望进行机器学习或深度学习理论的入门学习，可参考 :doc:`附录“参考资料与推荐阅读” <appendix/recommended_books>` 中提供的一些入门资料。

如何使用本书
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于已有一定机器学习或深度学习基础，着重于使用 TensorFlow 2 进行模型建立与训练的学生和研究者，建议顺序阅读本书的“基础”部分。为了帮助部分新入门机器学习的读者理解内容，本手册在“基础”部分的章节中使用独立的信息框，提供了一些与行文内容相关的机器学习基础知识。这些内容旨在帮助读者将机器学习理论知识与具体的 TensorFlow 程序代码进行结合，以深入了解 TensorFlow 代码的内在机制，让读者在调用 TensorFlow 的 API 时能够知其所以然。然而，这些内容对于没有机器学习基础的读者而言很可能仍是完全不足够的。若读者发现阅读这些内容时有很强的陌生感，则应该先行学习一些机器学习相关的基础概念。部分章节的开头提供了“前置知识”部分，方便读者查漏补缺。

对于希望将 TensorFlow 模型部署到实际环境中的开发者和工程师，可以重点阅读本书的“部署”部分，尤其是需要结合代码示例进行亲手操作。不过，依然非常建议学习一些机器学习的基本知识并阅读本手册的“基础”部分，以更加深入地了解 TensorFlow 2。

对于已有 TensorFlow 1.X 使用经验的开发者，可以从本手册的“高级”部分开始阅读，尤其是“ `图执行模式下的 TensorFlow 2 <https://mp.weixin.qq.com/s?__biz=MzU1OTMyNDcxMQ==&mid=2247487599&idx=1&sn=13a53532ad1d2528f0ece4f33e3ae143&chksm=fc185b27cb6fd2313992f8f2644b0a10e8dd7724353ff5e93a97d121cd1c7f3a4d4fcbcb82e8&scene=21#wechat_redirect>`_ ”和“ `tf.GradientTape 详解 <https://mp.weixin.qq.com/s?__biz=MzU1OTMyNDcxMQ==&mid=2247487701&idx=1&sn=2ef1a15986b1ffd1127460bf6029be01&chksm=fc185b9dcb6fd28befe1a199741b78a6d64cc9eaa1f29c8a0ca251c4e2157a8395917d67b49e&scene=21#wechat_redirect>`_ ”两章。随后，可以快速浏览一遍“基础”部分以了解即时执行模式下 TensorFlow 的使用方式。

在整本手册中，带“*”的部分均为选读。

本书的所有示例代码可至 https://github.com/snowkylin/tensorflow-handbook/tree/master/source/_static/code 获得。其中 ``zh`` 目录下是含中文注释的代码（对应于本书的中文版本，即您手上的这一本）， ``en`` 目录下是含英文版注释的代码（对应于本书的英文版本）。在使用时，建议将代码根目录加入到 ``PYTHONPATH`` 环境变量，或者使用合适的IDE（如PyCharm）打开代码根目录，从而使得代码间的相互调用（形如 ``import zh.XXX`` 的代码）能够顺利运行。

致谢
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

首先感谢我的好友兼同学 Chris Wu 编写的《简单粗暴 :math:`\text{\LaTeX}` 》（https://github.com/wklchris/Note-by-LaTeX，纸质版名为《简单高效LaTeX》）为本手册的初始体例编排提供了模范和指引。该手册清晰精炼，是 :math:`\text{\LaTeX}` 领域不可多得的中文资料，本手册的初始名称《简单粗暴 TensorFlow》也是在向这本手册致敬。本手册最初是在我的好友 Ji-An Li 所组织的深度学习研讨小组中，作为预备知识讲义编写和使用的。好友们卓著的才学与无私分享的精神是编写此拙作的重要助力。

本手册的TensorFlow.js和TensorFlow Lite章节分别由李卓桓和朱金鹏两位在JavaScript和Android领域有丰富履历的GDE和前GDE撰写，同时，卓桓撰写了TensorFlow for Swift和TPU部分的介绍，金鹏提供了TensorFlow Hub的介绍。来自豆瓣阅读的王子阳也提供了关于Node.js和阿里云的部分示例代码和说明。在此特别表示感谢。

在基于本手册初稿的多场线下、线上教学活动和TensorFlow官方微信公众号连载中，大量活动参与者与读者为本手册提供了有价值的反馈，促进了本手册的持续更新。谷歌开发者社群和 TensorFlow User Group 的多位志愿者们也为这些活动的顺利举办做出了重要贡献。来自中国科学技术大学的 Zida Jin 将本手册2018年初版的大部分内容翻译为了英文，Ming 和 Ji-An Li 在英文版翻译中亦有贡献，促进了本手册在世界范围内的推广。Eric ShangKuan、Jerry Wu 、Hsiang Huang、Po-Yi Li、Charlie Li 、Chunju Hsu 协助了本手册的简转繁转译。在此一并表示由衷的谢意。

衷心感谢 Google 开发者关系团队和 TensorFlow 工程团队的成员及前成员们对本手册的编写所提供的帮助。其中，开发者关系团队的 Luke Cheng 在本手册初版编写过程中提供重要的思路启发和鼓励，且提供本手册在线版本的域名 `tf.wiki <https://tf.wiki>`_ 和留言版 https://discuss.tf.wiki ；开发者关系团队的 Soonson Kwon、Lily Chen、Wei Duan、Tracy Wang、Rui Li、Pryce Mu，TensorFlow 产品经理 Mike Liang 和谷歌开发者大使 Paige Bailey 为本手册宣传及推广提供了大力支持；开发者关系团队的 Eric ShangKuan 协助了本手册的繁体版转译。TensorFlow 工程团队的 Tiezhen Wang 在本手册的工程细节方面提供了诸多建议和补充；TensorFlow 中国研发负责人 Shuangfeng Li 和 TensorFlow 工程团队的其他工程师们为本手册提供了专业的审阅意见。同时感谢 TensorFlow 工程总监 Rajat Monga 和 Google AI 负责人 Jeff Dean 在社交媒体上对本手册的推荐与关注。感谢 Google Summer of Code 2019 对本开源项目的资助。

本手册主体部分为我在北京大学信息科学技术学院智能科学系攻读硕士学位时所撰写。感谢我的导师童云海教授和实验室的同学们对本手册的支持和建议。

最后，感谢人民邮电出版社的王军花、武芮欣两位编辑对本手册纸质版的细致编校及出版流程跟进。感谢我的父母和好友对本手册的关注和支持。

关于本手册的意见和建议，欢迎在 https://discuss.tf.wiki 提交。您的宝贵意见将促进本手册的持续更新。

|

Google Developers Expert in Machine Learning

Xihan Li ( `snowkylin <https://snowkylin.github.io/>`_ )

2020 年 5 月于深圳
