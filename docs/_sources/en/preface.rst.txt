Preface
=======

On March 30, 2018, Google held the second TensorFlow Dev Summit in Mountain View, California and announced the official release of TensorFlow version 1.8. As one of the first `Google Developers Experts <https://developers.google.com/community/experts>`_ in China, I was honoured to receive Google's funding to pay a visit to the summit in person, witnessing the release of this landmark new version. Many new functions were added and supported, revealing TensorFlow's great ambition. Meanwhile, the Eager Execution mode was finally officially included in this version, and became the recommended mode for TensorFlow newcomers.

Before then, the disadvantages of the conventional Graph Execution mode on which TensorFlow was based, such as high entry barrier, difficult debugging, poor flexibility and inability to use Python's native control statements, have long been criticized by developers. Some new DL frameworks based on dynamic computational graph mechanism (e.g., PyTorch) have also emerged and have won their places with their usability and rapid development features. These dynamic DL frameworks have become mainstream in areas such as academic research that requires rapid iterative development models. In fact, I was the only person who used the "old-fashioned" TensorFlow in my ML laboratory where I worked with dozens of colleagues. However, until now, most of the Chinese technical books and materials about TensorFlow are still based on Graph Execution mode, which really dissuades beginners (especially those undergraduates who have just finished their ML courses) from learning. 

Therefore, as TensorFlow officially supports Eager Execution, I think it is necessary to publish a brand new handbook to help beginners and researchers who need to iterate models rapidly and get started quickly from a new perspective. I start writing this handbook since the spring of 2018, and the first bilingual version was released on GitHub in August 2018, quickly gaining the attention of many developers in China and around the world. In particular, I was encouraged by the recommendations and attention given to this handbook on social media by Rajat Monga, Director of Engineering at TensorFlow, Jeff Dean, Head of Google AI, and TensorFlow official Twitter account. At the same time, as a Google developer expert, I have been invited by the Google Developers Group (`Google Developers Group <https://developers.google.com/community/gdg>`_ , GDG) many times to as main speaker of TensorFlow Codelab activities at GDG DevFest, TensorFlow Day and Women Techmakers events, etc. I use this handbook as the main teachine material and received many constructive feedback and suggestions. These have contributed to the updating and quality improvement of this handbook.

At the 3rd TensorFlow Dev Summit Summit in March 2019, I was once again invited to Google's Silicon Valley headquarter to witness the release of TensorFlow 2.0 alpha. At this point, many components of TensorFlow ecosystem like TensorFlow Lite, TensorFlow.js, Swift for TensorFlow and TPU support becomes more mature, and TensorFlow 2 has added new features to improve usability (e.g., a unified high-level API with ``tf.keras``, the use of ``tf.function`` to build the dataflow graph, and the default use of eager execution mode). Two JavaScript and Android experts from the GDE community, Huan Li and Jinpeng Zhu, have joined the writing of this handbook, adding lots of industry-oriented details and examples about some TensorFlow module. In the meantime, I successfully joined the Google Summer of Code 2019 event. As one of 20 student developers worldwide funded by the Google TensorFlow program, I have significantly expanded and improved the handbook based on TensorFlow 2.0 Beta in the summer of 2019. This has allowed this handbook to grow from a small introductory guide to a comprehensive TensorFlow technical handbook and development guide.

On October 1, 2019, the official release of TensorFlow 2.0 was announced, and this handbook has also started a long serialization on the official TensorFlow Wechat account (TensorFlow_official) in China. During the serialization process, I received a lot of questions and feedback from readers. As well as answering questions for the reader, I have revised many details in the handbook. Affected by COVID-19, the 4th TensorFlow Dev Summit takes place live online in March 2020. I've added to the handbook based on the summit, specifically introducing the basic use of TensorFlow Quantum, a hybrid quantum-classical machine learning library. In April 2020, I was invited by the TensorFlow User Group (TFUG) and the Google developer community to launch the "Machine Learning Study Jam" activity on the official TensorFlow Wechat account, and launched a forum https://discuss.tf.wiki for interactive Q&A. A number of learners contributed important improvements to this handbook in the teaching.

Since my research focuses on reinforcement learning, I have included a chapter "Introduction to Reinforcement Learning" in the appendix of this handbook, which provides a more detailed introduction to reinforcement learning. Unlike most reinforcement learning tutorials that begin with an introduction to the Markov decision process and various concepts, I introduce reinforcement learning from a purely dynamic programming perspective, combined with concrete examples, in an attempt to make the relationship between reinforcement learning and dynamic programming clearer and more programmer-friendly. This is a relatively unique perspective, so please correct me if you find any errors.

The main features of this handbook consist of:

* Mainly based on the TensorFlow's latest Eager Execution mode to facilitate rapid iterative development of models. Use ``tf.function`` to write code in Graph Execution mode;
* Mainly positioned as a technical handbook whose layout is centred on various concepts and functions of TensorFlow, striving to enable TensorFlow developers to refer to it quickly. Chapters are relatively independent with each other thus it is not necessary to read in order;
* All codes are carefully reviewed for conciseness and clarity. All model implementations follow the way of inheriting ``tf.keras.Model`` and ``tf.keras.layer.Layer`` recently proposed by `TensorFlow official documentation <https://www.tensorflow.org/programmers_guide/eager#build_a_model>`_ to guarantee the high reusability of codes. The codes for each complete project do not exceed 100 lines, allowing readers to comprehend quickly and learn by analogy;
* Properly detailed, less is more. No pursuit of everything, no long speeches in the text.

Target readers
^^^^^^^^^^^^^^

This book is intended for the following type of readers:

* Students and researchers who already have a certain foundation of ML / DL and want to use TensorFlow to practice their theories.

* Developers who have used or are using TensorFlow version 1.X or other DL frameworks (e.g., PyTorch) and want to know about the new features of TensorFlow 2.0.

* Developers or engineers who wish to apply existing TensorFlow models to the industry.

.. admonition:: Hint

    This book is not a tutorial of Machine Learning or Deep Learning. Please refer to the :doc:`appendix <appendix/recommended_books>` for some introductory materials if you want to get started with those theories.

Usage
^^^^^

For students and researchers who already possess certain knowledge of Machine Learning or Deep Learning, it is advised to view the "basic" part sequentially. To help some readers who are new to machine learning to understand the content, this handbook provides some machine learning basics related to the content of the handbook, using separate information boxes in the "Basics" part of the chapter. This content is designed to help readers combine their theoretical knowledge of machine learning with specific TensorFlow program code to gain insight into the mechanisms inherent in TensorFlow code, so that readers can know what is going on when they call TensorFlow's API. However, this content is likely to remain inadequate for readers without any knowledge about machine learning. If readers find reading this content to be highly unfamiliar, they should learn some basic concepts related to machine learning first. A "prerequisite" section is provided at the beginning of some chapters to make it easier for the reader to check for gaps.

For developers and engineers who wish to deploy TensorFlow models to the real environments, we suggest you pay attention on the "deployment" part, especially for hands-on work in conjunction with code examples. However, it is still highly recommended to learn some of the basics of machine learning and read the "Basics" part of this handbook to get a deeper understanding of TensorFlow 2.

Some supplementary comments are displayed in collapsible boxes which can be all collapsed by clicking the "Fold all admonitions" button at the top of the page anytime.

The parts marked with an "*" are optional in this handbook.

Acknowledgement
^^^^^^^^^^^^^^^

This handbook was tentatively named as "A Concise Handbook of TensorFlow" in order to pay a tribute to the book "A Concise Handbook of :math:`\text{\LaTeX}`" (https://github.com/wklchris/Note-by-LaTeX) written by my friend and colleague Chris Wu. The latter is a rare Chinese material about :math:`\text{\LaTeX}`. I also learned from it while I was writing this handbook. This handbook was initially written and used by meself as a prerequisite handout in a DL seminar organized by my friend Ji-An Li. My friends' wisdom and selflessness also prompted me to accomplish this project.

The TensorFlow.js and TensorFlow Lite sections of this handbook were respectively written by Huan Li and Jinpeng Zhu, two GDE and former GDE with rich experience in JavaScript and Android. Meanwhile, Huan provided an introduction to TensorFlow for Swift and TPU part. In addition, Ziyang Wang from Douban provided some sample codes about TensorFlow Serving and Aliyun with instructions. Contents written by relevant contributors are marked in the articles and special thanks are given hereby.

A large number of participants and readers have provided valuable feedback to this handbook and contributed to the continuous updating of this handbook. Several volunteers from the Google Developers Group and TensorFlow User Group have also made important contributions to the smooth running of these events. Zida Jin from the University of Science and Technology of China translated most of the contents of the initial 2018 edition of this handbook into English. Ming and Ji-An Li also contributed to the English translation, facilitating the worldwide promotion of this handbook. Eric ShangKuan, Jerry Wu, Hsiang Huang, Po-Yi Li, Charlie Li, and Chunju Hsu assisted in the simplified to traditional Chinese translation of this handbook. I would also like to express my sincere gratitude. 

Sincere thanks to the members of the Google China Developer Relations team and the TensorFlow engineering team for their assistance in writing this handbook, which includes the insipiration and continuous encouragement from Luke Cheng of the Developer Relations team (and the domain name ``tf.wiki`` of the online version); the strong support on the promotion of this handbook from Soonson Kwon, Lily Chen, Wei Duan, Tracy Wang, Rui Li, Pryce Mu of the Developer Relations team, TensorFlow product manager Mike Liang and Google Developers Advocate Paige Bailey; many suggestions and supplements on the engineering details from Tiezhen Wang of TensorFlow engineering team; and reviews from Shuangfeng Li, Head of R&D at TensorFlow China, and other engineers in the TensorFlow engineering team. Thanks also to Rajat Monga, Director of Engineering at TensorFlow, and Jeff Dean, Head of Google AI, for recommending and following this handbook on social media. Thanks to Google Summer of Code 2019 for funding this open source project.

The main part of this handbook was written while I was pursuing my master's degree in the Department of Intelligent Science, School of Electrical Engineering And Computer Science, Peking University. I would like to thank my M.S. advisor, Prof. Yunhai Tong, and my classmates in the lab for their support and advice on this handbook.

Finally, I would like to thank Junhua Wang and Ruixin Wu, editors of the Post and Telecom Press, for their careful editing and follow-up of the publication process of the Chinese paper version of this handbook. Thanks to my parents and friends for their support of this handbook.

Please feel free to submit any comment and suggestion about this handbook at https://discuss.tf.wiki. This is an open source project (https://github.com/snowkylin/tensorflow-handbook) and your valuable feedback will facilitate the continuous updating of this handbook.

|

Google Developers Expert in Machine Learning

Xihan Li ( `Snowkylin <https://snowkylin.github.io>`_ )

May 2020 in Shenzhen
