Preface
=======

On March 30, 2018, Google held the second TensorFlow Dev Summit in Mountain View, California and announced the official release of TensorFlow version 1.8. I was honoured to receive Google's funding to pay a visit to the summit in person, witnessing the release of this landmark new version. Many new functions were added and supported, revealing TensorFlow's great ambition. Meanwhile, the Eager Execution mode that has been tested early since 2017 fall was finally officially included in this version, and became the recommended mode for TensorFlow newcomers.

    The easiest way to get started with TensorFlow is using Eager Execution.
    
    —— https://www.tensorflow.org/get_started/

Before then, the disadvantages of the conventional Graph Execution mode on which TensorFlow was based, such as high entry barrier, difficult debugging, poor flexibility and inability to use Python's native control statements, have long been criticized by developers. Some new DL frameworks based on dynamic computational graph mechanism (e.g., PyTorch) have also emerged and have won their places with their usability and rapid development features. These dynamic DL frameworks have become mainstream, especially in areas such as academic research that requires rapid iterative development models. In fact, I was the only person who used the "old-fashioned" TensorFlow in my ML laboratory where I worked with dozens of colleagues. However, until now, most of the Chinese technical books and materials about TensorFlow are still based on Graph Execution mode, which really dissuades beginners (especially those undergraduates who have just finished their ML courses) from learning. Therefore, as TensorFlow officially supports Eager Execution, it is necessary to publish a brand new handbook to help beginners and researchers who need to iterate models rapidly and get started quickly from a new perspective.

Meanwhile, this handbook also has another task. Most Chinese technical books on the market related to TensorFlow are mainly based on DL and regard TensorFlow as a mere means to implement these models. Although this kind of arrangement has the advantage of a complete system, it is not friendly enough for readers who already have an understanding of ML or DL theories, hoping to focus on learning TensorFlow itself. Hence I hope to write a handbook to reveal the main features of TensorFlow as a computing framework in a concise manner and make up for the shortcomings of the official manual, trying to make readers who already possess certain knowledge of ML / DL and programming skills get started with TensorFlow very quickly as well as view it to solve practical problems at any time in real programming.

The main features of this handbook consist of:

* Mainly based on the TensorFlow's latest Eager Execution mode to facilitate rapid iterative development of models, although the conventional Graph Execution mode is still included. Codes are as compatible as possible with both;
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

    This book is not a tutorial of ML / DL. Please refer to the :doc:`appendix <appendix/recommended_books>` for some introductory materials if you want to get started with those theories.

Usage
^^^^^

For students and researchers who already possess certain knowledge of ML / DL, it is advised to view the "basic" part sequentially. For developers and engineers who wish to deploy TensorFlow models to the real environments, we suggest you pay attention on the "deployment" part. The beginning of each chapter provides "prerequisites" for readers to fill in the gaps. Some supplementary comments are displayed in collapsible boxes which can be all collapsed by clicking the "Fold all admonitions" button at the top of the page anytime.

The parts marked with an "*" are optional in this handbook.

Acknowledgement
^^^^^^^^^^^^^^^

This handbook was tentatively named as "A Concise Handbook of TensorFlow" in order to pay a tribute to the book "A Concise Handbook of :math:`\text{\LaTeX}`" (https://github.com/wklchris/Note-by-LaTeX) written by my friend and colleague Chris Wu. The latter is a rare Chinese material about :math:`\text{\LaTeX}`. I also learned from it while I was writing this handbook. This handbook was initially written and used by meself as a prerequisite handout in a DL seminar organized by my friend Ji-An Li. My friends' wisdom and selflessness also prompted me to accomplish this project.

The TensorFlow.js and TensorFlow Lite sections of this handbook were respectively written by Huan Li and Jinpeng Zhu, two Google Developers Expert with rich experience in JavaScript and Android. Meanwhile, Huan provided an introduction to TensorFlow for Swift and TPU part. In addition, Ziyang Wang from Douban provided an introduction to TensorFlow for Julia as well as some sample codes with instructions. Contents written by relevant contributors are marked in the articles and special thanks are given hereby.

Sincere thanks to the members of the Google China Developer Relations team and the TensorFlow engineering team for their assistance in writing this handbook, which includes the insipiration and continuous encouragement from Luke Cheng of the Developer Relations team throughout the writing of this manual (and the domain name ``tf.wiki`` of the online version); the strong support on the publication and promotion of this manual from Rui Li, Pryce Mu of the Developer Relations team and friends who maintain the TensorFlow community; many suggestions and supplements on the engineering details from Tiezhen Wang of TensorFlow engineering team; and reviews from other engineers such as Shuangfeng Li of the TensorFlow engineering team.

.. The English version of this handbook is translated by my friend Zida Jin and Ming, and revised by Ji-An Li and me (still in progress). My three friends sacrificed a lot of valuable time to translate and proofread this handbook. Ji-An Li also provided valuable comments on the teaching content and code details of this manual. I would like to express my heartfelt thanks to my friends for their hard work.

Please feel free to submit any comment and suggestion about this handbook at https://github.com/snowkylin/tensorflow-handbook/issues. This is an open source project and your valuable feedback will facilitate the continuous updating of this handbook.

|

Google Developers Expert in Machine Learning

Xihan Li (Snowkylin)

August 2019 in Yanyuan
