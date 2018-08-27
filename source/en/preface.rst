Preface
=========

On Mar. 30th, 2018, Google held the second TensorFlow Dev Summit in Mountain View, California and announced the official release of TensorFlow version 1.8. I was fortunate to attend the summit with Google's sponsorship, witnessing the release of this new version, a milestone. Lots of new functions added and supported shows the ambition of TensorFlow. Meanwhile, Eager Execution, which has been tested since 2017 fall, was finally included officially in this version and became the recommended mode for newcomers of TensorFlow.

    The easiest way to get started with TensorFlow is using Eager Execution.
    
    —— https://www.tensorflow.org/get_started/

Before then, the disadvantages of Graph Execution mode in TensorFlow, such as high learning threshold, difficulty in debugging, poor flexibility and inability to use Python native controlling statements, have already been criticized by developers for a long time. Some new deep learning frameworks based on dynamic computational graph (e.g. PyTorch) have come out and won their places by their usability and efficiency for development. These dynamic deep learning frameworks are popular especially in academic researches where fast iterative development of models are required. In fact, I was the only person who used "old-fashioned" TensorFlow in my machine learning laboratory where I worked with dozens of colleagues. However, until now, most of the Chinese technical books and materials about TensorFlow still based on Graph Execution mode, which really dissuades beginners (especially those undergraduates who have just finished their machine learning courses) from learning. Therefore, as TensorFlow officially supports Eager Execution, it's necessary to publish a brand new handbook which helps beginners and researchers who need to iterate models rapidly and to get started quickly from a new perspective.

Meanwhile, this handbook has another mission. Most Chinese technical books about TensorFlow focus mainly on deep learning and regard TensorFlow as a mere tool to implement deep learning models. Admittedly, they are self-contained, but it's not friendly enough for those who have already known about machine learning and deep learning theories and want to focus on learning TensorFlow itself. In addition, though TensorFlow has its official documentation (https://tensorflow.google.cn/tutorials), its structure is not well organized, lacking the step-by-step feature of a common tutorial, thus being more similar to a technological documentation.

The main features of this handbook are:

* This book is mainly based on the most up-to-date Eager Execution mode in TensorFlow for fast iterative development of models. However, traditional Graph Execution mode is also included and we will do our best to make the codes provided in this book compatible with both modes.
* We position this book mainly as a tutorial and handbook, and arrange the concepts and functions of TensorFlow as the core part, for TensorFlow developers to refer quickly. Chapters are relatively independent with one another, therefore it's not necessary to read this book in a sequential order. There won't be much theory of deep learning and machine learning in the text, however some recommendation is still provided for beginners to grasp related basic knowledge.
* All codes are carefully written in order to be concise and efficient. All models are implemented based on the ``tf.keras.Model`` and ``tf.keras.layers.Layer`` methods, which just proposed by `TensorFlow official documentation <https://www.tensorflow.org/programmers_guide/eager#build_a_model>`_ and are barely introduced in other technical documentations. This implementation guarantees high reusability. Each project is written by codes fewer than 100 lines for readers to understand and practice quickly.
* Less is more. No all-rounded or large blocks of details.

The parts marked "*" are optional in this handbook.

This handbook is tentatively named as "A Concise Handbook of TensorFlow" in order to pay a tribute to the book "A Concise Handbook of :math:`\text{\LaTeX}`" written by my friend and colleague Chris Wu. The latter is a rare Chinese material about :math:`\text{\LaTeX}`. I also learned from it while I was writing this handbook. This handbook was initially written and used by meself as a prerequisite handout in a deep learning seminar organized by my friend Ji-An Li. My friends' wise and selflessness also prompted me to finish this work.

The English version of this handbook is translated by my friend Zida Jin (Chapter 1-4) and Ming (Chapter 5-6), and revised by Ji-An Li and me. My three friends sacrificed a lot of valuable time to translate and proofread this handbook. Ji-An Li also provided valuable comments on the teaching content and code details of this manual. I would like to express my heartfelt thanks to my friends for their hard work.

I am grateful to the members of the Google China Developer Relations team and the TensorFlow engineering team for their help in writing this handbook. Among them, Luke Cheng of the Developer Relations team provided inspiration and continuous encouragement throughout the writing of this manual. Rui Li and Pryce Mu of the Developer Relations team provided strong support in the promotion of this manual, as well as Tiezhen Wang of TensorFlow team provided many suggestions to the engineering details of the manual.

|

Xihan Li (Snowkylin)

August 2018 in Yanyuan
