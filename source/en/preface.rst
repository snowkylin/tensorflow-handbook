Preface
=========

On Mar 30th, 2018, Google held the second TensorFlow Dev Summit in Mountain View, California and announced the official release of TensorFlow version 1.8. I was fortunate to attend the summit with Google's sponsorship, witnessing the release of this milestone new version. Lots of new functions being added and supported shows the ambition of TensorFlow. Meanwhile, Eager Execution, which has been tested since 2017 fall, was finally included officially in this version and became the recommended mode for newcomers of TensorFlow.

    The easiest way to get started with TensorFlow is using Eager Execution.
    
    —— https://www.tensorflow.org/get_started/

Before then, the disadvantages of TensorFlow that had based on traditional Graph Execution, like the high entry barrier, being difficult to debug, poor flexibility and being unable to use Python native controlling statements have already been criticized for a long time by developers. Some new deep learning models (e.g. PyTorch) based on Eager Execution has come out and won their places by their easy-to-use and rapid development features. Emerging deep learning models like PyTorch has been popular especially in those academic research field where fast iterative models are required. It was surprising that it was me that had been the only person who used "traditional" TensorFlow in the machine learning laboratory where I worked with dozens of colleagues. However, most of the Chinese technological books and files about TensorFlow on the market have still been based on Graph Execution mode till now, which really dissuades beginners (mostly those undergraduates who have just finished their machine learning courses) from entering. Therefore it's necessary to publish a brand new handbook which helps the beginners and those who need to learn fast iterative models get started quickly from a new perspective as TensorFlow officially supports Eager Execution.

Meanwhile, this handbook has another task to do. Most Chinese technological books about TensorFlow on the market focus mainly on deep learning and regard TensorFlow as a mere implementation of these models. Of course its integral system is an advantage, but it's not friendly enough for those who has already known about machine learning and deep learning theories and wants to focus on learning TensorFlow itself. In addition, nevertheless TensorFlow has its official documentation (https://tensorflow.google.cn/tutorials), its structure is not well organized, lacking the step-by-step feature that usually a tutorial has, thus being more similar to a technological documentation.

The main features of this handbook includes:

* This book is mainly based on the most up-to-date Eager Execution mode in TensorFlow for fast iterative development of models. However traditional Graph Execution mode is also included and we will do our best to make the codes provided in this book compatible with both modes.
* We position this book mainly as a tutorial and reference and arrange the concepts and functions in TensorFlow as the core part for TensorFlow developers to check quickly. Chapters are relatively independent with one another, therefore it's not necessary to read this book in a sequential order. There won't be much theory of deep learning and machine learning in the text, however some recommendation is still provided for beginners to grasp related basic knowledge.
* All codes are carefully written in order to provide concise and efficient representations. All models are implemented based on the ``tf.keras.Model`` and ``tf.keras.layers.Layer`` methods that `TensorFlow official documentation <https://www.tensorflow.org/programmers_guide/eager#build_a_model>`_ proposed latest (which are barely introduced in other technological documentations), which guarantees high reusability. Each project is written by codes less than 80 lines for readers to understand and practice quickly.
* Less is more. No all-rounded details, no large-scale discussion.

The part marked "*" is optional in this handbook.

This handbook is tentatively named as "A Concise Handbook of TensorFlow" in order to pay tribute to the book "A Concise Handbook of :math:`\text{\LaTeX}`" written by my friend and colleague Chris Wu. The latter is a rare Chinese file about :math:`\text{\LaTeX}` and is also what I reference from while I was writing this handbook. This handbook was initially written and used by me as a prerequisites knowledge handout when in a studying discussion created by my friend Ji'an Li. Our friends' wise and selflessness also prompt me to finish this work.

Thanks to Lu Cheng from Google China Developer Relations Team and Tiezhen Wang from Google TensorFlow Team for giving a helping hand on this work.

|

Xihan Li (Snowkylin)

Jul 2018 in Yanyuan