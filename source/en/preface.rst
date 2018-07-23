Foreword
======

On March 30, 2018, Google held the second TensorFlow Dev Summit Developer Summit in Mountain View, California, and announced the official release of TensorFlow 1.8. I was fortunate to receive a grant from Google to attend the summit and witnessed the release of this milestone  version. Numerous new features added in this version demonstrates TensorFlow's ambitions, and Eager Execution, which began testing in the fall of 2017, has finally been officially released in this version and becomes the official recommendation for entry-level learning of TensorFlow.

    The easiest way to get started with TensorFlow is using Eager Execution.
    
    -- https://www.tensorflow.org/get_started/

Prior to this, the drawbacks of the traditional Graph Execution based on TensorFlow, such as high barriers to entry, difficulty in debugging, poor flexibility, and inability to use Python native control statements, have long been criticized by developers for a long time. Some new deep learning frameworks based on dynamic graphing mechanisms (such as PyTorch) have also emerged, and have taken their place with ease of use and rapid development. Especially in areas such as academic research that require rapid iterative models, emerging deep learning frameworks such as PyTorch have become mainstream. In the machine learning lab where the author is in the dozens of people, only the author used the TensorFlow "old-fashioned". However, until now, the relevant Chinese technical books and materials related to TensorFlow on the market are still based on the traditional Graph Execution model, which has made many beginners (especially those who have just studied machine learning courses) discouraged. As a result, when TensorFlow officially supports Eager Execution, it is necessary to have a new technical manual to help beginners and researchers who need to quickly iterate models to quickly get started with TensorFlow from a new perspective.

At the same time, this manual has a second task. Most of the Chinese technical books related to TensorFlow in the market are based on deep learning, and TensorFlow is used as the implementation of these deep learning models. This has the advantage of a complete system, but it is not friendly enough for readers who already have an understanding of machine learning or deep learning theory and who want to focus on learning TensorFlow itself. At the same time, although TensorFlow has official teaching documents (https://tensorflow.google.cn/tutorials), it is not logical enough in the system, lacking the general teaching documents from shallow to deep, progressive characteristics, and more similar Listed in a series of technical documents. Therefore, the author hopes to write a manual to show the main features of TensorFlow as a computing framework as much as possible, and to make up for the shortcomings of the official manual, in an effort to make readers who already have certain machine learning/deep learning knowledge and programming skills get started quickly. TensorFlow, and can actually review and solve practical problems during the actual programming process.

The main features of this manual are:

* Mainly based on TensorFlow's latest Eager Execution mode for rapid iterative development of models. But still will include the traditional Graph Execution mode, the code is as compatible as possible;
* Positioning is based on teaching and reference books. The layout is based on TensorFlow's concepts and functions, and is designed to enable TensorFlow developers to quickly access them. The chapters are relatively independent and do not necessarily need to be read in order. There will not be too many theoretical introductions to deep learning and machine learning in the text, but there will be a number of reading recommendations for beginners to master the relevant basics;
* Code implementations are carefully scrutinized in an effort to be concise, efficient, and ideographic. Model implementations are uniformly used with the official documentation of `TensorFlow <https://www.tensorflow.org/programmers_guide/eager#build_a_model>`_ The latest proposed inheritance ``tf.keras.Model`` and  ``tf.keras.layers.Layer`` (described rarely in other technical documents) guarantees a high degree of reusability of the code. The total number of lines per complete project is no more than 80 lines, so that readers can quickly understand and give inferences;
* Pay attention to the details, less is more, do not pursue the details and all-round, do not carry out large-scale details.

In the entire manual, the sections marked with "*" are optional.

The tentative name of this manual, "Simple and Easy TensorFlow" salutes "Simple and Easy :math:`\text{\LaTeX}` " (https://github.com/wklchris/Note-by-LaTeX), a clear and concise Chinese manual about :math:`\text{\LaTeX}` written by my friend and classmate Chris Wu. It is also the object I learned when writing this technical document. This manual was originally written and used by me as a preparatory knowledge in the in-depth study group organized by my good friend Ji'an Li. The talented and unselfish character of friends is also an important help in writing this work.

Thanks to Lu Cheng of Google China's Developer Relations team and Tiezhen Wang of the Google TensorFlow team for their support and valuable comments on this manual.

|

Xihan Li (Snowkylin)

July 2018 in Peking University