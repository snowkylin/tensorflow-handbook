.. 简单粗暴TensorFlow documentation master file, created by
   sphinx-quickstart on Sat Jan 20 00:48:15 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================================================
简单粗暴TensorFlow 2.0 | A Concise Handbook of TensorFlow 2.0
==================================================================

*基于Eager Execution | Based on Eager Execution*

..
    本文档为未完成版本，内容会随时更改修订，目前请不要扩散。

    This document is unfinished, content will be updated rapidly. Please keep it internal at this time.

本手册是一篇精简的TensorFlow 2.0入门指导，基于TensorFlow的Eager Execution（动态图）模式，力图让具备一定机器学习及Python基础的开发者们快速上手TensorFlow 2.0。

本文的所有代码基于TensorFlow 2.0 beta版本。

This handbook is a concise introduction to TensorFlow 2.0 based on Eager Execution mode, trying to help developers with some basic machine learning and Python knowledge to get started with TensorFlow 2.0 quickly.

The code of this handbook is based on TensorFlow 2.0 beta.

..
    .. hint:: 这是一本TensorFlow技术手册，而不是一本机器学习/深度学习原理入门手册。如果发现阅读中有难以理解的部分，请检查每章的“前置知识”部分，这里提供了一些机器学习原理的入门资料链接。
        
        This is a TensorFlow technical handbook rather than a tutorial for machine learning or deep learning theories. If you find something difficult to understand in reading, please check the "Prerequisites" part of each chapter, where some good basic machine learning documents are provided by url links.

..
    +------------------------+------------------------+
    | .. toctree::           | .. toctree::           |
    |    :maxdepth: 2        |    :maxdepth: 2        |
    |    :caption: 目录      |    :caption: Contents  |
    |                        |                        |
    |    zh/preface          |    en/preface          |
    |    zh/introduction     |    en/introduction     |
    |    zh/installation     |    en/installation     |
    |    zh/basic            |    en/basic            |
    |    zh/models           |    en/models           |
    |    zh/extended         |    en/extended         |
    |    zh/deployment       |    en/deployment       |
    |    zh/javascript       |    en/javascript       |
    |    zh/training         |    en/training         |
    |    zh/application/rl   |    en/application/rl   |
    |    zh/application/rnn  |    en/application/rnn  |
    |    zh/application/prob |    en/application/prob |
    |    zh/static           |    en/static           |
    |    zh/swift            |    en/swift            |
    |    zh/reuse            |    en/reuse            |
    |    zh/addons           |    en/addons           |
    |    zh/custom_op        |    en/custom_op        |
    |    zh/config           |    en/config           |
    |    zh/recommended_books|    en/recommended_books|
    +------------------------+------------------------+

.. toctree:: 
    :maxdepth: 2
    :caption: 目录

    zh/preface
    zh/introduction

.. toctree:: 
    :maxdepth: 2
    :caption: 基础

    zh/basic/installation
    zh/basic/basic
    zh/basic/models
    zh/basic/tools

.. toctree:: 
    :maxdepth: 2
    :caption: 部署

    zh/deployment/export
    zh/deployment/serving
    zh/deployment/lite
    zh/deployment/javascript

.. toctree:: 
    :maxdepth: 2
    :caption: 大规模训练与加速

    zh/appendix/distributed
    zh/appendix/tpu

.. toctree:: 
    :maxdepth: 2
    :caption: 扩展

    zh/appendix/tfhub
    zh/appendix/tfds  
    zh/appendix/swift
    zh/appendix/julia

..
    .. toctree:: 
        :maxdepth: 2
        :caption: 应用

        zh/application/rl
        zh/application/chatbot

.. toctree:: 
    :maxdepth: 2
    :caption: 附录

    zh/appendix/static 
    zh/appendix/config 
    zh/appendix/optimization
    zh/appendix/recommended_books
    zh/appendix/terms

..
    .. toctree:: 
        en/preface
        en/installation
        en/basic
        en/models
        en/extended 
        en/static

答疑区 

- （中文）TensorFlow中文社区“简单粗暴TensorFlow”版面： https://www.tensorflowers.cn/b/48 （中文的疑问和建议请来此处，将以中文回答和讨论。欢迎使用中文的开发者们前来TensorFlow中文社区交流讨论）
- （英文）https://github.com/snowkylin/TensorFlow-cn/releases （英文的疑问或建议可在GitHub issue中提出，会以英文回答）

PDF下载：

- 中文版：https://www.tensorflowers.cn/t/6230 （同时也有英文版下载）
- 英文版：https://github.com/snowkylin/TensorFlow-cn/releases

GitHub: https://github.com/snowkylin/TensorFlow-cn

Q&A area

- (Chinese) TensorFlow Chinese community "A Concise Handbook of TensorFlow" forum: https://www.tensorflowers.cn/b/48 
- (English) https://github.com/snowkylin/TensorFlow-cn/issues

PDF download: 

- Chinese version: https://www.tensorflowers.cn/t/6230
- English version: https://github.com/snowkylin/TensorFlow-cn/releases

GitHub: https://github.com/snowkylin/TensorFlow-cn

..  
   preface
   introduction
   installation
   basic
   ops
   models
    --
   visualization
   debugging   
    --
   distributed
   dynamic   
   code
   appendix

.. only:: html

    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`

    .. raw:: html
    
        <a href="https://info.flagcounter.com/Hyjs"><img src="https://s05.flagcounter.com/count2/Hyjs/bg_FFFFFF/txt_000000/border_CCCCCC/columns_2/maxflags_16/viewers_0/labels_1/pageviews_1/flags_0/percent_0/" alt="Flag Counter" border="0"></a>

