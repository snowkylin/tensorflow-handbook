.. 简单粗暴TensorFlow documentation master file, created by
   sphinx-quickstart on Sat Jan 20 00:48:15 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================================================
简单粗暴 TensorFlow 2 | A Concise Handbook of TensorFlow 2
==================================================================

这是一本简明的 TensorFlow 2 入门指导手册，基于 Keras 和即时执行模式（Eager Execution），力图让具备一定机器学习及 Python 基础的开发者们快速上手 TensorFlow 2。

本手册的所有代码基于 TensorFlow 2.1 和 2.0 正式版。文中的所有示例代码可至 `这里 <https://github.com/snowkylin/tensorflow-handbook/tree/master/source/_static/code/en>`_ 获得。

本手册正于TensorFlow官方微信公众号（TensorFlow_official）连载，可点此查看 `连载文章目录 <https://mp.weixin.qq.com/s/cvZHUWS3MiGHq3UDynucxw>`_ 。本手册的原始语言为中文，其英文版仍在翻译中。请访问 https://v1.tf.wiki 以查看本手册上一版的英文版。本手册是  `Google Summer of Code 2019 <https://summerofcode.withgoogle.com/archive/2019/projects/5460192307707904/>`_  项目之一。

自2020年4月起，在每章文末加入了留言区，欢迎有需要的读者在文末讨论交流。

.. admonition:: 线上教学活动：ML Study Jam

    本手册正在与TensorFlow官方微信公众号合作开展为期三周的“ML Study Jam”线上学习活动。活动从2020年4月20日开始，可以访问 `这里 <https://tf.wiki/zh/mlstudyjam.html>`_ 或TensorFlow官方微信公众号（TensorFlow_official）以了解详情。

网站：https://tf.wiki

GitHub： https://github.com/snowkylin/tensorflow-handbook

答疑区： https://discuss.tf.wiki

.. toctree:: 
    :maxdepth: 2
    :caption: 目录

    zh/preface
    zh/introduction

.. toctree:: 
    :maxdepth: 3
    :caption: 基础

    zh/basic/installation
    zh/basic/basic
    zh/basic/models
    zh/basic/tools

.. toctree:: 
    :maxdepth: 3
    :caption: 部署

    zh/deployment/export
    zh/deployment/serving
    zh/deployment/lite
    zh/deployment/javascript

.. toctree:: 
    :maxdepth: 3
    :caption: 大规模训练与加速

    zh/appendix/distributed
    zh/appendix/tpu

.. toctree:: 
    :maxdepth: 3
    :caption: 扩展

    zh/appendix/tfhub
    zh/appendix/tfds  
    zh/appendix/swift
    zh/appendix/quantum

.. toctree:: 
    :maxdepth: 3
    :caption: 高级

    zh/advanced/static 
    zh/advanced/tape
    zh/advanced/optimization

.. toctree:: 
    :maxdepth: 3
    :caption: 附录
    
    zh/appendix/rl
    zh/appendix/docker
    zh/appendix/cloud
    zh/appendix/jupyterlab    
    zh/appendix/recommended_books
    zh/appendix/terms

