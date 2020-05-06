.. 简单粗暴TensorFlow documentation master file, created by
   sphinx-quickstart on Sat Jan 20 00:48:15 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================================================
简单粗暴 TensorFlow 2 | A Concise Handbook of TensorFlow 2
==================================================================

*基于Keras和Eager Execution | Based on Keras and Eager Execution*

.. raw:: html

    <table style="width: 100%; margin: 20px 20px">
        <tbody witth=100%>
        <tr>
            <td width=33% style="text-align: center">
                <b><a href="/zh_hans">简体中文版</a></b>           
            </td>
            <td width=33% style="text-align: center">
                <b><a href="/zh_hant">繁體中文版</a></b>           
            </td>
            <td width=33% style="text-align: center">
                <b><a href="/en">English Version<br />(in progress)</a></b>
            </td>
        </tr>
        </tbody>
    </table>

..
    本文档为未完成版本，内容会随时更改修订，目前请不要扩散。

    This document is unfinished, content will be updated rapidly. Please keep it internal at this time.

简体中文版
==========================

这是一本简明的 TensorFlow 2 入门指导手册，基于 Keras 和即时执行模式（Eager Execution），力图让具备一定机器学习及 Python 基础的开发者们快速上手 TensorFlow 2。

本手册的所有代码基于 TensorFlow 2.1 和 2.0 正式版。文中的所有示例代码可至 `这里 <https://github.com/snowkylin/tensorflow-handbook/tree/master/source/_static/code/en>`_ 获得。

本手册正于TensorFlow官方微信公众号（TensorFlow_official）连载，可点此查看 `连载文章目录 <https://mp.weixin.qq.com/s/cvZHUWS3MiGHq3UDynucxw>`_ 。本手册的原始语言为中文，其英文版仍在翻译中。请访问 https://v1.tf.wiki 以查看本手册上一版的英文版。本手册是  `Google Summer of Code 2019 <https://summerofcode.withgoogle.com/archive/2019/projects/5460192307707904/>`_  项目之一。

自2020年4月起，在每章文末加入了留言区，欢迎有需要的读者在文末讨论交流。

.. admonition:: 线上教学活动：ML Study Jam

    本手册正在与TensorFlow官方微信公众号合作开展为期三周的“ML Study Jam”线上学习活动。活动从2020年4月20日开始，可以访问 `这里 <https://tf.wiki/zh_hans/mlstudyjam.html>`_ 或 `TensorFlow官方微信公众号（TensorFlow_official） <http://mp.weixin.qq.com/s?__biz=MzU1OTMyNDcxMQ==&mid=2247488326&idx=1&sn=e5507c80e3419ae30425b7dfac4ce164&chksm=fc18580ecb6fd11808c35c18ed3e61dd39f36d3fbdfcacefaff03e7a5ab6b07b788d1b87e467&mpshare=1&scene=23&srcid=&sharer_sharetime=1587465932630&sharer_shareid=b6f86ab8b392c4d4036aa6a1d3b82824#rd>`_ 以了解详情。

GitHub： https://github.com/snowkylin/tensorflow-handbook

答疑区： https://discuss.tf.wiki

.. toctree:: 
    :maxdepth: 3
    :caption: 教学活动
    
    zh_hans/mlstudyjam

.. toctree:: 
    :maxdepth: 2
    :caption: 目录

    zh_hans/preface
    zh_hans/introduction

.. toctree:: 
    :maxdepth: 3
    :caption: 基础

    zh_hans/basic/installation
    zh_hans/basic/basic
    zh_hans/basic/models
    zh_hans/basic/tools

.. toctree:: 
    :maxdepth: 3
    :caption: 部署

    zh_hans/deployment/export
    zh_hans/deployment/serving
    zh_hans/deployment/lite
    zh_hans/deployment/javascript

.. toctree:: 
    :maxdepth: 3
    :caption: 大规模训练与加速

    zh_hans/appendix/distributed
    zh_hans/appendix/tpu

.. toctree:: 
    :maxdepth: 3
    :caption: 扩展

    zh_hans/appendix/tfhub
    zh_hans/appendix/tfds  
    zh_hans/appendix/swift
    zh_hans/appendix/quantum

..
    .. toctree:: 
        :maxdepth: 3
        :caption: 高级

        zh_hans/advanced/static 
        zh_hans/advanced/tape
        zh_hans/advanced/optimization

.. toctree:: 
    :maxdepth: 3
    :caption: 附录
    
    zh_hans/appendix/rl
    zh_hans/appendix/docker
    zh_hans/appendix/cloud
    zh_hans/appendix/jupyterlab    
    zh_hans/appendix/recommended_books
    zh_hans/appendix/terms

繁体中文版
==========================

.. toctree:: 
    :maxdepth: 3
    :caption: 教學活動
    
    zh_hant/mlstudyjam

.. toctree:: 
    :maxdepth: 2
    :caption: 目錄

    zh_hant/preface
    zh_hant/introduction

.. toctree:: 
    :maxdepth: 3
    :caption: 基礎

    zh_hant/basic/installation
    zh_hant/basic/basic
    zh_hant/basic/models
    zh_hant/basic/tools

.. toctree:: 
    :maxdepth: 3
    :caption: 部署

    zh_hant/deployment/export
    zh_hant/deployment/serving
    zh_hant/deployment/lite
    zh_hant/deployment/javascript

.. toctree:: 
    :maxdepth: 3
    :caption: 大規模訓練與加速

    zh_hant/appendix/distributed
    zh_hant/appendix/tpu

.. toctree:: 
    :maxdepth: 3
    :caption: 擴展

    zh_hant/appendix/tfhub
    zh_hant/appendix/tfds  
    zh_hant/appendix/swift
    zh_hant/appendix/quantum

..
    .. toctree:: 
        :maxdepth: 3
        :caption: 高級

        zh_hant/advanced/static 
        zh_hant/advanced/tape
        zh_hant/advanced/optimization

.. toctree:: 
    :maxdepth: 3
    :caption: 附錄
    
    zh_hant/appendix/rl
    zh_hant/appendix/docker
    zh_hant/appendix/cloud
    zh_hant/appendix/jupyterlab    
    zh_hant/appendix/recommended_books
    zh_hant/appendix/terms

English Version (in progress)
==============================================

This is a concise handbook of TensorFlow 2.0 based on Keras and Eager Execution mode, aiming to help developers with some basic machine learning and Python knowledge to get started with TensorFlow 2.0 quickly.

The code of this handbook is based on TensorFlow 2.0 stable version and beta1 version. All sample code in this handbook can be viewed `here <https://github.com/snowkylin/tensorflow-handbook/tree/master/source/_static/code/>`_ .

The English version of this handbook is still in progress (section title with a ✔️ means that the translation of this section is finished). Please refer to https://v1.tf.wiki for the eariler version. This handbook is a project of `Google Summer of Code 2019 <https://summerofcode.withgoogle.com/archive/2019/projects/5460192307707904/>`_ .

GitHub： https://github.com/snowkylin/tensorflow-handbook

Q&A: https://discuss.tf.wiki

.. toctree:: 
    :maxdepth: 2
    :caption: Preface

    en/preface
    en/introduction

.. toctree:: 
    :maxdepth: 3
    :caption: Basic

    en/basic/installation
    en/basic/basic
    en/basic/models
    en/basic/tools

.. toctree:: 
    :maxdepth: 3
    :caption: Deployment

    en/deployment/export
    en/deployment/serving
    en/deployment/lite
    en/deployment/javascript

.. toctree:: 
    :maxdepth: 3
    :caption: Large-scale Training

    en/appendix/distributed
    en/appendix/tpu

.. toctree:: 
    :maxdepth: 3
    :caption: Extensions

    en/appendix/tfhub
    en/appendix/tfds  
    en/appendix/swift
    en/appendix/quantum

.. toctree:: 
    :maxdepth: 3
    :caption: Appendix

    en/appendix/rl
    en/appendix/docker
    en/appendix/cloud
    en/appendix/jupyterlab
    en/appendix/recommended_books
    en/appendix/terms

.. only:: html

    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`

    .. raw:: html
    
        <img src="https://s05.flagcounter.com/count2/Hyjs/bg_FFFFFF/txt_000000/border_CCCCCC/columns_2/maxflags_16/viewers_0/labels_1/pageviews_1/flags_0/percent_0/" alt="Flag Counter" border="0">

..
    https://info.flagcounter.com/Hyjs

