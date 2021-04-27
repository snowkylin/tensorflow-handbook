.. 简单粗暴TensorFlow documentation master file, created by
   sphinx-quickstart on Sat Jan 20 00:48:15 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================================================
简单粗暴 TensorFlow 2 | A Concise Handbook of TensorFlow 2
==================================================================

*基于Keras和即时执行模式 | Based on Keras and Eager Execution*

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
                <b><a href="/en">English Version</a></b>
            </td>
        </tr>
        </tbody>
    </table>

.. figure:: /_static/image/index/snow_leopard.jpg
        :width: 60%
        :align: center

简体中文版
==========================

这是一本简明的 TensorFlow 2 入门指导手册，基于 Keras 和即时执行模式（Eager Execution），力图让具备一定机器学习及 Python 基础的开发者们快速上手 TensorFlow 2。同时也是纸质版技术手册 `《简明的 TensorFlow 2》 <https://item.jd.com/12980534.html>`_ 的部分草稿。

本手册的所有代码基于 TensorFlow 2.2 正式版。文中的所有示例代码可至 `这里 <https://github.com/snowkylin/tensorflow-handbook/tree/master/source/_static/code>`_ 获得。

本手册正于TensorFlow官方微信公众号（TensorFlow_official）连载，可点此查看 `连载文章目录 <https://mp.weixin.qq.com/mp/appmsgalbum?action=getalbum&__biz=MzU1OTMyNDcxMQ==&scene=23&album_id=1338132220393111552#wechat_redirect>`_ 。本手册的原始语言为简体中文，并有 `繁体中文版 </zh_hant>`_ 和 `英文版 </en>`_ 。本手册是 `Google Summer of Code 2019 <https://summerofcode.withgoogle.com/archive/2019/projects/5460192307707904/>`_  项目之一，并获得 `谷歌开源贡献奖（Google Open Source Peer Bonus） <https://opensource.googleblog.com/2020/10/announcing-latest-google-open-source.html>`_ 。

自2020年4月起，在每章文末加入了留言区，欢迎有需要的读者在文末讨论交流。

**GitHub：**  https://github.com/snowkylin/tensorflow-handbook

.. raw:: html

    <object type="image/svg+xml" data="https://gh-card.dev/repos/snowkylin/tensorflow-handbook.svg?link_target=_blank"></object>

|

**教程答疑区：**  https://discuss.tf.wiki

.. raw:: html

    <a href="https://discuss.tf.wiki" target="_blank"><img src="https://tfugcs.andfun.cn/original/1X/77cdc4166bb18d58a9c19efde029a215612cb461.png" width="300px"/></a>

**纸质完整版：《简明的 TensorFlow 2》** 

.. raw:: html

    <table style="width: 100%; margin: 20px 20px">
        <tbody witth=100%>
        <tr>
            <td width=33% style="text-align: center">
                <img src="/_static/image/index/cover.jpg" width="100%"/>      
            </td>
            <td width=66% style="padding: 20px">
                <p>本书纸质版《简明的 TensorFlow 2》由人民邮电出版社（图灵社区）出版，在本在线手册的基础上进行了细致的编排校对，并增加了若干 TensorFlow 高级专题，全彩印刷，为读者带来更好的阅读体验。</p>
                <p>豆瓣评分：<a href="https://book.douban.com/subject/35217981/" target="_blank">https://book.douban.com/subject/35217981/</a></p>    
                <p>纸质版购买链接：</p>
                <ul>
                    <li><b><a href="https://item.jd.com/12980534.html" target="_blank">京东</a></b></li>
                    <li><b><a href="http://product.dangdang.com/29132630.html" target="_blank">当当</a></b></li>
                    <li><b><a href="https://detail.tmall.com/item.htm?id=628240887768" target="_blank">天猫</a></b></li>
                    <li><b><a href="https://www.ituring.com.cn/book/2705" target="_blank">图灵社区</a></b></li>
                </ul>                                  
            </td>
        </tr>
        </tbody>
    </table>

.. toctree:: 
    :maxdepth: 2
    :caption: 目录

    zh_hans/foreword
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
    :maxdepth: 2
    :caption: 目錄

    zh_hant/foreword
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

The code of this handbook is based on TensorFlow 2.0 stable version and beta1 version. All sample code in this handbook can be viewed `here <https://github.com/snowkylin/tensorflow-handbook/tree/master/source/_static/code/en>`_ .

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

.. toctree:: 
    :maxdepth: 3
    :caption: Large-scale Training

    en/appendix/distributed

.. toctree:: 
    :maxdepth: 3
    :caption: Extensions

    en/appendix/tfds  
    en/appendix/quantum

..
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

