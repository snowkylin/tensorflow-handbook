==================================================================
簡單粗暴 TensorFlow 2 | A Concise Handbook of TensorFlow 2
==================================================================

.. raw:: html

    <table style="width: 100%; margin: 20px 20px">
        <tbody witth=100%>
        <tr>
            <td width=33% style="text-align: center">
                <b><a href="/zh_hans">簡體中文版</a></b>           
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

這是一本簡明的 TensorFlow 2 入門指導手冊，基於 Keras 和即時執行模式（Eager Execution），力圖讓具備一定機器學習及 Python 基礎的開發者們快速上手 TensorFlow 2。

本手冊的所有代碼基於 TensorFlow 2.1 和 2.0 正式版。文中的所有示例代碼可至 `這裡 <https://github.com/snowkylin/tensorflow-handbook/tree/master/source/_static/code/en>`_ 獲得。

本手冊正於TensorFlow官方微信公衆號（TensorFlow_official）連載，可點此查看 `連載文章目錄 <https://mp.weixin.qq.com/s/cvZHUWS3MiGHq3UDynucxw>`_ 。本手冊的原始語言爲中文，其英文版仍在翻譯中。請訪問 https://v1.tf.wiki 以查看本手冊上一版的英文版。本手冊是  `Google Summer of Code 2019 <https://summerofcode.withgoogle.com/archive/2019/projects/5460192307707904/>`_  項目之一。

自2020年4月起，在每章文末加入了留言區，歡迎有需要的讀者在文末討論交流。

GitHub： https://github.com/snowkylin/tensorflow-handbook

答疑區： https://discuss.tf.wiki

.. toctree:: 
    :maxdepth: 2
    :caption: 目錄

    preface
    introduction

.. toctree:: 
    :maxdepth: 3
    :caption: 基礎

    basic/installation
    basic/basic
    basic/models
    basic/tools

.. toctree:: 
    :maxdepth: 3
    :caption: 部署

    deployment/export
    deployment/serving
    deployment/lite
    deployment/javascript

.. toctree:: 
    :maxdepth: 3
    :caption: 大規模訓練與加速

    appendix/distributed
    appendix/tpu

.. toctree:: 
    :maxdepth: 3
    :caption: 擴展

    appendix/tfhub
    appendix/tfds  
    appendix/swift
    appendix/quantum

..
    .. toctree:: 
        :maxdepth: 3
        :caption: 高級

        advanced/static 
        advanced/tape
        advanced/optimization

.. toctree:: 
    :maxdepth: 3
    :caption: 附錄
    
    appendix/rl
    appendix/docker
    appendix/cloud
    appendix/jupyterlab    
    appendix/recommended_books
    appendix/terms

.. only:: html

    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`

    .. raw:: html
    
        <img src="https://s05.flagcounter.com/count2/Hyjs/bg_FFFFFF/txt_000000/border_CCCCCC/columns_2/maxflags_16/viewers_0/labels_1/pageviews_1/flags_0/percent_0/" alt="Flag Counter" border="0">