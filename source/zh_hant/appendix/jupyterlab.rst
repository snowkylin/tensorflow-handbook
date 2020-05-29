部署自己的互動式 Python 開發環境 JupyterLab
============================================

如果你既希望獲得本機端或雲端強大的計算能力，又希望獲得 Jupyter Notebook 或 Colab 中方便的線上 Python 互動式運行環境，可以自己為的本機伺服器或雲伺服器安裝 JupyterLab。JupyterLab 可以想像成升級版的 Jupyter Notebook/Colab，提供多標簽頁支援，線上終端和文件管理等一系列方便的功能，接近於一個線上版的 Python IDE。

.. tip:: 部分雲服務提供了能夠立即使用的 JupyterLab 環境，例如前章介紹的 :ref:`GCP中AI Platform的Notebook <zh_hant_notebook>` ，以及 `FloydHub <https://www.floydhub.com/>`_ 。


在已經部署 Python 環境後，使用以下命令安裝 JupyterLab：

::

    pip install jupyterlab

然後使用以下命令運行 JupyterLab：

::

    jupyter lab --ip=0.0.0.0

然後根據輸出的提示，使用瀏覽器連線 ``http://伺服器地址:8888`` ，並使用輸出中提供的 token 直接登錄（或設置密碼後登錄）即可。

JupyterLab界面如下所示：

.. figure:: /_static/image/jupyterlab/jupyterlab.png
    :width: 100%
    :align: center

.. hint:: 可以使用 ``--port`` 參數指定埠號。

    部分雲端服務（如 GCP）的實例預設不會開放大多數網路端口。如果使用默認端口號，需要在防火牆設置中開啟端口（例如 GCP 需要在 “虛擬機實例詳情 - 網路接口 - 查看詳情” 中新建防火牆規則，開放對應端口並應用到目前的實例）。
    如果需要在終端退出後仍然持續運行 JupyterLab，可以使用 ``nohup`` 指令及 ``&`` 放入後台運行，即：
    ::

        nohup jupyter lab --ip=0.0.0.0 &

    程式輸出可以在當前目錄下的 ``nohup.txt`` 找到。

..
    https://stackoverflow.com/questions/53923773/how-to-run-jupyter-lab-in-a-conda-environment-on-a-google-compute-engine-deep-l

為了在 JupyterLab 的 Notebook 中使用自己的 Conda 環境，需要使用以下命令：

::

    conda activate  環境名稱（比如在GCP章節建立的tf2.0-beta-gpu）
    conda install ipykernel
    ipython kernel install --name 環境名稱 --user

然後重新啟動 JupyterLab，即可在 Kernel 選項和啟動器中建立 Notebook 的選項中找到自己的 Conda 環境。

.. figure:: /_static/image/jupyterlab/add_env.png
    :width: 100%
    :align: center

    Notebook中新增了「tf2.0-beta-gpu」選項

.. figure:: /_static/image/jupyterlab/kernel.png
    :width: 100%
    :align: center

    可以在Kernel中選擇「tf2.0-beta-gpu」

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 204 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>