部署自己的交互式Python開發環境JupyterLab
============================================

如果你既希望獲得本地或雲端強大的計算能力，又希望獲得Jupyter Notebook或Colab中方便的在線Python交互式運行環境，可以自己爲的本地伺服器或雲伺服器安裝JupyterLab。JupyterLab可以理解成升級版的Jupyter Notebook/Colab，提供多標籤頁支持，在線終端和文件管理等一系列方便的功能，接近於一個在線的Python IDE。

.. tip:: 部分雲服務提供了開箱即用的JupyterLab環境，例如前章介紹的 :ref:`GCP中AI Platform的Notebook <notebook>` ，以及 `FloydHub <https://www.floydhub.com/>`_ 。


在已經部署Python環境後，使用以下命令安裝JupyterLab：

::

    pip install jupyterlab

然後使用以下命令運行JupyterLab：

::

    jupyter lab --ip=0.0.0.0

然後根據輸出的提示，使用瀏覽器訪問 ``http://伺服器地址:8888`` ，並使用輸出中提供的token直接登錄（或設置密碼後登錄）即可。

JupyterLab界面如下所示：

.. figure:: /_static/image/jupyterlab/jupyterlab.png
    :width: 100%
    :align: center

.. hint:: 可以使用 ``--port`` 參數指定埠號。

    部分雲服務（如GCP）的實例默認不開放大多數網絡埠。如果使用默認埠號，需要在防火牆設置中打開埠（例如GCP需要在「虛擬機實例詳情-網絡接口-查看詳情」中新建防火牆規則，開放對應埠並應用到當前實例）。

    如果需要在終端退出後仍然持續運行JupyterLab，可以使用 ``nohup`` 命令及 ``&`` 放入後台運行，即：

    ::

        nohup jupyter lab --ip=0.0.0.0 &

    程序輸出可以在當前目錄下的 ``nohup.txt`` 找到。

..
    https://stackoverflow.com/questions/53923773/how-to-run-jupyter-lab-in-a-conda-environment-on-a-google-compute-engine-deep-l

爲了在JupyterLab的Notebook中使用自己的Conda環境，需要使用以下命令：

::

    conda activate 環境名（比如在GCP章節建立的tf2.0-beta-gpu）
    conda install ipykernel
    ipython kernel install --name 環境名 --user

然後重新啓動JupyterLab，即可在Kernel選項和啓動器中建立Notebook的選項中找到自己的Conda環境。

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