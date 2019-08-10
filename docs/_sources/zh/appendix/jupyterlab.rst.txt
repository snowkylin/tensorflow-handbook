部署自己的交互式Python开发环境JupyterLab
============================================

如果你既希望获得本地或云端强大的计算能力，又希望获得Jupyter Notebook或Colab中方便的在线Python交互式运行环境，可以自己为的本地服务器或云服务器安装JupyterLab。JupyterLab可以理解成升级版的Jupyter Notebook/Colab，提供多标签页支持，在线终端和文件管理等一系列方便的功能，接近于一个在线的Python IDE。

.. tip:: 部分云服务提供了开箱即用的JupyterLab环境，例如前章介绍的 :ref:`GCP中AI Platform的Notebook <notebook>` ，以及 `FloydHub <https://www.floydhub.com/>`_ 。


在已经部署Python环境后，使用以下命令安装JupyterLab：

::

    pip install jupyterlab

然后使用以下命令运行JupyterLab：

::

    jupyter lab --ip=0.0.0.0

然后根据输出的提示，使用浏览器访问 ``http://服务器地址:8888`` ，并使用输出中提供的token直接登录（或设置密码后登录）即可。

JupyterLab界面如下所示：

.. figure:: /_static/image/jupyterlab/jupyterlab.png
    :width: 100%
    :align: center

.. hint:: 可以使用 ``--port`` 参数指定端口号。

    部分云服务（如GCP）的实例默认不开放大多数网络端口。如果使用默认端口号，需要在防火墙设置中打开端口（例如GCP需要在“虚拟机实例详情-网络接口-查看详情”中新建防火墙规则，开放对应端口并应用到当前实例）。

    如果需要在终端退出后仍然持续运行JupyterLab，可以使用 ``nohup`` 命令及 ``&`` 放入后台运行，即：

    ::

        nohup jupyter lab --ip=0.0.0.0 &

    程序输出可以在当前目录下的 ``nohup.txt`` 找到。

..
    https://stackoverflow.com/questions/53923773/how-to-run-jupyter-lab-in-a-conda-environment-on-a-google-compute-engine-deep-l

为了在JupyterLab的Notebook中使用自己的Conda环境，需要使用以下命令：

::

    conda activate 环境名（比如在GCP章节建立的tf2.0-beta-gpu）
    conda install ipykernel
    ipython kernel install --name 环境名 --user

然后重新启动JupyterLab，即可在Kernel选项和启动器中建立Notebook的选项中找到自己的Conda环境。

.. figure:: /_static/image/jupyterlab/add_env.png
    :width: 100%
    :align: center

    Notebook中新增了“tf2.0-beta-gpu”选项

.. figure:: /_static/image/jupyterlab/kernel.png
    :width: 100%
    :align: center

    可以在Kernel中选择“tf2.0-beta-gpu”