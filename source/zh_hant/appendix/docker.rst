.. _install_by_docker:

使用Docker部署TensorFlow環境
============================================

.. hint:: 本部分面向沒有Docker經驗的讀者。對於已熟悉Docker的讀者，可直接參考 `TensorFlow官方文檔 <https://www.tensorflow.org/install/docker>`_ 進行部署。

Docker是輕量級的容器（Container）環境，通過將程序放在虛擬的「容器」或者說「保護層」中運行，既避免了配置各種庫、依賴和環境變量的麻煩，又克服了虛擬機資源占用多、啓動慢的缺點。使用Docker部署TensorFlow的步驟如下：

1. 安裝 `Docker <https://www.docker.com/>`_ 。Windows下，下載官方網站的安裝包進行安裝即可。Linux下建議使用 `官方的快速腳本 <https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-using-the-convenience-script>`_ 進行安裝，即命令行下輸入：

::

    wget -qO- https://get.docker.com/ | sh

如果當前的用戶非root用戶，可以執行 ``sudo usermod -aG docker your-user`` 命令將當前用戶加入 ``docker`` 用戶組。重新登錄後即可直接運行Docker。

Linux下通過以下命令啓動Docker服務：

::

    sudo service docker start

2. 拉取TensorFlow映像。Docker將應用程序及其依賴打包在映像文件中，通過映像文件生成容器。使用 ``docker image pull`` 命令拉取適合自己需求的TensorFlow映像，例如：

::

    docker image pull tensorflow/tensorflow:latest-py3        # 最新穩定版本TensorFlow（Python 3.5，CPU版）
    docker image pull tensorflow/tensorflow:latest-gpu-py3    # 最新穩定版本TensorFlow（Python 3.5，GPU版）

更多映像版本可參考 `TensorFlow官方文檔 <https://www.tensorflow.org/install/docker#download_a_tensorflow_docker_image>`_ 。

.. tip:: 可以視網絡環境使用 `DaoCloud的Docker映像鏡像 <https://www.daocloud.io/mirror>`_ 以提高下載速度。


3. 基於拉取的映像文件，創建並啓動TensorFlow容器。使用  ``docker container run`` 命令創建一個新的TensorFlow容器並啓動。

**CPU版本的TensorFlow：**

::

    docker container run -it tensorflow/tensorflow:latest-py3 bash

.. hint::  ``docker container run`` 命令的部分選項如下：

    * ``-it`` 讓docker運行的容器能夠在終端進行交互，具體而言：

        * ``-i`` （ ``--interactive`` ）：允許與容器內的標準輸入 (STDIN) 進行交互。
        * ``-t`` （ ``--tty`` ）：在新容器中指定一個僞終端。

    * ``--rm`` ：當容器中的進程運行完畢後自動刪除容器。
    * ``tensorflow/tensorflow:latest-py3`` ：新容器基於的映像。如果本地不存在指定的映像，會自動從公有倉庫下載。
    * ``bash`` 在容器中運行的命令（進程）。Bash是大多數Linux系統的默認Shell。

**GPU版本的TensorFlow：**

若需在TensorFlow Docker容器中開啓GPU支持，需要具有一塊NVIDIA顯卡並已正確安裝驅動程序（詳見 :ref:`「TensorFlow安裝」一章 <gpu_tensorflow>` ）。同時需要安裝 `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ 。依照官方文檔中的quickstart部分逐行輸入命令即可。

.. warning:: 當前nvidia-docker僅支持Linux。

安裝完畢後，在 ``docker container run`` 命令中添加 ``--runtime=nvidia`` 選項，並基於具有GPU支持的TensorFlow Docker映像啓動容器即可，即：

::

    docker container run -it --runtime=nvidia tensorflow/tensorflow:latest-gpu-py3 bash

.. admonition:: Docker常用命令

    映像（image）相關操作：

    ::

        docker image pull [image_name]  # 從倉庫中拉取映像[image_name]到本機 
        docker image ls                 # 列出所有本地映像
        docker image rm [image_name]    # 刪除名爲[image_name]的本地映像

    容器（container）相關操作：

    ::
        
        docker container run [image_name] [command] # 基於[image_name]映像建立並啓動容器，並運行[command]
        docker container ls                         # 列出本機正在運行的容器
                                                    # （加入--all參數列出所有容器，包括已停止運行的容器）
        docker container rm [container_id]          # 刪除ID爲[container_id]的容器

    Docker入門教程可參考 `阮一峯的Docker入門教程 <http://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html>`_ 和 `Docker Cheat Sheet <https://www.docker.com/sites/default/files/Docker_CheatSheet_08.09.2016_0.pdf>`_ 。

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 202 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>