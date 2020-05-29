.. _zh_hant_install_by_docker:

使用Docker部署TensorFlow環境
============================================

.. hint:: 本章節主要針對沒有 Docker 經驗的讀者們。對於已熟悉 Docker 的讀者，可直接參考`TensorFlow官方文檔 <https://www.tensorflow.org/install/docker>`_ 進行部署。

Docker 是輕量級的容器（Container）環境，通過將程式放在虛擬的 “容器” 或者說 “保護層” 中運行，既避免了配置各種函式庫、相關參數設定和環境變數的麻煩，又克服了虛擬機資源占用太多、啟動慢的缺點。使用 Docker 部署 TensorFlow 的步驟如下：

1. 安裝 `Docker <https://www.docker.com/>`_ 。Windows下，下載官方網站的安裝包進行安裝即可。Linux 下建議使用 `官方的快速腳本 <https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-using-the-convenience-script>`_ 進行安裝，在終端機下輸入：

::

    wget -qO- https://get.docker.com/ | sh

如果當前的用戶非 root 用戶，可以執行 ``sudo usermod -aG docker your-user`` 命令將當前用戶加入 ``docker`` 用戶組。重新登錄後即可直接運行 Docker。

Linux下通過以下命令啓動Docker服務：

::

    sudo service docker start

2. 選取 TensorFlow 映像檔。Docker 將應用程式及其相關參數設定打包在映像文件中，通過映像文件生成容器。使用 ``docker image pull`` 命令拉取適合自己需求的 TensorFlow 映像檔，例如：

::

    docker image pull tensorflow/tensorflow:latest-py3        # 最新穩定版本TensorFlow（Python 3.5，CPU版）
    docker image pull tensorflow/tensorflow:latest-gpu-py3    # 最新穩定版本TensorFlow（Python 3.5，GPU版）

更多映像版本可參考 `TensorFlow官方文檔 <https://www.tensorflow.org/install/docker#download_a_tensorflow_docker_image>`_ 。

.. tip:: 建議使用 `Docker映像鏡像 <https://hub.docker.com/>`_ 將能夠提高下載速度。


3. 基於選取的映像文件，創建並啟動 TensorFlow 容器。使用 ``docker container run`` 命令創建一個新的 TensorFlow 容器並啟動。


**CPU版本的TensorFlow：**

::

    docker container run -it tensorflow/tensorflow:latest-py3 bash

.. hint::  ``docker container run`` 指令的部分選項如下：

    * ``-it`` 讓 docker 運行的容器能夠在終端進行互動，具體而言：

        * ``-i`` （ ``--interactive`` ）：允許與容器內的標準輸入 (STDIN) 進行互動。
        * ``-t`` （ ``--tty`` ）：在新容器中指定一個偽終端。

    * ``--rm`` ：當容器中的進程運行完畢後自動刪除容器。
    * ``tensorflow/tensorflow:latest-py3`` ：新容器基於的映像檔。如果當前不存在指定的映像檔，會自動從網路上下載。
    * ``bash`` 在容器中運行的命令（進程）。Bash 是大多數 Linux 系統的默認 Shell。

**GPU版本的TensorFlow：**

若需在 TensorFlow Docker 容器中開啟 GPU 支援，需要有一塊 NVIDIA 顯示卡並已正確安裝驅動程式（詳見 :ref:`「TensorFlow安裝」一章 <zh_hant_gpu_tensorflow>` ）。同時需要安裝 `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ 。依照官方文件中的 quickstart 部分逐行輸入命令即可。

.. warning:: 當前nvidia-docker僅支援Linux。

安裝完畢後，在 ``docker container run`` 指令中加入 ``--runtime=nvidia`` 選項，並基於具有 GPU 支援的 TensorFlow Docker 映像檔啟動容器即可，如下：

::

    docker container run -it --runtime=nvidia tensorflow/tensorflow:latest-gpu-py3 bash

.. admonition:: Docker常用指令

    映像（image）相關操作：

    ::

        docker image pull [image_name]  # 選取映像檔[image_name]到本機
        docker image ls                 # 列出所有的映像檔
        docker image rm [image_name]    # 刪除名為[image_name]的映像檔

    容器（container）相關操作：

    ::
        
        docker container run [image_name] [command] # 基於[image_name]映像檔建立並啟動容器，並運行[command]
        docker container ls                         # 列出本機正在運行的容器
                                                    # （加入--all參數列出所有容器，包括已停止運行的容器）
        docker container rm [container_id]          # 刪除ID為[container_id]的容器

   Docker 入門教程可參考 `Docker 入門教程 <https://medium.com/unorthodox-paranoid/docker-tutorial-101-c3808b899ac6>`_ 和 `Docker Cheat Sheet <https://www.docker.com/sites/default/files/Docker_CheatSheet_08.09.2016_0.pdf>`_ 。

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