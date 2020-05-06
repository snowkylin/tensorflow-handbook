在雲端使用TensorFlow
============================================

.. _colab:

在Colab中使用TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Google Colab是谷歌的免費在線交互式Python運行環境，且提供GPU支持，使得機器學習開發者們無需在自己的電腦上安裝環境，就能隨時隨地從雲端訪問和運行自己的機器學習代碼。

.. admonition:: 學習資源

    - `Colab官方教程 <https://colab.research.google.com/notebooks/welcome.ipynb>`_
    - `Google Colab Free GPU Tutorial <https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d>`_ （`中文翻譯 <https://juejin.im/post/5c05e1bc518825689f1b4948>`_）

進入Colab（https://colab.research.google.com），新建一個Python3筆記本，界面如下：

.. figure:: /_static/image/colab/new.png
    :width: 100%
    :align: center

如果需要使用GPU，則點擊菜單「代碼執行程序-更改運行時類型」，在「硬件加速器」一項中選擇「GPU」，如下圖所示：

.. figure:: /_static/image/colab/select_env.png
    :width: 40%
    :align: center

我們在主界面輸入一行代碼，例如 ``import tensorflow as tf`` ，然後按 ``ctrl + enter`` 執行代碼（如果直接按下 ``enter`` 是換行，可以一次輸入多行代碼並運行）。此時Colab會自動連接到雲端的運行環境，並將狀態顯示在右上角。

運行完後，點擊界面左上角的「+代碼」，此時界面上會新增一個輸入框，我們輸入 ``tf.__version__`` ，再次按下 ``ctrl + enter`` 執行代碼，以查看Colab默認的TensorFlow版本，執行情況如下：

.. figure:: /_static/image/colab/tf_version.png
    :width: 100%
    :align: center

.. tip:: Colab支持代碼提示，可以在輸入 ``tf.`` 後按下 ``tab`` 鍵，即會彈出代碼提示的下拉菜單。

可見，截至本文寫作時，Colab中的TensorFlow默認版本是1.14.0。在Colab中，可以使用 ``!pip install`` 或者 ``!apt-get install`` 來安裝Colab中尚未安裝的Python庫或Linux軟件包。比如在這裡，我們希望使用TensorFlow 2.0 beta1版本，即點擊左上角的「+代碼」，輸入::

    !pip install tensorflow-gpu==2.0.0-beta1

按下 ``ctrl + enter`` 執行，結果如下：

.. figure:: /_static/image/colab/install_tf.png
    :width: 100%
    :align: center

可見，Colab提示我們重啓運行環境以使用新安裝的TensorFlow版本。於是我們點擊運行框最下方的Restart Runtime（或者菜單「代碼執行程序-重新啓動代碼執行程序」），然後再次導入TensorFlow並查看版本，結果如下：

.. figure:: /_static/image/colab/view_tf_version.png
    :width: 100%
    :align: center

我們可以使用 ``tf.test.is_gpu_available`` 函數來查看當前環境的GPU是否可用：

.. figure:: /_static/image/colab/view_gpu.png
    :width: 100%
    :align: center

可見，我們成功在Colab中配置了TensorFlow 2.0環境並啓用了GPU支持。

你甚至可以通過 ``!nvidia-smi`` 查看當前的GPU信息：

.. figure:: /_static/image/colab/nvidia_smi.png
    :width: 100%
    :align: center

可見GPU的型號爲Tesla T4。

.. _GCP:

在Google Cloud Platform（GCP）中使用TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://medium.com/@kstseng/%E5%9C%A8-google-cloud-platform-%E4%B8%8A%E4%BD%BF%E7%94%A8-gpu-%E5%92%8C%E5%AE%89%E8%A3%9D%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E7%9B%B8%E9%97%9C%E5%A5%97%E4%BB%B6-1b118e291015
    
`Google Cloud Platform（GCP） <https://cloud.google.com/>`_ 是Google的雲計算服務。GCP收費靈活，默認按時長計費。也就是說，你可以迅速建立一個帶GPU的實例，訓練一個模型，然後立即關閉（關機或刪除實例）。GCP只收取在實例開啓時所產生的費用，關機時只收取磁盤存儲的費用，刪除後即不再繼續收費。

我們可以通過兩種方式在GCP中使用TensorFlow：使用Compute Engine建立帶GPU的實例，或使用AI Platform中的Notebook建立帶GPU的在線JupyterLab環境。

在Compute Engine建立帶GPU的實例並部署TensorFlow
----------------------------------------------------------------

GCP的Compute Engine類似於AWS、阿里雲等，允許用戶快速建立自己的虛擬機實例。在Compute Engine中，可以很方便地建立具有GPU的虛擬機實例，只需要進入Compute Engine的VM實例（https://console.cloud.google.com/compute/instances），並在創建實例的時候選擇GPU類型和數量即可。

.. figure:: /_static/image/gcp/create_instance.png
    :width: 100%
    :align: center

需要注意兩點：

1. 只有特定區域的機房具有GPU，且不同類型的GPU地區範圍也不同，可參考 `GCP官方文檔 <https://cloud.google.com/compute/docs/gpus>`_ 並選擇適合的地區建立實例；
#. 默認情況下GCP賬號的GPU配額非常有限（可能是怕你付不起錢？）。你很可能需要在使用前申請提升自己賬號在特定地區的特定型號GPU的配額，可參考 `GCP官方文檔：申請提升配額 <https://cloud.google.com/compute/quotas?hl=zh-cn#requesting_additional_quota>`_ ，GCP會有工作人員手動處理申請，並給你的郵箱發送郵件通知，大約需要數小時至兩個工作日不等。

當建立好具有GPU的GCP虛擬機實例後，配置工作與在本地基本相同。系統中默認並沒有NVIDIA顯卡驅動，依然需要自己安裝。

以下命令示例了在Tesla K80，Ubuntu 18.04 LTS的GCP虛擬機實例中配置NVIDIA 410驅動、CUDA 10.0、cuDNN 7.6.0以及TensorFlow 2.0 beta環境的過程：

.. code-block:: bash

    sudo apt-get install build-essential    # 安裝編譯環境
    wget http://us.download.nvidia.com/tesla/410.104/NVIDIA-Linux-x86_64-410.104.run    # 下載NVIDIA驅動
    sudo bash NVIDIA-Linux-x86_64-410.104.run   # 安裝驅動（一路Next）
    # nvidia-smi  # 查看虛擬機中的GPU型號
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  # 下載Miniconda
    bash Miniconda3-latest-Linux-x86_64.sh      # 安裝Miniconda（安裝完需要重啓終端）
    conda create -n tf2.0-beta-gpu python=3.6
    conda activate tf2.0-beta-gpu
    conda install cudatoolkit=10.0
    conda install cudnn=7.6.0
    pip install tensorflow-gpu==2.0.0-beta1

輸入 ``nvidia-smi`` 會顯示：

.. code-block:: bash

    ~$ nvidia-smi
    Fri Jul 12 10:30:37 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.104      Driver Version: 410.104      CUDA Version: 10.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   63C    P0    88W / 149W |      0MiB / 11441MiB |    100%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

.. _notebook:

使用AI Platform中的Notebook建立帶GPU的在線JupyterLab環境
----------------------------------------------------------------

如果你不希望繁雜的配置，希望迅速獲得一個開箱即用的在線交互式Python環境，可以使用GCP的AI Platform中的Notebook。其預安裝了JupyterLab，可以理解爲Colab的付費升級版，具備更多功能且限制較少。

進入 https://console.cloud.google.com/mlengine/notebooks ，點擊「新建實例-TensorFlow 2.0-With 1 NVIDIA Tesla K80」，界面如下：

.. figure:: /_static/image/gcp/create_notebook.png
    :width: 100%
    :align: center

也可以點擊「自定義」來進一步配置實例，例如選擇區域、GPU類型和個數，與創建Compute Engine實例類似。

.. hint:: 和Compute Engine實例一樣，你很可能需要在這裡選擇自己適合的區域，以及申請提升自己賬號在特定地區的特定型號GPU的配額。

建立完成後，點擊「打開JUPYTERLAB」，即可進入以下界面：

.. figure:: /_static/image/gcp/notebook_index.png
    :width: 100%
    :align: center

建立一個Python 3筆記本，測試TensorFlow環境：

.. figure:: /_static/image/gcp/notebook_test.png
    :width: 100%
    :align: center

我們還可以點擊左上角的「+」號，新建一個終端：

.. figure:: /_static/image/gcp/notebook_terminal.png
    :width: 100%
    :align: center

在阿里雲上使用 GPU 實例運行 Tensorflow（Ziyang）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

國內也有部分雲服務商（如 `阿里雲 <https://cn.aliyun.com/product/ecs/gpu>`_ 和 `騰訊雲 <https://cloud.tencent.com/product/gpu>`_ ）提供了 GPU 實例，且可按量計費。至本手冊撰寫時，具備單個GPU的實例價格在數元（Tesla P4）至二十多元（Tesla V100）每小時不等。以下我們簡要介紹在阿里雲使用 GPU 實例。

.. hint:: 根據不同的地區、配置和付費方式，實例的價格也是多樣化的，請根據需要合理選擇。如果是臨時需要的計算任務，可以考慮按量付費以及使用搶占式VPS，以節約資金。

訪問 https://cn.aliyun.com/product/ecs/gpu ，點擊購買，界面如下：

.. figure:: /_static/image/aliyun/vps_select.png
    :width: 100%
    :align: center

此處，我們選擇一個帶有 Tesla P4 計算卡的實例。

在系統鏡像中，阿里雲提供多種選擇，可以根據需要選擇合適的鏡像。

.. figure:: /_static/image/aliyun/os_image_config_with_driver.png
    :width: 100%
    :align: center

    如果選擇「公共鏡像」，可以根據提示選擇提前預裝GPU驅動，可以避免後續安裝驅動的麻煩。

.. figure:: /_static/image/aliyun/os_image_with_RAPIDS.png
    :width: 100%
    :align: center

    在「鏡像市場」中，官方也提供了適合深度學習的定製鏡像。在本示例中我們選擇預裝了 NVIDIA RAPIDS 的 Ubuntu 16.04 鏡像。 

然後，通過 ssh 連接上我們選購的服務器，並使用 ``nvidia-smi`` 查看 GPU 信息：

.. code-block:: bash

    (rapids) root@iZ8vb2567465uc1ty3f4ovZ:~# nvidia-smi
    Sun Aug 11 23:53:52 2019
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla P4            On   | 00000000:00:07.0 Off |                    0 |
    | N/A   29C    P8     6W /  75W |      0MiB /  7611MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

確認了驅動無誤之後，其他操作就可以照常執行了。

.. hint:: 阿里雲等雲服務提供商一般對於 VPS 的端口進行了安全策略限制，請關注所使用的端口是否在安全策略的放行列表中，以免影響Tensorflow Serving和Tensorboard的使用。

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 203 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>