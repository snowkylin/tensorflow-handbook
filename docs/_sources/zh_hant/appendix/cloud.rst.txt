在雲端使用TensorFlow
============================================

.. _zh_hant_colab:

在Colab中使用TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Google Colab 是谷歌的免費線上互動式 Python 運行環境，且提供 GPU 支持，使得機器學習開發者們無需在自己的電腦上安裝環境，就能隨時隨地從雲端使用和運行自己的機器學習程式碼。


.. admonition:: 學習資源

    - `Colab官方文件 <https://colab.research.google.com/notebooks/welcome.ipynb>`_
    - `Google Colab Free GPU Tutorial <https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d>`_ （`中文翻譯 <https://juejin.im/post/5c05e1bc518825689f1b4948>`_）
    - `五分鐘學會在Colab上使用免費的TPU訓練模型 <https://dataology.blogspot.com/2020/03/colabtpu.html>`_
進入Colab（https://colab.research.google.com），新建一個Python3筆記本，界面如下：

.. figure:: /_static/image/colab/new_ch.png
    :width: 100%
    :align: center

如果需要使用 GPU，則點選上方 Edit  “Notebook settings - 更改運行時類型”，在 “硬體加速器” 一項中選擇 “GPU”，如下圖所示：


.. figure:: /_static/image/colab/select_env_ch.png
    :width: 40%
    :align: center

我們在主界面輸入一行程式碼，例如 ``import tensorflow as tf`` ，然後按 ``ctrl + enter`` 執行代碼（如果直接按下 ``enter`` 是換行，可以一次輸入多行代碼並執行）。此時 Colab 會自動連接到雲端的執行環境，並將狀態顯示在右上角。

執行完後，點擊界面左上角的 “+ Code”，此時界面上會新增一個輸入框，我們輸入 ``tf.__version__`` ，再次按下 ``ctrl + enter`` 執行代碼，以查看 Colab 預設的 TensorFlow 版本，執行情況如下：


.. figure:: /_static/image/colab/tf_version_ch.png
    :width: 100%
    :align: center

.. tip:: Colab 支援程式碼提示功能，可以在輸入 ``tf.`` 後按下 ``tab`` 鍵，將會出現程式碼提示的下拉選單。



可見，到目前本文程式撰寫，Colab 中的 TensorFlow 預設版本是 2.2.0。在 Colab 中，可以使用 ``!pip install`` 或者 ``!apt-get install`` 來安裝 Colab 中尚未安裝的 Python 函式庫或 Linux 軟體套件。比如在這裡，我們希望使用 TensorFlow 2.2.0rc4 的版本，即點擊左上角的 “+ Code”，輸入:


    !pip install tensorflow-gpu==2.0.0-beta1

按下 ``ctrl + enter`` 執行，結果如下：


.. figure:: /_static/image/colab/install_tf_ch.png
    :width: 100%
    :align: center

能夠發現，Colab 提示我們重新運行環境以使用新安裝的 TensorFlow 版本。於是我們點擊運行框最下方的 Restart Runtime（或者選單 “Runtime - Restart Runtime”），然後再次導入 TensorFlow 並查看版本，結果如下：


.. figure:: /_static/image/colab/view_tf_version_ch.png
    :width: 100%
    :align: center

我們可以使用 ``tf.test.is_gpu_available`` 函數來查看當前環境的 GPU 是否可用：

.. figure:: /_static/image/colab/view_gpu_ch.png
    :width: 100%
    :align: center

可見，我們成功在 Colab 中配置了 TensorFlow 2.0 環境並啟用了 GPU 支援。

你甚至可以透過 ``!nvidia-smi`` 指令查看目前的 GPU 資訊:


.. figure:: /_static/image/colab/nvidia_smi_ch.png
    :width: 100%
    :align: center

可見 GPU 的型號為 Tesla T4。

.. _zh_hant_GCP:

在Google Cloud Platform（GCP）中使用TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://medium.com/@kstseng/%E5%9C%A8-google-cloud-platform-%E4%B8%8A%E4%BD%BF%E7%94%A8-gpu-%E5%92%8C%E5%AE%89%E8%A3%9D%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E7%9B%B8%E9%97%9C%E5%A5%97%E4%BB%B6-1b118e291015
    
`Google Cloud Platform（GCP） <https://cloud.google.com/>`_ 是 Google 的雲端計算服務。GCP 收費靈活，預設依照使用時間計費。也就是說，你可以快速建立一個支援 GPU 的實例，訓練一個模型，然後立即關閉（關機或刪除實例）。GCP 只收取在實例開啟時所產生的費用，關機時只收取資料存儲的費用，刪除後即不再繼續收費。


我們可以通過兩種方式在 GCP 中使用 TensorFlow：使用 Compute Engine 建立支援GPU 的實例，或使用 AI Platform 中的 Notebook 建立資源 GPU 的線上 JupyterLab 環境。


在 Compute Engine 建立支援 GPU 的實例並部署 TensorFlow
----------------------------------------------------------------

GCP 的 Compute Engine 類似於 AWS、阿里雲等，允許使用者快速建立自己的虛擬機實例。在 Compute Engine 中，可以很方便的建立具有 GPU 的虛擬機實例，只需要進入 Compute Engine 的 VM 實例（https://console.cloud.google.com/compute/instances），並在創建實例的時候選擇 GPU 類型和數量即可。


.. figure:: /_static/image/gcp/create_instance_ch.png
    :width: 100%
    :align: center

需要注意兩點：

1. 只有特定區域的機房具有 GPU，且不同類型的 GPU 地區範圍也不同，可參考`GCP官方文件 <https://cloud.google.com/compute/docs/gpus>`_ 並選擇適合的地區建立實例；
#. 預設情況下 GCP 帳號的 GPU 配額非常有限（可能是怕你付不起錢？）。你很可能需要在使用前申請提升自己帳號在特定地區的特定型號 GPU 的配額，可參考 `GCP 官方文件：申請提升配額 <https://cloud.google.com/compute/quotas#requesting_additional_quota>`_ ，GCP 會有工作人員手動處理申請，並給你的信箱發送信件通知，大約需要數小時至兩個工作日不等。

當建立好支援 GPU 的 GCP 虛擬機實例後，配置工作與在本機端大致相同。系統中預設並沒有 NVIDIA 顯卡驅動，依然需要自己安裝。

以下指令範例在 Tesla K80，Ubuntu 18.04 LTS 的 GCP 虛擬機實例中配置 NVIDIA 410 驅動、CUDA 10.0、cuDNN 7.6.0 以及 TensorFlow 2.0 beta 環境的過程：

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

.. _zh_hant_notebook:

使用 AI Platform 中的 Notebook 建立資源GPU 的線上 JupyterLab 環境 
----------------------------------------------------------------

如果你不希望繁雜的設定，希望快速獲得一個能夠立即使用的線上互動式 Python 環境，可以使用 GCP 的 AI Platform 中的 Notebook。其預先安裝了 JupyterLab，可以理解為 Colab 的付費升級版，具備更多功能且限制較少。

進入 https://console.cloud.google.com/mlengine/notebooks ，點擊 “新建實例 - TensorFlow 2.0-With 1 NVIDIA Tesla K80”，界面如下：

.. figure:: /_static/image/gcp/create_notebook_ch.png
    :width: 100%
    :align: center

也可以按下 “自定義” 來進一步設定實例，例如選擇區域、GPU 類型和個數，與創建 Compute Engine 實例類似。

.. hint:: 和 Compute Engine 實例一樣，你很可能需要在這裡選擇自己適合的區域，以及申請提升自己帳號在特定地區的特定型號 GPU 的配額。

建立完成後，點擊 “打開 JUPYTERLAB”，即可進入以下界面：

.. figure:: /_static/image/gcp/notebook_index_ch.png
    :width: 100%
    :align: center

建立一個Python 3筆記本，測試TensorFlow環境：

.. figure:: /_static/image/gcp/notebook_test_ch.png
    :width: 100%
    :align: center

我們還可以按下左上角的 “+” 號，新建一個終端：

.. figure:: /_static/image/gcp/notebook_terminal_ch.png
    :width: 100%
    :align: center

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
