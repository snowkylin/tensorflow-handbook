TensorFlow 安裝與環境配置
======================================

TensorFlow 的最新安裝步驟可參考官方網站上的說明（https://tensorflow.google.cn/install）。TensorFlow支援Python、Java、Go、C等多種程式語言以及 Windows、OSX、Linux 等多種作業系統，此處及後文均以 Python 3.7 為準。

.. hint:: 本章介紹在一般的個人電腦或伺服器上直接安裝 TensorFlow 2 的方法。關於在容器環境（Docker）、雲端平台中部署 TensorFlow 或在線上環境中使用 TensorFlow 的方法，見附錄 :doc:`使用 Docker 部署 TensorFlow 環境 <../appendix/docker>` 和 :doc:`在雲端使用 TensorFlow <../appendix/cloud>` 。軟體的安裝方法往往具有時效性，本節的更新日期為 2019 年 10 月。

一般安裝步驟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. 安裝 Python 環境。此處建議安裝 `Anaconda <https://www.anaconda.com/>`_ 的 Python 3.7 版本（後文均以此為準），這是一個開源的 Python 發行版本，提供了一個完整的科學計算環境，包括 NumPy、SciPy 等常用科學計算套件。當然，你有權選擇自己喜歡的 Python 環境。Anaconda 的安裝包可在 `這裡 <https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/>`_ 獲得。

2. 使用 Anaconda 內建的 conda 套件管理器建立一個 Conda 虛擬環境，並進入該虛擬環境。在命令列下輸入：

::

    conda create --name tf2 python=3.7      # “tf2”是你建立的conda虛擬環境的名字
    conda activate tf2                      # 進入名為“tf2”的conda虛擬環境

3. 使用 Python 套件管理器 pip 安裝 TensorFlow。在命令列下輸入：

::

    pip install tensorflow

等待片刻即安裝完畢。

.. tip:: 

    1. 也可以使用 ``conda install tensorflow`` 來安裝 TensorFlow，不過 conda 來源的版本往往更新較慢，難以第一時間獲得最新的 TensorFlow 版本；
    2. 從 TensorFlow 2.1 開始，透過pip 安裝 ``tensorflow`` 即同時包含 GPU 支援，無需通過特定的 pip ``tensorflow-gpu`` 安裝 GPU 版本。如果對 pip 安裝之檔案大小敏感，可使用 ``tensorflow-cpu`` 安裝僅支援 CPU 的 TensorFlow 版本。
    3. 在 Windows 下，需要打開開始介面中的 “Anaconda Prompt” 進入 Anaconda 的命令列環境；
    4. pypi 和 Anaconda 網址連結如下；
        
        - pypi：https://pypi.org/
        - Anaconda：https://www.anaconda.com/
    5. 如果對硬碟空間要求嚴格（比如伺服器環境），可以安裝 `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ ，僅包含 Python 和 Conda，其他的的套件可自己按需安裝。Miniconda 的安裝包可在 `這裡 <https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/>`_ 下載。

.. admonition:: pip 和 conda 套件管理器

    pip 是最為廣泛使用的 Python 套件管理器，可以幫助我們獲得最新的 Python 套件並進行管理。常用指令如下：

    ::

        pip install [package-name]              # 安裝名為[package-name]的套件
        pip install [package-name]==X.X         # 安裝名為[package-name]的套件並指定版本X.X
        pip install [package-name] --proxy=代理伺服器IP:埠號         # 使用代理伺服器安裝
        pip install [package-name] --upgrade    # 更新名為[package-name]的的套件
        pip uninstall [package-name]            # 刪除名為[package-name]的的套件
        pip list                                # 列出當前環境下已安裝的所有套件
    
    conda 套件管理器是 Anaconda 內建的的套件管理器，可以幫助我們在 conda 環境下輕鬆地安裝各種套件。相較於 pip 而言，conda 的通用性更強（不僅是 Python 套件，其他套件如 CUDA Toolkit 和 cuDNN 也可以安裝），但 conda 來源的版本更新往往較慢。常用指令如下：

    ::

        conda install [package-name]        # 安裝名為[package-name]的套件
        conda install [package-name]=X.X    # 安裝名為[package-name]的套件並指定版本X.X
        conda update [package-name]         # 更新名為[package-name]的的套件
        conda remove [package-name]         # 刪除名為[package-name]的的套件
        conda list                          # 列出當前環境下已安裝的所有的套件
        conda search [package-name]         # 列出名為[package-name]的的套件在conda源中的所有可用版本

    conda 中配置代理：在主目錄下的 .condarc 文件中增加以下內容：

    ::

        proxy_servers:
            http: http://代理伺服器IP:埠號

.. admonition:: Conda虛擬環境

    在 Python 開發中，很多時候我們希望每個應用有一個獨立的 Python 環境（比如應用 1 需要用到 TensorFlow 1.X，而應用 2 使用 TensorFlow 2.0）。這時，Conda 虛擬環境即可為一個應用創建一套 “隔離” 的 Python 運行環境。使用 Python 的的套件管理器 conda 即可輕鬆地創建 Conda 虛擬環境。常用指令如下：

    ::

        conda create --name [env-name]      # 建立名為[env-name]的Conda虛擬環境
        conda activate [env-name]           # 進入名為[env-name]的Conda虛擬環境
        conda deactivate                    # 退出當前的Conda虛擬環境
        conda env remove --name [env-name]  # 刪除名為[env-name]的Conda虛擬環境
        conda env list                      # 列出所有Conda虛擬環境

.. _gpu_tensorflow:

GPU 版本 TensorFlow 安裝指南
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GPU 版本的 TensorFlow 可以利用 NVIDIA GPU 強大的計算加速能力，使 TensorFlow 的運行更為高效，尤其是可以成倍提升模型訓練的速度。

在安裝 GPU 版本的 TensorFlow 前，你需要具有一塊不太舊的 NVIDIA 顯卡，以及正確安裝 NVIDIA 顯卡驅動程式、CUDA Toolkit 和 cuDNN。

GPU 硬體的準備
-------------------------------------------

TensorFlow 對 NVIDIA 顯卡的支援較為完備。對於 NVIDIA 顯卡，要求其 CUDA Compute Capability 須不低於 3.5，可以到 `NVIDIA 的官方網站 <https://developer.nvidia.com/cuda-gpus/>`_ 查詢自己所用顯卡的 CUDA Compute Capability。目前，AMD 的顯卡也開始對 TensorFlow 提供支援，可參考  `這篇文章 <https://medium.com/tensorflow/amd-rocm-gpu-support-for-tensorflow-33c78cc6a6cf>`_  查看詳情。

NVIDIA 驅動程式的安裝
-------------------------------------------

**Windows** 

Windows 環境中，如果系統具有 NVIDIA 顯卡，則往往已經自動安裝了 NVIDIA 顯卡驅動程式。如未安裝，直接到 `NVIDIA 官方網站 <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_ 下載並安裝對應型號的最新公版驅動程式即可。

**Linux** 

在伺服器版 Linux 系統下，同樣訪問 `NVIDIA 官方網站 <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_ 下載驅動程式（為 ``.run`` 文件），並使用 ``sudo bash DRIVER_FILE_NAME.run`` 指令安裝驅動程式即可。在安裝之前，可能需要使用 ``sudo apt-get install build-essential`` 安裝合適的編譯環境。

在具有圖形界面的桌面版 Linux 系統上，NVIDIA 顯卡驅動程式需要一些額外的配置，否則會出現無法登錄等各種錯誤。如果需要在 Linux 下手動安裝 NVIDIA 驅動程式，注意在安裝前進行以下步驟（以 Ubuntu 為例）：

- 禁用系統內建的開源顯卡驅動程式 Nouveau（在 ``/etc/modprobe.d/blacklist.conf`` 文件中添加一行 ``blacklist nouveau`` ，使用 ``sudo update-initramfs -u`` 更新內核，並重啟電腦）
- 禁用主板的 Secure Boot 功能
- 停用桌面環境（如 ``sudo service lightdm stop``）
- 刪除原有 NVIDIA 驅動程式（如 ``sudo apt-get purge nvidia*``）

.. tip:: 對於桌面版 Ubuntu 系統，有一個很簡易的 NVIDIA 驅動程式安裝方法：在系統設置（System Setting）裡面選軟體與更新（Software & Updates），然後點選 Additional Drivers 裡面的 “Using NVIDIA binary driver” 選項並點選右下角的 “Apply Changes” 即可，系統即會自動安裝 NVIDIA 驅動程式，但是通過這種安裝方式安裝的 NVIDIA 驅動程式往往版本較舊。

NVIDIA 驅動程式安裝完成後，可在終端機下使用 ``nvidia-smi`` 指令檢查是否安裝成功，若成功則會顯示出當前系統安裝的 NVIDIA 驅動程式資訊，形式如下：

::
    
    $ nvidia-smi
    Mon Jun 10 23:19:54 2019
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 419.35       Driver Version: 419.35       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 106... WDDM  | 00000000:01:00.0  On |                  N/A |
    | 27%   51C    P8    13W / 180W |   1516MiB /  6144MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0       572    C+G   Insufficient Permissions                   N/A      |
    +-----------------------------------------------------------------------------+

.. hint:: 指令 ``nvidia-smi`` 可以查看機器上現有的 GPU 及使用情況。（在 Windows 下，將 ``C:\Program Files\NVIDIA Corporation\NVSMI`` 加入 Path 環境變數中即可，或 Windows 10 下可使用工作管理員的 “效能” 標籤查看顯卡資訊）

更詳細的 GPU 環境配置指導可以參考 `這篇文章 <https://www.linkedin.com/pulse/installing-nvidia-cuda-80-ubuntu-1604-linux-gpu-new-victor/>`_ 和 `這篇中文文章 <https://medium.com/@maniac.tw/%E5%9C%A8-ubuntu-14-04-16-04-%E5%AE%89%E8%A3%9D-nvidia-%E9%A1%AF%E5%8D%A1%E9%A9%85%E5%8B%95%E7%A8%8B%E5%BC%8F-cuda-toolkit-%E5%8F%8A-cudnn-875e294530ed>`_ 。

CUDA Toolkit 和 cuDNN 的安裝
-------------------------------------------

在 Anaconda 環境下，推薦使用

::

    conda install cudatoolkit=X.X
    conda install cudnn=X.X.X

安裝 CUDA Toolkit 和 cuDNN，其中 X.X 和 X.X.X 分別為需要安裝的 CUDA Toolkit 和 cuDNN 版本號，必須嚴格按照 `TensorFlow 官方網站所說明的版本 <https://www.tensorflow.org/install/gpu#software_requirements>`_ 安裝。例如，對於 TensorFlow 2.1，可使用::

    conda install cudatoolkit=10.1
    conda install cudnn=7.6.5

在安裝前，可使用 ``conda search cudatoolkit`` 和 ``conda search cudnn`` 搜尋 conda 能夠支援的版本號。

當然，也可以按照 `TensorFlow 官方網站上的說明 <https://www.tensorflow.org/install/gpu>`_ 手動下載 CUDA Toolkit 和 cuDNN 並安裝，不過過程會稍繁瑣。

使用 conda 套件管理器安裝 GPU 版本的 TensorFlow 時，會自動安裝對應版本的 CUDA Toolkit 和 cuDNN。conda 來源的更新往往較慢，如果對版本不太介意，也可以直接使用 ``conda install tensorflow-gpu`` 進行安裝。

第一個程式
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

安裝完畢後，我們來編寫一個簡單的程式來驗證安裝。

在終端機下輸入 ``conda activate tf2`` 進入之前建立的安裝有 TensorFlow 的 Conda 虛擬環境，再輸入 ``python`` 進入 Python 環境，逐行輸入以下程式碼：

.. code-block:: python

    import tensorflow as tf

    A = tf.constant([[1, 2], [3, 4]])
    B = tf.constant([[5, 6], [7, 8]])
    C = tf.matmul(A, B)

    print(C)

如果能夠最終輸出::

    tf.Tensor(
    [[19 22]
    [43 50]], shape=(2, 2), dtype=int32)

說明 TensorFlow 已安裝成功。運行途中可能會輸出一些 TensorFlow 的提示資訊，屬於正常現象。

.. warning:: 如果你在 Windows 下安裝了 TensorFlow 2.1 正式版，可能會在導入 TensorFlow 時出現 `DLL載入錯誤 <https://github.com/tensorflow/tensorflow/issues/35749>`_ 。此時安裝 `Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019 <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_ 即可正常使用。

此處使用的是 Python 語言，關於 Python 語言的入門教程可以參考 `中文 Python 3 教程 <https://openhome.cc/Gossip/CodeData/PythonTutorial/index.html>`_ 或 `英文版 Python 教程 <https://www.w3schools.com/python/>`_ ，本手冊之後將預設讀者擁有 Python 語言的基本知識。不用緊張，Python 語言易於上手，而 TensorFlow 本身也不會用到 Python 語言太多高級的複雜應用。

IDE 設置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

對於機器學習的研究員和從業者，建議使用 `PyCharm <http://www.jetbrains.com/pycharm/>`_ 作為 Python 開發的 IDE。

在新建項目時，你需要選定項目的 Python Interpreter，也就是用怎樣的 Python 環境來運行你的項目。在安裝部分，你所建立的每個 Conda 虛擬環境其實都有一個自己獨立的 Python Interpreter，你只需要將它們添加進來即可。選擇 “Add”，並在接下來的視窗選擇 “Existing Environment”，在 Interpreter 處選擇 ``Anaconda安裝目錄/envs/所需要添加的Conda環境名字/python.exe`` （Linux 下無 ``.exe`` 副檔名）並按下 “OK” 即可。如果選中了 “Make available to all projects”，則在所有項目中都可以選擇該 Python Interpreter。注意，在 Windows 下 Anaconda 的預設安裝目錄比較特殊，一般為  ``C:\Users\用户名\Anaconda3\`` 或 ``C:\Users\用户名\AppData\Local\Continuum\anaconda3`` 。此處 ``AppData`` 是隱藏資料夾。

對於 TensorFlow 開發而言，PyCharm 的 Professonal 版本非常有用的一個特性是 **遠端偵測** （Remote Debugging）。當你編寫程式的終端機性能有限，但又有一台可遠端ssh 訪問的高性能電腦（一般具有高性能 GPU）時，遠端偵錯功能可以讓你在終端機編寫程式的同時，在遠端電腦上除錯與運行程式（尤其是訓練模型）。你在終端機上對程式碼和資料修改可以自動同步到遠端電腦中，在實際使用的過程中如同在遠端電腦上編寫程式一般，與串流遊戲有異曲同工之妙。不過遠端偵錯對網路的穩定性要求高，如果需要長時間訓練模型，建議登錄遠端電腦的終端直接訓練模型（Linux 下可以結合 ``nohup`` 指令 [#nohup]_ ，讓進程在後端運行，不受終端退出的影響）。遠端除錯功能的具體配置步驟見 `PyCharm文件 <https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html>`_ 。

.. tip:: 如果你是學生並有.edu 結尾的信箱的話，可以在 `這裡 <http://www.jetbrains.com/student/>`_ 申請 PyCharm 的免費 Professional 版本授權。

對於 TensorFlow 及深度學習的業餘愛好者或者初學者， `Visual Studio Code <https://code.visualstudio.com/>`_ 或者一些線上的交互式 Python 環境（比如免費的 `Google Colab <https://colab.research.google.com/>`_ ）也是不錯的選擇。Colab 的使用方式可參考 :ref:`附錄 <colab>` 。

.. warning:: 如果你使用的是舊版本的 PyCharm ，可能會在安裝 TensorFlow 2 後出現部分程式碼自動補全功能遺失的問題。升級到新版的 PyCharm （2019.3 及以後版本）即可解決這一問題。


.. [#nohup] 關於  ``nohup`` 指令可參考 http://linux.vbird.org/linux_basic/0440processcontrol.php#background_term

TensorFlow 所需的硬體配置 *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint:: 對於學習而言，TensorFlow 的硬體門檻並不高。甚至，借助 :ref:`免費 <colab>` 或 :ref:`靈活 <gcp>` 的雲端計算資源，只要你有一台能上網的電腦，就能夠熟練掌握 TensorFlow！

在很多人的刻板印象中，TensorFlow 乃至深度學習是一件非常 “吃硬體資源” 的事情，以至於一接觸 TensorFlow，第一件事情可能就是想如何升級自己的電腦硬體。不過，TensorFlow 所需的硬體配置很大程度是依照任務和使用環境而定的：

- 對於 TensorFlow 初學者，無需硬體升級也可以很好地的學習和掌握 TensorFlow。本手冊中的大部分教學範例，大部分當前主流的個人電腦（即使沒有 GPU）均可勝任，無需添置其他硬體設備。對於本手冊中部分計算量較大的範例（例如 :ref:`在cats_vs_dogs資料集上訓練 CNN 圖片分類 <cats_vs_dogs>` ），一張主流的 NVIDIA GPU 會大幅加速訓練。如果自己的個人電腦難以勝任，可以考慮在雲端（例如 ref:`免費的 Colab <colab>` ）進行模型訓練。
- 對於參加資料科學競賽（比如 Kaggle）或者經常在本機進行訓練的個人愛好者或開發者，一塊高性能的 NVIDIA GPU 往往是必要的。CUDA 核心數和顯示卡內存大小是決定顯卡機器學習性能的兩個關鍵參數，前者決定訓練速度，後者決定可以訓練多大的模型以及訓練時的最大 Batch Size，對於較大規模的訓練而言尤其更為明顯。
- 對於前瞻的機器學習研究（尤其是電腦視覺和自然語言處理領域），多 GPU 平行訓練是標準配置。為了快速疊代實驗結果以及訓練更大規模的模型以提升性能，4 塊顯示卡、8 塊顯示卡或更高的 GPU 數量是常態。

作為參考，筆者給出截至本手冊撰寫時，自己所在工作環境的一些硬體配置：

- 筆者寫作本書的範例程式碼時，除了分佈式和雲端訓練相關章節，其他部分均使用一台 Intel i5 處理器，16GB DDR3 記憶體內存的普通電腦（未使用 GPU）
- 在筆者的研究工作中，長年使用一塊 NVIDIA GTX 1060 （單卡 6GB 記憶體）在本地環境進行模型的基礎開發和測試；
- 筆者所在的實驗室使用一台 4 塊 NVIDIA GTX 1080 Ti （單卡 11GB 記憶體）平行的工作站和一台 10 塊 NVIDIA GTX 1080 Ti （單卡 11GB 記憶體）平行的伺服器進行電腦視覺模型的訓練；
- 筆者合作過的公司使用 8 塊 NVIDIA Tesla V100 （單卡 32GB 記憶體）平行的伺服器進行自然語言處理（如大規模機器翻譯）模型的訓練。

儘管科技研究單位或公司使用的計算硬體配置堪稱豪華，不過與其他前瞻科研領域（例如生物）動輒幾十上百萬的儀器試劑費用相比，依然不算太貴的深度學習伺服器就可以供數位研究者使用很長時間。因此，機器學習相對而言還是十分平易近人的。

關於深度學習工作站的具體配置，由於硬體行情更新較快，故不在此列出具體配置，推薦參考 `深度學習電腦硬體配備怎麼選？ <https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/>`_ ，並結合最新市場情況進行配置。

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 188 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>