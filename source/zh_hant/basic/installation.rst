TensorFlow安裝與環境配置
======================================

TensorFlow的最新安裝步驟可參考官方網站上的說明（https://tensorflow.google.cn/install）。TensorFlow支持Python、Java、Go、C等多種程式語言以及Windows、OSX、Linux等多種作業系統，此處及後文均以Python 3.7為準。

.. hint:: 本章介紹在一般的個人電腦或伺服器上直接安裝TensorFlow 2的方法。關於在容器環境（Docker）、雲平台中部署TensorFlow或在線上環境中使用TensorFlow的方法，見附錄 :doc:`使用Docker部署TensorFlow環境 <../appendix/docker>` 和 :doc:`在雲端使用TensorFlow <../appendix/cloud>` 。軟體的安裝方法往往具有時效性，本節的更新日期爲2019年10月。

一般安裝步驟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. 安裝Python環境。此處建議安裝 `Anaconda <https://www.anaconda.com/>`_ 的Python 3.7版本（後文均以此為準），這是一個開源的Python發行版本，提供了一個完整的科學計算環境，包括NumPy、SciPy等常用科學計算庫。當然，你有權選擇自己喜歡的Python環境。Anaconda的安裝包可在 `這裡 <https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/>`_ 獲得。

2. 使用Anaconda自帶的conda包管理器建立一個Conda虛擬環境，並進入該虛擬環境。在命令行下輸入：

::

    conda create --name tf2 python=3.7      # 「tf2」是你建立的conda虛擬環境的名字
    conda activate tf2                      # 進入名爲「tf2」的conda虛擬環境

3. 使用Python包管理器pip安裝TensorFlow。在命令行下輸入：

::

    pip install tensorflow

等待片刻即安裝完畢。

.. tip:: 

    1. 也可以使用 ``conda install tensorflow`` 來安裝TensorFlow，不過conda源的版本往往更新較慢，難以第一時間獲得最新的TensorFlow版本；
    2. 從 TensorFlow 2.1 開始，pip 包 ``tensorflow`` 即同時包含 GPU 支持，無需通過特定的 pip 包 ``tensorflow-gpu`` 安裝GPU版本。如果對pip包的大小敏感，可使用 ``tensorflow-cpu`` 包安裝僅支持CPU的TensorFlow版本。
    3. 在Windows下，需要打開開始菜單中的「Anaconda Prompt」進入Anaconda的命令行環境；
    4. 如果默認的pip和conda網絡連接速度慢，可以嘗試使用鏡像，將顯著提升pip和conda的下載速度（具體效果視您所在的網絡環境而定）；
        
        - 北京清華大學的pypi鏡像：https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
        - 北京清華大學的Anaconda鏡像：https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/
    5. 如果對磁碟空間要求嚴格（比如伺服器環境），可以安裝 `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ ，僅包含Python和Conda，其他的包可自己按需安裝。Miniconda的安裝包可在 `這裡 <https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/>`_ 獲得。

.. admonition:: pip和conda包管理器

    pip是最爲廣泛使用的Python包管理器，可以幫助我們獲得最新的Python包並進行管理。常用命令如下：

    ::

        pip install [package-name]              # 安裝名爲[package-name]的包
        pip install [package-name]==X.X         # 安裝名爲[package-name]的包並指定版本X.X
        pip install [package-name] --proxy=代理伺服器IP:埠號         # 使用代理伺服器安裝
        pip install [package-name] --upgrade    # 更新名爲[package-name]的包
        pip uninstall [package-name]            # 刪除名爲[package-name]的包
        pip list                                # 列出當前環境下已安裝的所有包
    
    conda包管理器是Anaconda自帶的包管理器，可以幫助我們在conda環境下輕鬆地安裝各種包。相較於pip而言，conda的通用性更強（不僅是Python包，其他包如CUDA Toolkit和cuDNN也可以安裝），但conda源的版本更新往往較慢。常用命令如下：

    ::

        conda install [package-name]        # 安裝名爲[package-name]的包
        conda install [package-name]=X.X    # 安裝名爲[package-name]的包並指定版本X.X
        conda update [package-name]         # 更新名爲[package-name]的包
        conda remove [package-name]         # 刪除名爲[package-name]的包
        conda list                          # 列出當前環境下已安裝的所有包
        conda search [package-name]         # 列出名爲[package-name]的包在conda源中的所有可用版本

    conda中配置代理：在用戶目錄下的 .condarc 文件中添加以下內容：

    ::

        proxy_servers:
            http: http://代理伺服器IP:埠號

.. admonition:: Conda虛擬環境

    在Python開發中，很多時候我們希望每個應用有一個獨立的Python環境（比如應用1需要用到TensorFlow 1.X，而應用2使用TensorFlow 2.0）。這時，Conda虛擬環境即可爲一個應用創建一套「隔離」的Python運行環境。使用Python的包管理器conda即可輕鬆地創建Conda虛擬環境。常用命令如下：

    ::

        conda create --name [env-name]      # 建立名爲[env-name]的Conda虛擬環境
        conda activate [env-name]           # 進入名爲[env-name]的Conda虛擬環境
        conda deactivate                    # 退出當前的Conda虛擬環境
        conda env remove --name [env-name]  # 刪除名爲[env-name]的Conda虛擬環境
        conda env list                      # 列出所有Conda虛擬環境

.. _gpu_tensorflow:

GPU版本TensorFlow安裝指南
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GPU版本的TensorFlow可以利用NVIDIA GPU強大的計算加速能力，使TensorFlow的運行更爲高效，尤其是可以成倍提升模型訓練的速度。

在安裝GPU版本的TensorFlow前，你需要具有一塊不太舊的NVIDIA顯卡，以及正確安裝NVIDIA顯卡驅動程序、CUDA Toolkit和cuDNN。

GPU硬體的準備
-------------------------------------------

TensorFlow對NVIDIA顯卡的支持較爲完備。對於NVIDIA顯卡，要求其CUDA Compute Capability須不低於3.5，可以到 `NVIDIA的官方網站 <https://developer.nvidia.com/cuda-gpus/>`_ 查詢自己所用顯卡的CUDA Compute Capability。目前，AMD的顯卡也開始對TensorFlow提供支持，可訪問  `這篇博客文章 <https://medium.com/tensorflow/amd-rocm-gpu-support-for-tensorflow-33c78cc6a6cf>`_  查看詳情。

NVIDIA驅動程序的安裝
-------------------------------------------

**Windows** 

Windows環境中，如果系統具有NVIDIA顯卡，則往往已經自動安裝了NVIDIA顯卡驅動程序。如未安裝，直接訪問 `NVIDIA官方網站 <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_ 下載並安裝對應型號的最新公版驅動程序即可。

**Linux** 

在伺服器版Linux系統下，同樣訪問 `NVIDIA官方網站 <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_ 下載驅動程序（爲 ``.run`` 文件），並使用 ``sudo bash DRIVER_FILE_NAME.run`` 命令安裝驅動即可。在安裝之前，可能需要使用 ``sudo apt-get install build-essential`` 安裝合適的編譯環境。

在具有圖形界面的桌面版Linux系統上，NVIDIA顯卡驅動程序需要一些額外的配置，否則會出現無法登錄等各種錯誤。如果需要在Linux下手動安裝NVIDIA驅動，注意在安裝前進行以下步驟（以Ubuntu爲例）：

- 禁用系統自帶的開源顯卡驅動Nouveau（在 ``/etc/modprobe.d/blacklist.conf`` 文件中添加一行 ``blacklist nouveau`` ，使用 ``sudo update-initramfs -u`` 更新內核，並重啓）
- 禁用主板的Secure Boot功能
- 停用桌面環境（如 ``sudo service lightdm stop``）
- 刪除原有NVIDIA驅動程序（如 ``sudo apt-get purge nvidia*``）

.. tip:: 對於桌面版Ubuntu系統，有一個很簡易的NVIDIA驅動安裝方法：在系統設置（System Setting）裡面選軟體與更新（Software & Updates），然後點選Additional Drivers裡面的「Using NVIDIA binary driver」選項並點選右下角的「Apply Changes」即可，系統即會自動安裝NVIDIA驅動，但是通過這種安裝方式安裝的NVIDIA驅動往往版本較舊。

NVIDIA驅動程序安裝完成後，可在命令行下使用 ``nvidia-smi`` 命令檢查是否安裝成功，若成功則會列印出當前系統安裝的NVIDIA驅動信息，形式如下：

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

.. hint:: 命令 ``nvidia-smi`` 可以查看機器上現有的GPU及使用情況。（在Windows下，將 ``C:\Program Files\NVIDIA Corporation\NVSMI`` 加入Path環境變量中即可，或Windows 10下可使用任務管理器的「性能」標籤查看顯卡信息）

更詳細的GPU環境配置指導可以參考 `這篇文章 <https://www.linkedin.com/pulse/installing-nvidia-cuda-80-ubuntu-1604-linux-gpu-new-victor/>`_ 和 `這篇中文博客 <https://blog.csdn.net/wf19930209/article/details/81877822>`_ 。

CUDA Toolkit和cuDNN的安裝
-------------------------------------------

在Anaconda環境下，推薦使用 

::

    conda install cudatoolkit=X.X
    conda install cudnn=X.X.X

安裝CUDA Toolkit和cuDNN，其中X.X和X.X.X分別爲需要安裝的CUDA Toolkit和cuDNN版本號，必須嚴格按照 `TensorFlow官方網站所說明的版本 <https://www.tensorflow.org/install/gpu#software_requirements>`_ 安裝。例如，對於TensorFlow 2.1，可使用::

    conda install cudatoolkit=10.1
    conda install cudnn=7.6.5

在安裝前，可使用 ``conda search cudatoolkit`` 和 ``conda search cudnn`` 搜索conda源中可用的版本號。

當然，也可以按照 `TensorFlow官方網站上的說明 <https://www.tensorflow.org/install/gpu>`_ 手動下載CUDA Toolkit和cuDNN並安裝，不過過程會稍繁瑣。

使用conda包管理器安裝GPU版本的TensorFlow時，會自動安裝對應版本的CUDA Toolkit和cuDNN。conda源的更新往往較慢，如果對版本不太介意，也可以直接使用 ``conda install tensorflow-gpu`` 進行安裝。

第一個程序
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

安裝完畢後，我們來編寫一個簡單的程序來驗證安裝。

在命令行下輸入 ``conda activate tf2`` 進入之前建立的安裝有TensorFlow的Conda虛擬環境，再輸入 ``python`` 進入Python環境，逐行輸入以下代碼：

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

說明TensorFlow已安裝成功。運行途中可能會輸出一些TensorFlow的提示信息，屬於正常現象。

.. warning:: 如果你在Windows下安裝了TensorFlow 2.1正式版，可能會在導入TensorFlow時出現 `DLL載入錯誤 <https://github.com/tensorflow/tensorflow/issues/35749>`_ 。此時安裝 `Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019 <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_ 即可正常使用。

此處使用的是Python語言，關於Python語言的入門教程可以參考 `runoob網站的Python 3教程 <http://www.runoob.com/python3/python3-tutorial.html>`_ 或 `廖雪峯的Python教程 <https://www.liaoxuefeng.com>`_ ，本手冊之後將默認讀者擁有Python語言的基本知識。不用緊張，Python語言易於上手，而TensorFlow本身也不會用到Python語言的太多高級特性。

IDE設置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

對於機器學習的研究者和從業者，建議使用 `PyCharm <http://www.jetbrains.com/pycharm/>`_ 作爲Python開發的IDE。

在新建項目時，你需要選定項目的Python Interpreter，也就是用怎樣的Python環境來運行你的項目。在安裝部分，你所建立的每個Conda虛擬環境其實都有一個自己獨立的Python Interpreter，你只需要將它們添加進來即可。選擇「Add」，並在接下來的窗口選擇「Existing Environment」，在Interpreter處選擇 ``Anaconda安裝目錄/envs/所需要添加的Conda環境名字/python.exe`` （Linux下無 ``.exe`` 後綴）並點擊「OK」即可。如果選中了「Make available to all projects」，則在所有項目中都可以選擇該Python Interpreter。注意，在Windows下Anaconda的默認安裝目錄比較特殊，一般爲  ``C:\Users\用戶名\Anaconda3\`` 或 ``C:\Users\用戶名\AppData\Local\Continuum\anaconda3`` 。此處 ``AppData`` 是隱藏文件夾。

對於TensorFlow開發而言，PyCharm的Professonal版本非常有用的一個特性是 **遠程調試** （Remote Debugging）。當你編寫程序的終端機性能有限，但又有一台可遠程ssh訪問的高性能計算機（一般具有高性能GPU）時，遠程調試功能可以讓你在終端機編寫程序的同時，在遠程計算機上調試和運行程序（尤其是訓練模型）。你在終端機上對代碼和數據的修改可以自動同步到遠程機，在實際使用的過程中如同在遠程機上編寫程序一般，與串流遊戲有異曲同工之妙。不過遠程調試對網絡的穩定性要求高，如果需要長時間訓練模型，建議登錄遠程機終端直接訓練模型（Linux下可以結合 ``nohup`` 命令 [#nohup]_ ，讓進程在後端運行，不受終端退出的影響）。遠程調試功能的具體配置步驟見 `PyCharm文檔 <https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html>`_ 。

.. tip:: 如果你是學生並有.edu結尾的郵箱的話，可以在 `這裡 <http://www.jetbrains.com/student/>`_ 申請PyCharm的免費Professional版本授權。

對於TensorFlow及深度學習的業餘愛好者或者初學者， `Visual Studio Code <https://code.visualstudio.com/>`_ 或者一些在線的交互式Python環境（比如免費的 `Google Colab <https://colab.research.google.com/>`_ ）也是不錯的選擇。Colab的使用方式可參考 :ref:`附錄 <colab>` 。

.. warning:: 如果你使用的是舊版本的 PyCharm ，可能會在安裝 TensorFlow 2 後出現部分代碼自動補全功能喪失的問題。升級到新版的 PyCharm （2019.3及以後版本）即可解決這一問題。


.. [#nohup] 關於  ``nohup`` 命令可參考 https://www.ibm.com/developerworks/cn/linux/l-cn-nohup/

TensorFlow所需的硬體配置 *
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint:: 對於學習而言，TensorFlow的硬體門檻並不高。甚至，藉助 :ref:`免費 <colab>` 或 :ref:`靈活 <gcp>` 的雲端計算資源，只要你有一台能上網的電腦，就能夠熟練掌握TensorFlow！

在很多人的刻板印象中，TensorFlow乃至深度學習是一件非常「吃硬體」的事情，以至於一接觸TensorFlow，第一件事情可能就是想如何升級自己的電腦硬體。不過，TensorFlow所需的硬體配置很大程度是視任務和使用環境而定的：

- 對於TensorFlow初學者，無需硬體升級也可以很好地學習和掌握TensorFlow。本手冊中的大部分教學示例，大部分當前主流的個人電腦（即使沒有GPU）均可勝任，無需添置其他硬體設備。對於本手冊中部分計算量較大的示例（例如 :ref:`在cats_vs_dogs數據集上訓練CNN圖像分類 <cats_vs_dogs>` ），一塊主流的NVIDIA GPU會大幅加速訓練。如果自己的個人電腦難以勝任，可以考慮在雲端（例如 :ref:`免費的 Colab <colab>` ）進行模型訓練。
- 對於參加數據科學競賽（比如Kaggle）或者經常在本機進行訓練的個人愛好者或開發者，一塊高性能的NVIDIA GPU往往是必要的。CUDA核心數和顯存大小是決定顯卡機器學習性能的兩個關鍵參數，前者決定訓練速度，後者決定可以訓練多大的模型以及訓練時的最大Batch Size，對於較大規模的訓練而言尤其敏感。
- 對於前沿的機器學習研究（尤其是計算機視覺和自然語言處理領域），多GPU並行訓練是標準配置。爲了快速疊代實驗結果以及訓練更大規模的模型以提升性能，4卡、8卡或更高的GPU數量是常態。

作爲參考，筆者給出截至本手冊撰寫時，自己所在工作環境的一些硬體配置：

- 筆者寫作本書的示例代碼時，除了分布式和雲端訓練相關章節，其他部分均使用一台Intel i5處理器，16GB DDR3內存的普通台式機（未使用GPU）在本地開發測試，部分計算量較大的模型使用了一塊淘寶上180元購買的 NVIDIA P106-90 （單卡3GB顯存）礦卡進行訓練；
- 在筆者的研究工作中，長年使用一塊 NVIDIA GTX 1060 （單卡6GB顯存）在本地環境進行模型的基礎開發和調試；
- 筆者所在的實驗室使用一台4塊 NVIDIA GTX 1080 Ti （單卡11GB顯存）並行的工作站和一台10塊 NVIDIA GTX 1080 Ti （單卡11GB顯存）並行的伺服器進行前沿計算機視覺模型的訓練；
- 筆者合作過的公司使用8塊 NVIDIA Tesla V100 （單卡32GB顯存）並行的伺服器進行前沿自然語言處理（如大規模機器翻譯）模型的訓練。

儘管科研機構或公司使用的計算硬體配置堪稱豪華，不過與其他前沿科研領域（例如生物）動輒幾十上百萬的儀器試劑費用相比，依然不算太貴（畢竟一台六七萬至二三十萬的深度學習伺服器就可以供數位研究者使用很長時間）。因此，機器學習相對而言還是十分平易近人的。

關於深度學習工作站的具體配置，由於硬體行情更新較快，故不在此列出具體配置，推薦關注 `知乎問題 - 如何配置一台適用於深度學習的工作站？ <https://www.zhihu.com/question/33996159>`_ ，並結合最新市場情況進行配置。

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