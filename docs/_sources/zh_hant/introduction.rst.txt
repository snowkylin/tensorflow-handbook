TensorFlow概述
======================

當我們在說「我想要學習一個深度學習框架」，或者「我想學習TensorFlow」、「我想學習TensorFlow 2.0」的時候，我們究竟想要學到什麼？事實上，對於不同羣體，可能會有相當不同的預期。

學生和研究者：模型的建立與訓練
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果你是一個初學機器學習/深度學習的學生，你可能已經啃完了Andrew Ng的機器學習公開課或者斯坦福的 `UFIDL Tutorial <http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial>`_ ，亦或是正在上學校里的深度學習課程。你可能也已經了解了鏈式求導法則和梯度下降法，知道了若干種損失函數，並且對卷積神經網絡（CNN）、循環神經網絡（RNN）和強化學習的理論也有了一些大致的認識。然而——你依然不知道這些模型在計算機中具體要如何實現。這時，你希望能有一個程序庫，幫助你把書本上的公式和算法運用於實踐。

具體而言，以最常見的有監督學習（supervised learning）爲例。假設你已經掌握了一個模型 :math:`\hat{y} = f(x, \theta)` （x、y爲輸入和輸出， :math:`\theta` 爲模型參數），確定了一個損失函數 :math:`L(y, \hat{y})` ，並獲得了一批數據 :math:`X`  和相對應的標籤 :math:`Y` 。這時，你會希望有一個程序庫，幫助你實現下列事情：

- 用電腦程式表示出向量、矩陣和張量等數學概念，並方便地進行運算；
- 方便地建立模型 :math:`f(x, \theta)` 和損失函數 :math:`L(y, \hat{y}) = L(y, f(x, \theta))` 。給定輸入 :math:`x_0 \in X` ，對應的標籤 :math:`y_0 \in Y` 和當前疊代輪的參數值 :math:`\theta_0` ，能夠方便地計算出模型預測值 :math:`\hat{y_0} = f(x_0, \theta_0)` ，並計算損失函數的值 :math:`L_0 = L(y_0, \hat{y_0}) = L(y_0, f(x_0, \theta_0))` ；
- 自動將損失函數 :math:`L` 求已知 :math:`x_0`、:math:`y_0`、:math:`\theta_0` 時對模型參數 :math:`\theta` 的偏導數值，即計算 :math:`\theta_0' = \frac{\partial L}{\partial \theta} |_{x = x_0, y = y_0, \theta = \theta_0}` ，無需人工推導求導結果（這意味著，這個程序庫需要支持某種意義上的「符號計算」，表現在能夠記錄下運算的全過程，這樣才能根據鏈式法則進行反向求導）；
- 根據所求出的偏導數 :math:`\theta_0'` 的值，方便地調用一些優化方法更新當前疊代輪的模型參數 :math:`\theta_0` ，得到下一疊代輪的模型參數 :math:`\theta_1` （比如梯度下降法， :math:`\theta_1 = \theta_0 - \alpha \theta_0'` ， :math:`\alpha` 爲學習率）。

更抽象一些地說，這個你所希望的程序庫需要能做到：

- 數學概念和運算的程序化表達；
- 對任意可導函數 :math:`f(x)` ，求在自變量 :math:`x = x_0` 給定時的梯度 :math:`\nabla f | _{x = x_0}` （「符號計算」的能力）。

開發者和工程師：模型的調用與部署
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果你是一位在IT行業沉澱多年的開發者或者工程師，你可能已經對大學期間學到的數學知識不再熟悉（「多元函數……求偏微分？那是什麼東西？」）。然而，你可能希望在你的產品中加入一些與人工智慧相關的功能，抑或需要將已有的深度學習模型部署到各種場景中。具體而言，包括：

* 如何導出訓練好的模型？
* 如何在本機使用已有的預訓練模型？
* 如何在伺服器、移動端、嵌入式設備甚至網頁上高效運行模型？
* ……

TensorFlow能幫助我們做什麼？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow可以爲以上的這些需求提供完整的解決方案。具體而言，TensorFlow包含以下特性：

- 訓練流程
    - **數據的處理** ：使用 tf.data 和 TFRecord 可以高效地構建和預處理數據集，構建訓練數據流。同時可以使用 TensorFlow Datasets 快速載入常用的公開數據集。
    - **模型的建立與調試** ：使用即時執行模式和著名的神經網絡高層 API 框架 Keras，結合可視化工具 TensorBoard，簡易、快速地建立和調試模型。也可以通過 TensorFlow Hub 方便地載入已有的成熟模型。
    - **模型的訓練** ：支持在 CPU、GPU、TPU 上訓練模型，支持單機和多機集羣並行訓練模型，充分利用海量數據和計算資源進行高效訓練。 
    - **模型的導出** ：將模型打包導出爲統一的 SavedModel 格式，方便遷移和部署。
- 部署流程
    - **伺服器部署** ：使用 TensorFlow Serving 在伺服器上爲訓練完成的模型提供高性能、支持並發、高吞吐量的API。
    - **移動端和嵌入式設備部署** ：使用TensorFlow Lite 將模型轉換爲體積小、高效率的輕量化版本，並在移動端、嵌入式端等功耗和計算能力受限的設備上運行，支持使用 GPU 代理進行硬體加速，還可以配合 Edge TPU 等外接硬體加速運算。
    - **網頁端部署** ：使用 TensorFlow.js，在網頁端等支持 JavaScript 運行的環境上也可以運行模型，支持使用 WebGL 進行硬體加速。


.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 187 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>

