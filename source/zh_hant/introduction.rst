TensorFlow概述
======================

當我們在說 “我想要學習一個深度學習框架”，或者 “我想學習 TensorFlow”、“我想學習 TensorFlow 2.0” 的時候，我們究竟想要學到什麼？事實上，對於不同群體，可能會有相當不同的預期。

學生和研究者：模型的建立與訓練
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果你是一個初學機器學習 / 深度學習的學生，你可能已經啃完了 Andrew Ng 的機器學習公開課程或者史丹佛大學的 `UFIDL Tutorial <http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial>`_ ，亦或是正在上學校裡的深度學習課程。你可能也已經了解鏈鎖法則(Chain Rule)和梯度下降法(Gradient descent)，也知道許多種損失函數(Loss function)，並且對卷積神經網路（CNN）、循環神經網路（RNN）和強化學習(Reinforcement Learning, RL)的理論也有了一些大致的認識。然而 —— 你依然不知道這些模型在電腦中具體要如何實現。這時，你希望能有一個函式庫，幫助你把書本上的公式和算法運用於實踐。

具體而言，以最常見的監督式學習（supervised learning）為例。假設你已經掌握了一個模型 :math:`\hat{y} = f(x, \theta)` （x、y 為輸入和輸出， :math:`\theta` 為模型參數），確定了一個損失函數 :math:`L(y, \hat{y})` ，並獲得了一批資料 :math:`X`  和相對應的標籤 :math:`Y` 。這時，你會希望有一個函式庫，幫助你實現下列事情：

- 用電腦程式表示出向量、矩陣和張量等數學概念，並方便的進行運算；
- 方便的建立模型 :math:`f(x, \theta)` 和損失函數 :math:`L(y, \hat{y}) = L(y, f(x, \theta))` 。給定輸入 :math:`x_0 \in X` ，對應的標籤 :math:`y_0 \in Y` 和當前疊代的參數值 :math:`\theta_0` ，能夠方便的計算出模型預測值 :math:`\hat{y_0} = f(x_0, \theta_0)` ，並計算損失函數的值 :math:`L_0 = L(y_0, \hat{y_0}) = L(y_0, f(x_0, \theta_0))` ；
- 自動將損失函數 :math:`L` 求已知 :math:`x_0`、:math:`y_0`、:math:`\theta_0` 時對模型參數 :math:`\theta` 的偏導數值，即計算 :math:`\theta_0' = \frac{\partial L}{\partial \theta} |_{x = x_0, y = y_0, \theta = \theta_0}` ，無需人工推導結果（這意味著，這個函式庫需要支援某種意義上的 “符號計算”，表現在能夠記錄下運算的全過程，這樣才能根據鏈鎖法則進行反向推導）；
- 根據所求出的偏導數 :math:`\theta_0'` 的值，方便的呼叫一些優化方法更新當前疊代的模型參數 :math:`\theta_0` ，得到下一疊代的模型參數 :math:`\theta_1` （比如梯度下降法， :math:`\theta_1 = \theta_0 - \alpha \theta_0'` ， :math:`\alpha` 為學習率）。

更抽象一些地說，這個你所希望的函式庫需要能做到：

- 數學概念和運算的程式化表達；
- 對任意可導函數 :math:`f(x)` ，求在自變數 :math:`x = x_0` 給定時的梯度 :math:`\nabla f | _{x = x_0}` （「符號計算」的能力）。

開發者和工程師：模型的呼叫與部署
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果你是一位在 IT 行業打滾多年的開發者或者工程師，你可能已經對大學期間學到的數學知識不再熟悉（“多元函數…… 求偏微分？那是什麼東西？”）。然而，你可能希望在你的產品中加入一些與人工智慧相關的功能，抑或者需要將已有的深度學習模型部署到各種場景中。具體而言，包括：

* 如何匯出訓練好的模型？
* 如何在本機使用已有的預訓練模型？
* 如何在伺服器、可攜式裝置、嵌入式設備甚至網頁上高效運行模型？
* ……

TensorFlow 能幫助我們做什麼？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow 可以為以上的這些需求提供完整的解決方案。具體而言，TensorFlow 包含以下特性：

- 訓練流程
    - **資料的處理** ：使用 tf.data 和 TFRecord 可以高效的建構預處理資料集與訓練資料集。同時可以使用 TensorFlow Datasets 快速載入常用的公開資料集。
    - **模型的建立與測試** ：使用即時執行模式和著名的神經網路高層 API 框架 Keras，結合可視化工具 TensorBoard，簡易、快速地建立和測試模型。也可以通過 TensorFlow Hub 方便的載入已有的成熟模型。
    - **模型的訓練** ：支援在 CPU、GPU、TPU 上訓練模型，支援單機和多台電腦平行訓練模型，充分利用大量資料和計算資源進行高效訓練。
    - **模型的匯出** ：將模型打包匯出為統一的 SavedModel 格式，方便遷移和部署。
- 部署流程
    - **伺服器部署** ：使用 TensorFlow Serving 在伺服器上為訓練完成的模型提供高性能、且可平行運算的高流量 API。
    - **可攜式裝置和嵌入式設備部署** ：使用 TensorFlow Lite 將模型轉換為體積小、高效率的輕量化版本，並在可攜式裝置、嵌入式端等功耗和計算能力受限的設備上運行，支援使用 GPU 代理進行硬體加速，還可以配合 Edge TPU 等外接硬體加速運算。
    - **網頁端部署** ：使用 TensorFlow.js，在網頁端等支援 JavaScript 運行的環境上也可以運行模型，支援使用 WebGL 進行硬體加速。


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

