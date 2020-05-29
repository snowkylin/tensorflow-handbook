前言
======

2018年3月30日，Google在加州山景城舉行了第二屆TensorFlow Dev Summit開發者峯會，並宣布正式發布TensorFlow 1.8版本。作爲大中華地區首批機器學習領域的谷歌開發者專家（ `Google Developers Expert <https://developers.google.cn/community/experts>`_ , GDE），我有幸獲得Google的資助親臨峯會現場，見證了這一具有里程碑式意義的新版本發布。衆多新功能的加入和支持展示了TensorFlow的雄心壯志，已經醞釀許久的即時執行模式（Eager Execution，或稱「動態圖模式」）在這一版本中也終於正式得到支持。

在此之前，TensorFlow 所基於的傳統的圖執行模式與會話機制（Graph Execution and Session，或稱「靜態圖模式」）的弊端，如入門門檻高、調試困難、靈活性差、無法使用 Python 原生控制語句等，已被開發者詬病許久。一些新的基於即時執行模式的深度學習框架（如 PyTorch）也橫空出世，並以其易用性和快速開發的特性而占據了一席之地。尤其是在學術研究等需要快速疊代模型的領域，PyTorch 等新興深度學習框架已成爲主流。我所在的數十人的機器學習實驗室中，竟只有我一人「守舊」地使用 TensorFlow。而市面上 TensorFlow 相關的中文技術書以及資料也仍然基於傳統的計算圖與會話機制，這讓不少初學者，尤其是剛學過機器學習課程的大學生望而卻步。

因此，在 TensorFlow 正式支持即時運行模式之際，我認爲有必要出現一本全新的入門手冊，幫助初學者及需要快速疊代模型的研究者，以「即時執行」的視角快速入門 TensorFlow。這也是我編寫本手冊的初衷。本手冊自2018年春開始編寫，並在2018年8月在 GitHub 發布了第一個中英雙語版本，很快得到了國內外不少開發者的關注。尤其是 TensorFlow 工程總監 Rajat Monga 、 Google AI 負責人 Jeff Dean 和 TensorFlow 官方在社交媒體上對本手冊給予了推薦與關注，這給了我很大的鼓舞。同時，我作爲谷歌開發者專家，多次受到谷歌開發者社羣（ `Google Developers Group <https://developers.google.cn/community/gdg>`_ , GDG）的邀請，在 GDG DevFest、TensorFlow Day 和 Women Techmakers 等活動中使用本手冊進行線下的 TensorFlow Codelab 教學，獲得了較好的反響，也得到了不少反饋和建議。這些都促進了本手冊的更新和質量改進。

在2019年3月的第三屆TensorFlow Dev Summit開發者峯會上，我再次受邀來到谷歌的矽谷總部，見證了 TensorFlow 2.0 alpha 的發布。此時的 TensorFlow 已經形成了一個擁有龐大版圖的生態系統。TensorFlow Lite、TensorFlow.js、TensorFlow for Swift、TPU 等各種組件日益成熟，同時 TensorFlow 2 加入了提升易用性的諸多新特性（例如以 ``tf.keras`` 爲核心的統一高層API、使用 ``tf.function`` 構建圖模型、默認使用即時執行模式等）。這使得本手冊的大幅擴充更新提上日程。GDE 社羣的兩位 JavaScript 和 Android 領域的資深專家李卓桓和朱金鵬加入了本手冊的編寫，使得本手冊增加了諸多面向業界的 TensorFlow 模塊詳解與實例。同時，我在谷歌開發者大使（Developer Advocate） Paige Bailey 的邀請下申請並成功加入了 Google Summer of Code 2019 活動。作爲全世界 20 位由 Google TensorFlow 項目資助的學生開發者之一，我在 2019 年的暑期基於 TensorFlow 2.0 Beta 版本，對本手冊進行了大幅擴充和可讀性上的改進。這使得本手冊從 2018 年發布的小型入門指南逐漸成長爲一本內容全面的 TensorFlow 技術手冊和開發指導。

2019年10月1日，TensorFlow 2.0 正式版發布，同時本手冊也開始了在 TensorFlow 官方公衆微信號（TensorFlow_official）上的長篇連載。在連載過程中，我收到了大量的讀者提問和意見反饋。爲讀者答疑的同時，我也修訂了手冊中的較多細節。受到新冠疫情的影響，2020年3月的第四屆 TensorFlow Dev Summit 開發者峯會在線上直播舉行。我根據峯會的內容爲手冊增添了部分內容，特別是介紹了 TensorFlow Quantum 這一混合量子-經典機器學習庫的基本使用方式。我在研究生期間旁聽過量子力學，還做過量子計算和機器學習結合的專題報告。TensorFlow Quantum 的推出著實讓我感到興奮，並迫不及待地希望介紹給廣大讀者。2020年4月，爲了適應新冠疫情期間的線上教學需求，我接受 TensorFlow Usergroup （TFUG）和谷歌開發者社羣的邀請，依託本手冊在 TensorFlow 官方公衆微信號上開展了「機器學習Study Jam」線上教學活動，並啓用了手冊留言版 https://discuss.tf.wiki 進行教學互動答疑。此次教學也同樣有不少學習者爲本手冊提供了重要的改進意見。

由於我的研究方向是強化學習，所以在本手冊的附錄中加入了「強化學習簡介」一章，對強化學習進行了更細緻的介紹。和絕大多數強化學習教程一開始就介紹馬爾可夫決策過程和各種概念不同，我從純動態規劃出發，結合具體算例來介紹強化學習，試圖讓強化學習和動態規劃的關係更清晰，以及對程式設計師更友好。這個視角相對比較特立獨行，如果您發現了謬誤之處，也請多加指正。

本手冊的主要特徵有：

* 主要基於 TensorFlow 2 最新的即時執行模式，以便於模型的快速疊代開發。同時使用 ``tf.function`` 實現圖執行模式；
* 定位以技術手冊爲主，編排以 TensorFlow 2 的各項概念和功能爲核心，力求能夠讓 TensorFlow 開發者快速查閱。各章相對獨立，不一定需要按順序閱讀；
* 代碼實現均進行仔細推敲，力圖簡潔高效和表意清晰。模型實現均統一使用繼承 ``tf.keras.Model`` 和 ``tf.keras.layers.Layer`` 的方式，保證代碼的高度可復用性。每個完整項目的代碼總行數均不超過一百行，讓讀者可以快速理解並舉一反三；
* 注重詳略，少即是多，不追求巨細無遺和面面俱到，不在正文中進行大篇幅的細節論述。

本書的適用羣體
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本書適用於以下讀者：

* 已有一定機器學習或深度學習基礎，希望將所學理論知識使用 TensorFlow 進行具體實現的學生和研究者；
* 曾使用或正在使用 TensorFlow 1.X 版本或其他深度學習框架（比如PyTorch），希望了解和學習 TensorFlow 2 新特性的開發者；
* 希望將已有的 TensorFlow 模型應用於業界的開發者或工程師。

.. hint:: 本書不是一本機器學習或深度學習原理入門手冊。若希望進行機器學習或深度學習理論的入門學習，可參考 :doc:`附錄「參考資料與推薦閱讀」 <appendix/recommended_books>` 中提供的一些入門資料。

如何使用本書
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

對於已有一定機器學習或深度學習基礎，著重於使用 TensorFlow 2 進行模型建立與訓練的學生和研究者，建議順序閱讀本書的「基礎」部分。爲了幫助部分新入門機器學習的讀者理解內容，本手冊在「基礎」部分的章節中使用獨立的信息框，提供了一些與行文內容相關的機器學習基礎知識。這些內容旨在幫助讀者將機器學習理論知識與具體的 TensorFlow 程序代碼進行結合，以深入了解 TensorFlow 代碼的內在機制，讓讀者在調用 TensorFlow 的 API 時能夠知其所以然。然而，這些內容對於沒有機器學習基礎的讀者而言很可能仍是完全不足夠的。若讀者發現閱讀這些內容時有很強的陌生感，則應該先行學習一些機器學習相關的基礎概念。部分章節的開頭提供了「前置知識」部分，方便讀者查漏補缺。

對於希望將 TensorFlow 模型部署到實際環境中的開發者和工程師，可以重點閱讀本書的「部署」部分，尤其是需要結合代碼示例進行親手操作。不過，依然非常建議學習一些機器學習的基本知識並閱讀本手冊的「基礎」部分，以更加深入地了解 TensorFlow 2。

對於已有 TensorFlow 1.X 使用經驗的開發者，可以從本手冊的「高級」部分開始閱讀，尤其是「圖執行模式下的 TensorFlow 2」和「 ``tf.GradientTape`` 詳解」兩章。隨後，可以快速瀏覽一遍「基礎」部分以了解即時執行模式下 TensorFlow 的使用方式。

在整本手冊中，帶「*」的部分均爲選讀。

本書的所有示例代碼可至 https://github.com/snowkylin/tensorflow-handbook/tree/master/source/_static/code 獲得。其中 ``zh`` 目錄下是含中文注釋的代碼（對應於本書的中文版本，即您手上的這一本）， ``en`` 目錄下是含英文版注釋的代碼（對應於本書的英文版本）。在使用時，建議將代碼根目錄加入到 ``PYTHONPATH`` 環境變量，或者使用合適的IDE（如PyCharm）打開代碼根目錄，從而使得代碼間的相互調用（形如 ``import zh.XXX`` 的代碼）能夠順利運行。

致謝
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

首先感謝我的好友兼同學 Chris Wu 編寫的《簡單粗暴 :math:`\text{\LaTeX}` 》（https://github.com/wklchris/Note-by-LaTeX，紙質版名爲《簡單高效LaTeX》）爲本手冊的初始體例編排提供了模範和指引。該手冊清晰精煉，是 :math:`\text{\LaTeX}` 領域不可多得的中文資料，本手冊的初始名稱《簡單粗暴 TensorFlow》也是在向這本手冊致敬。本手冊最初是在我的好友 Ji-An Li 所組織的深度學習研討小組中，作爲預備知識講義編寫和使用的。好友們卓著的才學與無私分享的精神是編寫此拙作的重要助力。

本手冊的TensorFlow.js和TensorFlow Lite章節分別由李卓桓和朱金鵬兩位在JavaScript和Android領域有豐富履歷的GDE和前GDE撰寫，同時，卓桓撰寫了TensorFlow for Swift和TPU部分的介紹，金鵬提供了TensorFlow Hub的介紹。來自豆瓣閱讀的王子陽也提供了關於Node.js和阿里雲的部分示例代碼和說明。在此特別表示感謝。

在基於本手冊初稿的多場線下、線上教學活動和TensorFlow官方微信公衆號連載中，大量活動參與者與讀者爲本手冊提供了有價值的反饋，促進了本手冊的持續更新。谷歌開發者社羣和 TensorFlow Usergroup 的多位志願者們也爲這些活動的順利舉辦做出了重要貢獻。來自中國科學技術大學的 Zida Jin 將本手冊2018年初版的大部分內容翻譯爲了英文，Ming 和 Ji-An Li 在英文版翻譯中亦有貢獻，促進了本手冊在世界範圍內的推廣。Eric ShangKuan、Jerry Wu 、Hsiang Huang、Po-Yi Li、Charlie Li 、Chunju Hsu 協助了本手冊的簡轉繁轉譯。在此一併表示由衷的謝意。

衷心感謝 Google 開發者關係團隊和 TensorFlow 工程團隊的成員及前成員們對本手冊的編寫所提供的幫助。其中，開發者關係團隊的 Luke Cheng 在本手冊初版編寫過程中提供重要的思路啓發和鼓勵，且提供本手冊在線版本的域名 `tf.wiki <https://tf.wiki>`_ 和留言版 https://discuss.tf.wiki ；開發者關係團隊的 Soonson Kwon、Lily Chen、Wei Duan、Tracy Wang、Rui Li、Pryce Mu，TensorFlow 產品經理 Mike Liang 和谷歌開發者大使 Paige Bailey 爲本手冊宣傳及推廣提供了大力支持；開發者關係團隊的 Eric ShangKuan 協助了本手冊的繁體版轉譯；TensorFlow 工程團隊的 Tiezhen Wang 在本手冊的工程細節方面提供了諸多建議和補充；TensorFlow 中國研發負責人 Shuangfeng Li 和 TensorFlow 工程團隊的其他工程師們爲本手冊提供了專業的審閱意見。同時感謝 TensorFlow 工程總監 Rajat Monga 和 Google AI 負責人 Jeff Dean 在社交媒體上對本手冊的推薦與關注。感謝 Google Summer of Code 2019 對本開源項目的資助。

本手冊主體部分為我在北京大學信息科學技術學院智能科學系攻讀碩士學位時所撰寫。感謝我的導師童雲海教授和實驗室的同學們對本手冊的支持和建議。

最後，感謝人民郵電出版社的王軍花、武芮欣兩位編輯對本手冊紙質版的細緻編校及出版流程跟進。感謝我的父母和好友對本手冊的關注和支持。

關於本手冊的意見和建議，歡迎在 https://discuss.tf.wiki 提交。您的寶貴意見將促進本手冊的持續更新。

|

Google Developers Expert in Machine Learning

Xihan Li ( `snowkylin <https://snowkylin.github.io/>`_ )

2020 年 5 月於深圳
