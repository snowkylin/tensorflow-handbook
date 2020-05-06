TensorFlow基礎
======================

本章介紹TensorFlow的基本操作。

.. admonition:: 前置知識

    * `Python基本操作 <http://www.runoob.com/python3/python3-tutorial.html>`_ （賦值、分支及循環語句、使用import導入庫）；
    * `Python的With語句 <https://www.ibm.com/developerworks/cn/opensource/os-cn-pythonwith/index.html>`_ ；
    * `NumPy <https://docs.scipy.org/doc/numpy/user/quickstart.html>`_ ，Python下常用的科學計算庫。TensorFlow與之結合緊密；
    * `向量 <https://zh.wikipedia.org/wiki/%E5%90%91%E9%87%8F>`_ 和 `矩陣 <https://zh.wikipedia.org/wiki/%E7%9F%A9%E9%98%B5>`_ 運算（矩陣的加減法、矩陣與向量相乘、矩陣與矩陣相乘、矩陣的轉置等。測試題：:math:`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = ?`）；
    * `函數的導數 <http://old.pep.com.cn/gzsx/jszx_1/czsxtbjxzy/qrzptgjzxjc/dzkb/dscl/>`_ ，`多元函數求導 <https://zh.wikipedia.org/wiki/%E5%81%8F%E5%AF%BC%E6%95%B0>`_ （測試題：:math:`f(x, y) = x^2 + xy + y^2, \frac{\partial f}{\partial x} = ?, \frac{\partial f}{\partial y} = ?`）；
    * `線性回歸 <http://old.pep.com.cn/gzsx/jszx_1/czsxtbjxzy/qrzptgjzxjc/dzkb/dscl/>`_ ；
    * `梯度下降方法 <https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95>`_ 求函數的局部最小值。

TensorFlow 1+1
^^^^^^^^^^^^^^^^^^^^^^^^^^^

我們可以先簡單地將TensorFlow視爲一個科學計算庫（類似於Python下的NumPy）。

首先，我們導入TensorFlow：

.. code-block:: python

    import tensorflow as tf

.. warning:: 本手冊基於TensorFlow的即時執行模式（Eager Execution）。在TensorFlow 1.X版本中， **必須** 在導入TensorFlow庫後調用 ``tf.enable_eager_execution()`` 函數以啓用即時執行模式。在 TensorFlow 2 中，即時執行模式將成爲默認模式，無需額外調用 ``tf.enable_eager_execution()`` 函數（不過若要關閉即時執行模式，則需調用 ``tf.compat.v1.disable_eager_execution()`` 函數）。

TensorFlow使用 **張量** （Tensor）作爲數據的基本單位。TensorFlow的張量在概念上等同於多維數組，我們可以使用它來描述數學中的標量（0維數組）、向量（1維數組）、矩陣（2維數組）等各種量，示例如下：

.. literalinclude:: /_static/code/zh/basic/eager/1plus1.py  
    :lines: 3-11

張量的重要屬性是其形狀、類型和值。可以通過張量的 ``shape`` 、 ``dtype`` 屬性和 ``numpy()`` 方法獲得。例如：

.. literalinclude:: /_static/code/zh/basic/eager/1plus1.py  
    :lines: 13-17

.. tip:: TensorFlow的大多數API函數會根據輸入的值自動推斷張量中元素的類型（一般默認爲 ``tf.float32`` ）。不過你也可以通過加入 ``dtype`` 參數來自行指定類型，例如 ``zero_vector = tf.zeros(shape=(2), dtype=tf.int32)`` 將使得張量中的元素類型均爲整數。張量的 ``numpy()`` 方法是將張量的值轉換爲一個NumPy數組。

TensorFlow里有大量的 **操作** （Operation），使得我們可以將已有的張量進行運算後得到新的張量。示例如下：

.. literalinclude:: /_static/code/zh/basic/eager/1plus1.py  
    :lines: 19-20

操作完成後， ``C`` 和 ``D`` 的值分別爲::
    
    tf.Tensor(
    [[ 6.  8.]
     [10. 12.]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[19. 22.]
     [43. 50.]], shape=(2, 2), dtype=float32)

可見，我們成功使用 ``tf.add()`` 操作計算出 :math:`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}`，使用 ``tf.matmul()`` 操作計算出 :math:`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\43 & 50 \end{bmatrix}` 。

.. _automatic_derivation:

自動求導機制
^^^^^^^^^^^^^^^^^^^^^^^^^^^

在機器學習中，我們經常需要計算函數的導數。TensorFlow提供了強大的 **自動求導機制** 來計算導數。在即時執行模式下，TensorFlow引入了 ``tf.GradientTape()`` 這個「求導記錄器」來實現自動求導。以下代碼展示了如何使用 ``tf.GradientTape()`` 計算函數 :math:`y(x) = x^2` 在 :math:`x = 3` 時的導數：

.. literalinclude:: /_static/code/zh/basic/eager/grad.py  
    :lines: 1-7

輸出::
    
    tf.Tensor(9.0, shape=(), dtype=float32)
    tf.Tensor(6.0, shape=(), dtype=float32)

這裡 ``x`` 是一個初始化爲3的 **變量** （Variable），使用 ``tf.Variable()`` 聲明。與普通張量一樣，變量同樣具有形狀、類型和值三種屬性。使用變量需要有一個初始化過程，可以通過在 ``tf.Variable()`` 中指定 ``initial_value`` 參數來指定初始值。這裡將變量 ``x`` 初始化爲 ``3.`` [#f0]_。變量與普通張量的一個重要區別是其默認能夠被TensorFlow的自動求導機制所求導，因此往往被用於定義機器學習模型的參數。

``tf.GradientTape()`` 是一個自動求導的記錄器。只要進入了 ``with tf.GradientTape() as tape`` 的上下文環境，則在該環境中計算步驟都會被自動記錄。比如在上面的示例中，計算步驟 ``y = tf.square(x)`` 即被自動記錄。離開上下文環境後，記錄將停止，但記錄器 ``tape`` 依然可用，因此可以通過 ``y_grad = tape.gradient(y, x)`` 求張量 ``y`` 對變量 ``x`` 的導數。

在機器學習中，更加常見的是對多元函數求偏導數，以及對向量或矩陣的求導。這些對於TensorFlow也不在話下。以下代碼展示了如何使用 ``tf.GradientTape()`` 計算函數 :math:`L(w, b) = \|Xw + b - y\|^2` 在 :math:`w = (1, 2)^T, b = 1` 時分別對 :math:`w, b` 的偏導數。其中 :math:`X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix},  y = \begin{bmatrix} 1 \\ 2\end{bmatrix}`。

.. literalinclude:: /_static/code/zh/basic/eager/grad.py  
    :lines: 9-16

輸出::

    tf.Tensor(125.0, shape=(), dtype=float32) 
    tf.Tensor(
    [[ 70.]
    [100.]], shape=(2, 1), dtype=float32) 
    tf.Tensor(30.0, shape=(), dtype=float32)

這裡， ``tf.square()`` 操作代表對輸入張量的每一個元素求平方，不改變張量形狀。 ``tf.reduce_sum()`` 操作代表對輸入張量的所有元素求和，輸出一個形狀爲空的純量張量（可以通過 ``axis`` 參數來指定求和的維度，不指定則默認對所有元素求和）。TensorFlow中有大量的張量操作API，包括數學運算、張量形狀操作（如 ``tf.reshape()``）、切片和連接（如 ``tf.concat()``）等多種類型，可以通過查閱TensorFlow的官方API文檔 [#f3]_ 來進一步了解。

從輸出可見，TensorFlow幫助我們計算出了

.. math::

    L((1, 2)^T, 1) &= 125
    
    \frac{\partial L(w, b)}{\partial w} |_{w = (1, 2)^T, b = 1} &= \begin{bmatrix} 70 \\ 100\end{bmatrix}
    
    \frac{\partial L(w, b)}{\partial b} |_{w = (1, 2)^T, b = 1} &= 30

..
    以上的自動求導機制結合 **優化器** ，可以計算函數的極值。這裡以線性回歸示例（本質是求 :math:`\min_{w, b} L = (Xw + b - y)^2` ，具體原理見 :ref:`後節 <linear-regression>` ）：

    .. literalinclude:: /_static/code/zh/basic/eager/regression.py  

.. _linear-regression:

基礎示例：線性回歸
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: 基礎知識和原理
    
    * UFLDL教程 `Linear Regression <http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/>`_ 一節。

考慮一個實際問題，某城市在2013年-2017年的房價如下表所示：

======  =====  =====  =====  =====  =====
年份    2013   2014   2015   2016   2017
房價    12000  14000  15000  16500  17500
======  =====  =====  =====  =====  =====

現在，我們希望通過對該數據進行線性回歸，即使用線性模型 :math:`y = ax + b` 來擬合上述數據，此處 ``a`` 和 ``b`` 是待求的參數。

首先，我們定義數據，進行基本的歸一化操作。

.. literalinclude:: /_static/code/zh/basic/example/numpy_manual_grad.py
    :lines: 1-7

接下來，我們使用梯度下降方法來求線性模型中兩個參數 ``a`` 和 ``b`` 的值 [#f1]_。

回顧機器學習的基礎知識，對於多元函數 :math:`f(x)` 求局部極小值，`梯度下降 <https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95>`_ 的過程如下：

* 初始化自變量爲 :math:`x_0` ， :math:`k=0` 
* 迭代進行下列步驟直到滿足收斂條件：

    * 求函數 :math:`f(x)` 關於自變量的梯度 :math:`\nabla f(x_k)` 
    * 更新自變量： :math:`x_{k+1} = x_{k} - \gamma \nabla f(x_k)` 。這裡 :math:`\gamma` 是學習率（也就是梯度下降一次邁出的「步子」大小）
    * :math:`k \leftarrow k+1` 

接下來，我們考慮如何使用程序來實現梯度下降方法，求得線性回歸的解 :math:`\min_{a, b} L(a, b) = \sum_{i=1}^n(ax_i + b - y_i)^2` 。

NumPy下的線性回歸
-------------------------------------------

機器學習模型的實現並不是TensorFlow的專利。事實上，對於簡單的模型，即使使用常規的科學計算庫或者工具也可以求解。在這裡，我們使用NumPy這一通用的科學計算庫來實現梯度下降方法。NumPy提供了多維數組支持，可以表示向量、矩陣以及更高維的張量。同時，也提供了大量支持在多維數組上進行操作的函數（比如下面的 ``np.dot()`` 是求內積， ``np.sum()`` 是求和）。在這方面，NumPy和MATLAB比較類似。在以下代碼中，我們手工求損失函數關於參數 ``a`` 和 ``b`` 的偏導數 [#f2]_，並使用梯度下降法反覆迭代，最終獲得 ``a`` 和 ``b`` 的值。

.. literalinclude:: /_static/code/zh/basic/example/numpy_manual_grad.py
    :lines: 9-

然而，你或許已經可以注意到，使用常規的科學計算庫實現機器學習模型有兩個痛點：

- 經常需要手工求函數關於參數的偏導數。如果是簡單的函數或許還好，但一旦函數的形式變得複雜（尤其是深度學習模型），手工求導的過程將變得非常痛苦，甚至不可行。
- 經常需要手工根據求導的結果更新參數。這裡使用了最基礎的梯度下降方法，因此參數的更新還較爲容易。但如果使用更加複雜的參數更新方法（例如Adam或者Adagrad），這個更新過程的編寫同樣會非常繁雜。

而TensorFlow等深度學習框架的出現很大程度上解決了這些痛點，爲機器學習模型的實現帶來了很大的便利。

.. _optimizer:

TensorFlow下的線性回歸
-------------------------------------------

TensorFlow的 **即時執行模式** [#f4]_ 與上述NumPy的運行方式十分類似，然而提供了更快速的運算（GPU支持）、自動求導、優化器等一系列對深度學習非常重要的功能。以下展示了如何使用TensorFlow計算線性回歸。可以注意到，程序的結構和前述NumPy的實現非常類似。這裡，TensorFlow幫助我們做了兩件重要的工作：

* 使用 ``tape.gradient(ys, xs)`` 自動計算梯度；
* 使用 ``optimizer.apply_gradients(grads_and_vars)`` 自動更新模型參數。

.. literalinclude:: /_static/code/zh/basic/example/tensorflow_eager_autograd.py
    :lines: 10-29

在這裡，我們使用了前文的方式計算了損失函數關於參數的偏導數。同時，使用 ``tf.keras.optimizers.SGD(learning_rate=5e-4)`` 聲明了一個梯度下降 **優化器** （Optimizer），其學習率爲5e-4。優化器可以幫助我們根據計算出的求導結果更新模型參數，從而最小化某個特定的損失函數，具體使用方式是調用其 ``apply_gradients()`` 方法。

注意到這裡，更新模型參數的方法 ``optimizer.apply_gradients()`` 需要提供參數 ``grads_and_vars``，即待更新的變量（如上述代碼中的 ``variables`` ）及損失函數關於這些變量的偏導數（如上述代碼中的 ``grads`` ）。具體而言，這裡需要傳入一個Python列表（List），列表中的每個元素是一個 ``（變量的偏導數，變量）`` 對。比如上例中需要傳入的參數是 ``[(grad_a, a), (grad_b, b)]`` 。我們通過 ``grads = tape.gradient(loss, variables)`` 求出tape中記錄的 ``loss`` 關於 ``variables = [a, b]`` 中每個變量的偏導數，也就是 ``grads = [grad_a, grad_b]``，再使用Python的 ``zip()`` 函數將 ``grads = [grad_a, grad_b]`` 和 ``variables = [a, b]`` 拼裝在一起，就可以組合出所需的參數了。

.. admonition:: Python的 ``zip()`` 函數

    ``zip()`` 函數是Python的內置函數。用自然語言描述這個函數的功能很繞口，但如果舉個例子就很容易理解了：如果 ``a = [1, 3, 5]``， ``b = [2, 4, 6]``，那麼 ``zip(a, b) = [(1, 2), (3, 4), ..., (5, 6)]`` 。即「將可迭代的對象作爲參數，將對象中對應的元素打包成一個個元組，然後返回由這些元組組成的列表」，和我們日常生活中拉上拉鏈（zip）的操作有異曲同工之妙。在Python 3中， ``zip()`` 函數返回的是一個 zip 對象，本質上是一個生成器，需要調用 ``list()`` 來將生成器轉換成列表。

    .. figure:: /_static/image/basic/zip.jpg
        :width: 60%
        :align: center

        Python的 ``zip()`` 函數圖示

在實際應用中，我們編寫的模型往往比這裡一行就能寫完的線性模型 ``y_pred = a * X + b`` （模型參數爲 ``variables = [a, b]`` ）要複雜得多。所以，我們往往會編寫並實例化一個模型類 ``model = Model()`` ，然後使用 ``y_pred = model(X)`` 調用模型，使用 ``model.variables`` 獲取模型參數。關於模型類的編寫方式可見 :doc:`"TensorFlow模型"一章 <models>`。

.. [#f0] Python中可以使用整數後加小數點表示將該整數定義爲浮點數類型。例如 ``3.`` 代表浮點數 ``3.0``。
.. [#f3] 主要可以參考 `Tensor Transformations <https://www.tensorflow.org/versions/r1.9/api_guides/python/array_ops>`_ 和 `Math <https://www.tensorflow.org/versions/r1.9/api_guides/python/math_ops>`_ 兩個頁面。可以注意到，TensorFlow的張量操作API在形式上和Python下流行的科學計算庫NumPy非常類似，如果對後者有所了解的話可以快速上手。
.. [#f1] 其實線性回歸是有解析解的。這裡使用梯度下降方法只是爲了展示TensorFlow的運作方式。
.. [#f2] 此處的損失函數爲均方誤差 :math:`L(x) = \sum_{i=1}^N (ax_i + b - y_i)^2`。其關於參數 ``a`` 和 ``b`` 的偏導數爲 :math:`\frac{\partial L}{\partial a} = 2 \sum_{i=1}^N (ax_i + b - y) x_i`，:math:`\frac{\partial L}{\partial b} = 2 \sum_{i=1}^N (ax_i + b - y)` 。本例中 :math:`N = 5` 。由於均方誤差取均值的係數 :math:`\frac{1}{N}` 在訓練過程中一般爲常數（ :math:`N` 一般爲批次大小），對損失函數乘以常數等價於調整學習率，因此在具體實現時通常不寫在損失函數中。
.. [#f4] 與即時執行模式相對的是圖執行模式（Graph Execution），即 TensorFlow 2 之前所主要使用的執行模式。本手冊以面向快速迭代開發的即時執行模式爲主，但會在 :doc:`附錄 <../appendix/static>` 中介紹圖執行模式的基本使用，供需要的讀者查閱。

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 189 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>