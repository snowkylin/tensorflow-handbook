import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# 以下三行定義了一個簡單的“計算圖”
a = tf.constant(1)  # 定義一個常數變數（Tensor）
b = tf.constant(1)
c = a + b           # 等同於 c = tf.add(a, b)，c是變數a和變數b通過 tf.add 這一操作（Operation）所形成的新變數
# 到此為止，計算圖定義完畢，然而程式還沒有進行任何實質計算。
# 如果此時直接輸出變數 c 的值，是無法獲得 c = 2 的結果的

sess = tf.Session()     # 實例化一個會話（Session）
c_ = sess.run(c)        # 通過會話的 run() 方法對計算圖里的節點（變數）進行實際的計算
print(c_)