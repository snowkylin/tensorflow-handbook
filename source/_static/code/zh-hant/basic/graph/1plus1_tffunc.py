import tensorflow as tf

# 以下被 @tf.function 修飾的函數定義了一個計算圖
@tf.function
def graph():
    a = tf.constant(1)
    b = tf.constant(1)
    c = a + b
    return c
# 到此為止，計算圖定義完畢。由於 graph() 是一個函數，在其被呼叫之前，程式是不會進行任何實質計算的。
# 只有呼叫函數，才能通過函數取得回傳值，取得 c = 2 的結果

c_ = graph()
print(c_.numpy())