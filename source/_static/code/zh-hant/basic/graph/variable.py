import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

a = tf.get_variable(name='a', shape=[])
initializer = tf.assign(a, 0.0)   # tf.assign(x, y)返回一個“將變數y的值指定給變數x”的操作
plus_one_op = tf.assign(a, a + 1.0)

sess = tf.Session()
sess.run(initializer)
for i in range(5):
    sess.run(plus_one_op)       # 對變數a執行此一操作
    print(sess.run(a))          # 輸出此時變數a在當前會話的計算圖的值
