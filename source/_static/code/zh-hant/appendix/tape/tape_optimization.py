import tensorflow as tf

x = tf.Variable(3.)
with tf.GradientTape(persistent=True) as tape:
    y = tf.square(x)
    y_grad = tape.gradient(y, x)    # 如果後續並不需要對 y_grad 求導，則不建議在上下文環境中求解
    with tape.stop_recording():     # 對於無需紀錄求導的計算步驟，可以暫停紀錄器後計算
        y_grad_not_recorded = tape.gradient(y, x)
d2y_dx2 = tape.gradient(y_grad, x)  # 如果後續需要對 y_grad 求導，則 y_grad 必須寫在上下文中