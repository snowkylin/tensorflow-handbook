import tensorflow as tf

x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:     # tf.GradientTape() 是上下文管理器，tape 是紀錄器
    y = tf.square(x)
    with tape.stop_recording():     # 在上下文管理器內，紀錄進行中，暫時停止紀錄成功
        print('temporarily stop recording')
with tape.stop_recording():         # 在上下文管理器外，紀錄已停止，嘗試暫時停止報錯
    pass
y_grad = tape.gradient(y, x)        # 在上下文管理器外，tape 的紀錄資訊仍然保留，導數計算成功