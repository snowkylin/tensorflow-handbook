import tensorflow as tf

x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:     # tf.GradientTape() 是上下文管理器，tape 是记录器
    y = tf.square(x)
    with tape.stop_recording():     # 在上下文管理器内，记录进行中，暂时停止记录成功
        print('temporarily stop recording')
with tape.stop_recording():         # 在上下文管理器外，记录已停止，尝试暂时停止记录报错
    pass
y_grad = tape.gradient(y, x)        # 在上下文管理器外，tape 的记录信息仍然保留，导数计算成功