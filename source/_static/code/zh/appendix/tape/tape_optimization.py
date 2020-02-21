import tensorflow as tf

x = tf.Variable(3.)
with tf.GradientTape(persistent=True) as tape:
    y = tf.square(x)
    y_grad = tape.gradient(y, x)    # 如果后续并不需要对 y_grad 求导，则不建议在上下文环境中求导
    with tape.stop_recording():     # 对于无需记录求导的计算步骤，可以暂停记录器后计算
        y_grad_not_recorded = tape.gradient(y, x)
d2y_dx2 = tape.gradient(y_grad, x)  # 如果后续需要对 y_grad 求导，则 y_grad 必须写在上下文中