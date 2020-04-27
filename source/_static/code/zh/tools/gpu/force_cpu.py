import tensorflow as tf
tf.debugging.set_log_device_placement(True)     # 设置输出运算所在的设备

cpus = tf.config.list_physical_devices('CPU')   # 获取当前设备的CPU列表
tf.config.set_visible_devices(cpus)             # 设置TensorFlow的可见设备范围为cpu

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)

print(C)