import tensorflow as tf
tf.debugging.set_log_device_placement(True)     # 設置輸入運算所在的設備

cpus = tf.config.list_physical_devices('CPU')   # 取得當前設備的CPU列表
tf.config.set_visible_devices(cpus)             # 設置TensorFlow的可用設備範圍為cpu

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)

print(C)