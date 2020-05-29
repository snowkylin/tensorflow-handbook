import tensorflow as tf
import numpy as np

# TensorFlow 的圖片表示為 [圖片數目，長，寬，色彩通道數] 的四維張量
# 這裡我們的輸入圖片 image 的張量形狀為 [1, 7, 7, 1]
image = np.array([[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 2, 1, 0],
    [0, 0, 2, 2, 0, 1, 0],
    [0, 1, 1, 0, 2, 1, 0],
    [0, 0, 2, 1, 1, 0, 0],
    [0, 2, 1, 1, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]], dtype=np.float32)
image = np.expand_dims(image, axis=-1)  
W = np.array([[
    [ 0, 0, -1], 
    [ 0, 1, 0 ], 
    [-2, 0, 2 ]
]], dtype=np.float32)
b = np.array([1], dtype=np.float32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        filters=1,              # 卷積層神經元（卷積核）數目
        kernel_size=[3, 3],     # 接受區大小
        kernel_initializer=tf.constant_initializer(W),
        bias_initializer=tf.constant_initializer(b)
    )]
)

output = model(image)
print(tf.squeeze(output))
