import tensorflow as tf
import os

data_dir = 'C:/datasets/cats_vs_dogs'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
tfrecord_file = data_dir + '/train/train.tfrecords'

train_cat_filenames = [train_cats_dir + filename for filename in os.listdir(train_cats_dir)]
train_dog_filenames = [train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)]
train_filenames = train_cat_filenames + train_dog_filenames
train_labels = [0] * len(train_cat_filenames) + [1] * len(train_dog_filenames)  # 將 cat 類的標籤設為0，dog 類的標籤設為1

with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for filename, label in zip(train_filenames, train_labels):
        image = open(filename, 'rb').read()     # 讀取資料集圖片到內存，image 為一個
        feature = {                             # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 圖片是一個 Bytes 對象
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 標籤是一個 Int 對象
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通過字典建立 Example
        writer.write(example.SerializeToString())   # 將Example序列化並寫入 TFRecord 文件

raw_dataset = tf.data.TFRecordDataset(tfrecord_file)    # 讀取 TFRecord 文件

feature_description = { # 定義Feature結構，告訴解碼器每個Feature的類型是什麼
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_example(example_string): # 將 TFRecord 文件中的每一個序列化的 tf.train.Example 解碼
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])    # 解碼JPEG圖片
    return feature_dict['image'], feature_dict['label']

dataset = raw_dataset.map(_parse_example)

import matplotlib.pyplot as plt 

for image, label in dataset:
    plt.title('cat' if label == 0 else 'dog')
    plt.imshow(image.numpy())
    plt.show()