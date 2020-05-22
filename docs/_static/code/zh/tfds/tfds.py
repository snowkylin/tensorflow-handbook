import tensorflow as tf
import tensorflow_datasets as tfds

dataset = tfds.load('tf_flowers', split=tfds.Split.TRAIN)
# dataset = dataset.shuffle(1024).batch(32)
for data in dataset:
    print(data)