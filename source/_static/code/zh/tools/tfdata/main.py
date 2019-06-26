import tensorflow as tf
import os

num_epochs = 10
batch_size = 32
learning_rate = 0.001

def _parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize(image_decoded, [256, 256])
    return image_resized, label

if __name__ == '__main__':
    train_cat_filenames = tf.constant(['cat_and_dog/train/cats/' + filename for filename in os.listdir('cat_and_dog/train/cats')])
    train_dog_filenames = tf.constant(['cat_and_dog/train/dogs/' + filename for filename in os.listdir('cat_and_dog/train/dogs')])
    train_filenames = tf.concat([train_cat_filenames, train_dog_filenames], axis=-1)
    labels = tf.concat([tf.zeros(train_cat_filenames.shape, dtype=tf.int32), tf.ones(train_dog_filenames.shape, dtype=tf.int32)], axis=-1)

    dataset = tf.data.Dataset.from_tensor_slices((train_filenames, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    batched_dataset = dataset.batch(batch_size)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

    model.fit(batched_dataset, epochs=num_epochs)

