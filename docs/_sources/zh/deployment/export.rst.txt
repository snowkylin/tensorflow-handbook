TensorFlow模型导出
==========================

SavedModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TODO: 2.0 eager + Keras Sequential save model way

Keras Sequential save方法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们以keras模型训练和保存为例进行讲解，如下是keras官方的mnist模型训练样例。

源码地址::
    
    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

引用源码如下

.. code-block:: python

    '''Trains a simple convnet on the MNIST dataset.
    
    Gets to 99.25% test accuracy after 12 epochs
    (there is still a lot of margin for parameter tuning).
    16 seconds per epoch on a GRID K520 GPU.
    '''
    
    from __future__ import print_function
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
    
    batch_size = 128
    num_classes = 10
    epochs = 12
    
    # input image dimensions
    img_rows, img_cols = 28, 28
    
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

以上代码，是基于keras的Sequential构建了多层的卷积神经网络，并进行训练。

为了方便起见可使用如下命令拷贝到本地::

    curl -LO https://raw.githubusercontent.com/keras-team/keras/master/examples/mnist_cnn.py

然后，在最后加上如下一行代码（主要是对keras训练完毕的模型进行保存）::

    model.save('mnist_cnn.h5')


在终端中执行mnist_cnn.py文件，如下::

    python mnist_cnn.py

.. warning:: 该过程需要连接网络获取mnist.npz文件（https://s3.amazonaws.com/img-datasets/mnist.npz），会被保存到$HOME/.keras/datasets/。如果网络连接存在问题，可以通过其他方式获取mnist.npz后，直接保存到该目录即可。

执行过程会比较久，执行结束后，会在当前目录产生`mnist_cnn.h5`文件（HDF5格式），就是keras训练后的模型，其中已经包含了训练后的模型结构和权重等信息。

在服务器端，可以直接通过keras.models.load_model("mnist_cnn.h5")加载，然后进行推理；在移动设备需要将HDF5模型文件转换为TensorFlow Lite的格式，然后通过相应平台的Interpreter加载，然后进行推理。
