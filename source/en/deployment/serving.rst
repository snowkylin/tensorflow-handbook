TensorFlow Serving
==================

When we have the model trained, it is often necessary to deploy the model in a production environment. The most common way to do this is to provide an API on the server, where the client sends a request in a specific format to one of the server's APIs, then the server receives the requested data, computes it through the model, and returns the results. The server API can be implemented very easily with Python web frameworks such as `Flask <https://palletsprojects.com/p/flask/>`_ if what we want is just a demo, regardless of the high concurrency and performance issues. However, most of the real production environment is not that case. Therefore, TensorFlow provides us with TensorFlow Serving, a serving system that helps us deploy machine learning models flexibly and with high performance in real production environments.

Installation of TensorFlow Serving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow Serving can be installed using either apt-get or Docker. In a production environment, it is recommended to `use Docker to deploy TensorFlow Serving <https://www.tensorflow.org/tfx/serving/docker>`_. However, as a tutorial, we will introduce `apt-get installization <https://www.tensorflow.org/tfx/serving/setup#installing_using_apt>`_ which do not rely on docker..

.. admonition:: Hint

    Software installation method is time-sensitive, this section is updated in August 2019. If you encounter problems, it is recommended to refer to the latest installation instructions on the `TensorFlow website <https://www.tensorflow.org/tfx/serving/setup>`_.

First add the package source:

::

    # Add the TensorFlow Serving package source from Google
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
    # Add gpg key
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

Then you can use apt-get to install TensorFlow Serving

::

    sudo apt-get update
    sudo apt-get install tensorflow-model-server

.. admonition:: Hint 

    You can use `Windows Subsystem Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ to install TensorFlow Serving on Windows for development.

TensorFlow Serving models deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow Serving can read models in SavedModel format directly for deployment (see :ref:`previous chapter <en_savedmodel>` for exporting models to SavedModel files). The command is as follows

::

    tensorflow_model_server \
        --rest_api_port=PORT(e.g., 8501) \
        --model_name=MODEL_NAME \
        --model_base_path="Absolute folder path of the SavedModel format model (without version number)"

.. note:: TensorFlow Serving supports hot update of models with the following typical model folder structure.

    ::
        
        /saved_model_files
            /1      # model files of version 1
                /assets
                /variables
                saved_model.pb
            ...
            /N      # model files of version N
                /assets
                /variables
                saved_model.pb

    
    The subfolders from 1 to N above represent models with different version numbers. When specifying ``--model_base_path``, you only need to specify the absolute address (not the relative address) of the root directory. For example, if the above folder structure is in the ``home/snowkylin`` folder, then ``--model_base_path`` should be set to ``home/snowkylin/saved_model_files`` (without the model version number). TensorFlow Serving will automatically select the model with the largest version number for loading. 

Keras Sequential mode models deployment
---------------------------------------

Since the Sequential model has fixed inputs and outputs, this type of model is easy to deploy without additional operations. For example, to deploy the `MNIST handwriting digit classification model <en_savedmodel>`_ (built using the Keras Sequential mode) in the previous chapter using SavedModel with the model name ``MLP`` on port ``8501``, you can directly use the following command.

::

    tensorflow_model_server \
        --rest_api_port=8501 \
        --model_name=MLP \
        --model_base_path="/home/.../.../saved"  # The absolute address of the SavedModel folder without version number

The model can then be called on the client using gRPC or RESTful API as described :ref:`later <en_call_serving_api>`.

Custom Keras models deployment
------------------------------

Custom Keras models built inheriting ``tf.keras.Model`` class are more flexible. Therefore, when using the TensorFlow Serving deployment model, there are additional requirements for the exported SavedModel file.

- Methods that need to be exported to the SavedModel format (e.g. ``call``) require not only being decorated by ``@tf.function``, but also specifying the ``input_signature`` parameter at the time of decoration to explicitly describe the input shape, using a list of ``tf.TensorSpec`` specifying the shape and type of each input tensor. For example, for MNIST handwriting digit classification, the input is a four-dimensional tensor of ``[None, 28, 28, 1]`` (``None`` denotes that the first dimension, i.e., the batch size, is not fixed). We can decorate the ``call`` method of the model as follows.

.. code-block:: python
    :emphasize-lines: 4

    class MLP(tf.keras.Model):
        ...

        @tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)])
        def call(self, inputs):
            ...

- When exporting the model using ``tf.saved_model.save``, we need to provide an additional "signature of the function to be exported" via the ``signature`` parameter. In short, since there may be multiple methods in a custom model class that need to be exported, TensorFlow Serving needs to be told which method is called when receiving a request from the client. For example, if we want to assign signature ``call`` to the ``model.call`` method, we can pass the ``signature`` parameter when exporting to tell the correspondence between the signature and the method to be exported, in the form of a key-value pair of ``dict``. The following code is an example

.. code-block:: python
    :emphasize-lines: 3

    model = MLP()
    ...
    tf.saved_model.save(model, "saved_with_signature/1", signatures={"call": model.call})

Once both of these steps have been completed, you can deploy the model using the following commands

::

    tensorflow_model_server \
        --rest_api_port=8501 \
        --model_name=MLP \
        --model_base_path="/home/.../.../saved_with_signature"  # 修改为自己模型的绝对地址

.. _en_call_serving_api:

Calling models deployed by TensorFlow Serving on client
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    https://www.tensorflow.org/tfx/serving/api_rest
    http://www.ruanyifeng.com/blog/2014/05/restful_api.html

TensorFlow Serving supports gRPC and RESTful API for models deployed with TensorFlow Serving. This handbook mainly introduces the more general RESTful API method.

The RESTful API use standard HTTP POST method, with both requests and response being JSON objects. In order to call the server-side model, we send a request to the server on the client side in the following format.

Server URI: ``http://SERVER_ADDRESS:PORT/v1/models/MODEL_NAME:predict`` 

Content of the request:

::

    {
        "signature_name": "the signature of the method to be called (do not request for Sequential models)",
        "instances": input data
    }

The format of the response is

::

    {
        "predictions": the returned value
    }

An example of Python client
---------------------------

The following example uses `Python's Requests library <https://2.python-requests.org//zh_CN/latest/user/quickstart.html>`_ (which you may need to install via ``pip install requests``) to send the first 10 images of the MNIST test set to the local TensorFlow Serving server and return the predicted results, which are then compared to the actual tags of the test set.

.. literalinclude:: /_static/code/zh/savedmodel/keras/client.py

Output:

::

    [7 2 1 0 4 1 4 9 6 9]
    [7 2 1 0 4 1 4 9 5 9]

It can be seen that the predicted results are very close to the true label values.

For a custom Keras model, simply add the ``signature_name`` parameter to the sent data, changing the ``data`` build process in the above code to

.. literalinclude:: /_static/code/zh/savedmodel/custom/client.py
    :lines: 8-11

An example of Node.js client
----------------------------

The following example uses `Node.js <https://nodejs.org/zh-cn/>`_ to convert the following image to a 28*28 grayscale image, send it to the local TensorFlow Serving server, and output the returned predicted values and probabilities. This program uses the `image processing library jimp <https://github.com/oliver-moran/jimp>`_ and the `HTTP library superagent <https://visionmedia.github.io/superagent/>`_, which can be installed using ``npm install jimp`` and ``npm install superagent``.

.. figure:: /_static/image/serving/test_pic_tag_5.png
    :align: center

    ``test_pic_tag_5.png`` : A handwritten number 5. (This image can be downloaded and placed in the same directory as the code when running the code below)

.. literalinclude:: /_static/code/zh/savedmodel/keras/client.js
    :language: javascript

The output is

::

    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1               1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1                 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1       1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1       1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1     1                 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1                         1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1         1 1 1 1 1 1     1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1       1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1         1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1         1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1     1 1 1         1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1                 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1         1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    We guess the number is 5, the probability is 0.846008837

The output can be seen to be as expected.

.. admonition:: Note

    If you are not familiar with HTTP POST, you can refer to `this article <https://www.runoob.com/tags/html-httpmethods.html>`_. In fact, when you fill out a form in your browser (let's say a personality test), click the "Submit" button and get a return result (let's say "Your personality is ISTJ"), you are most likely sending an HTTP POST request to the server and getting a response from the server.

    RESTful API is a popular API design theory that is briefly described at `this article <http://www.ruanyifeng.com/blog/2014/05/restful_api.html>`_.

    The complete use of the RESTful API for TensorFlow Serving can be found in the `documentation <https://www.tensorflow.org/tfx/serving/api_rest>`_.

.. raw:: html

    <script>
        $(document).ready(function(){
            $(".rst-footer-buttons").after("<div id='discourse-comments'></div>");
            DiscourseEmbed = { discourseUrl: 'https://discuss.tf.wiki/', topicId: 358 };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript'; d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        });
    </script>



