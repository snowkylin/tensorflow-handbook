Tensorflow Python Chitchat Bot
==============================

在本章，我们实现一个可以用来闲聊的对话模型。这个对话模型将基于序列到序列（Seq2Seq）来对电影台词中的对话数据进行训练。

序列到序列模型（Sequence to Sequence, SEQ2SEQ）是一种基于 RNN 的 Encoder-Decoder 结构，它也是现在谷歌应用于线上机器翻译的算法，翻译质量已经和人类水平不相上下。

这里通过 Keras 自定义模型建立一个闲聊对话模型（Seq2Seq）。它使用 Encoder-Decoder 结构，简单的来说就是算法包含两部分，一个负责对输入的信息进行 Encoding，将输入转换为向量形式；然后由 Decoder 对这个向量进行解码，还原为输出序列。

关于 Seq2Seq 的原理和介绍，可以参考 Keras 的博客：A ten-minute introduction to sequence-to-sequence learning in Keras。地址： https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

这里，我们使用 Seq2Seq 来实现一个闲聊（ChitChat）对话机器人。除了闲聊任务（输入一句话，输出一句回复）之外，它也可以被直接应用于解决其他类似问题，比如：翻译（输入一句英文，输出一句中文）、摘要（输入一篇文章，输出一份总结）、作诗（输入几个关键字，输出一首短诗）、对对联（输入上联，输出下联），等等。

这个任务对比与之前的RNN尼采风格文本生成，区别在于我们预测的不再是文本的连续字母概率分布，而是通过一个序列，来预测另外一个对应的完整序列。举例来说，针对一句常见的打招呼::

    How are you


这个句子（序列）一共有3个单词。当我们听到这个由3个单词组成的句子后，根据我们的习惯，我们最倾向与回复的一句话是 "Fine thank you"。我们希望建立这样的模型，输入 num_batch 个由编码后单词和字符组成的，长为 max_length 的序列，输入张量形状为 [num_batch, max_length]，输出与这个序列对应的序列（如聊天回复、翻译等）中单词和字符的概率分布，概率分布的维度为词汇表大小 voc_size，输出张量形状为 [num_batch, max_length, voc_size]。

首先，还是实现一个简单的 ``DataLoader`` 类来读取文本，

.. code-block:: python

    DATASET_URL = 'https://github.com/huan/python-concise-chitchat/releases/download/v0.0.1/dataset.txt.gz'
    DATASET_FILE_NAME = 'concise-chitchat-dataset.txt.gz'
    START_TOKEN = '\t'
    END_TOKEN = '\n'

    class DataLoader():
        def __init__(self):
            dataset_file = tf.keras.utils.get_file(DATASET_FILE_NAME, origin=DATASET_URL)
            with gzip.open(dataset_file, 'rt') as f:
                self.raw_text = f.read().lower()
            self.queries, self.responses = self.__parse_raw_text(self.raw_text)
            self.size = len(self.queries)

        def get_batch(self, batch_size=32):
            batch_indices = np.random.choice(len(self.queries), size=batch_size)
            batch_queries = self.queries[batch_indices]
            batch_responses = self.responses[batch_indices]
            return batch_queries, batch_responses

        def __parse_raw_text(self, raw_text: str):
            query_list = []
            response_list = []
            for line in raw_text.strip('\n').split('\n'):
                query, response = line.split('\t')
                query_list.append(query)
                response_list.append('{} {} {}'.format(START_TOKEN, response, END_TOKEN))
            return np.array(query_list), np.array(response_list)

其次，我们还需要基于 `DataLoader` 加载的文本数据，建立一个词汇表 `Vocabulary` 来负责管理以下5项任务：

1. 将所有单词和标点符号进行编码；
2. 记录词汇表大小；
3. 建立单词到编码数字，以及编码数字到单词的映射字典；
4. 负责将文本句子转化为填充后的编码序列，形状为[batch_size, max_length]；

.. code-block:: python

    MAX_LEN = 10

    class Vocabulary:
        def __init__(self, text):
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
            self.tokenizer.fit_on_texts([END_TOKEN, START_TOKEN] + re.split(r'[\n\s\t]',text))
            self.size = 1 + len(self.tokenizer.word_index.keys())

        def texts_to_padded_sequences(self, text_list):
            sequence_list = self.tokenizer.texts_to_sequences(text_list)
            padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
                sequence_list, maxlen=MAX_LEN, padding='post',truncating='post')
            return padded_sequences

接下来进行模型的实现。我们建立一个ChitChat模型。ChitChat 模型是一个 Seq2Seq 的模型，主要由 ChitEncoder 和 ChatDecoder 组成。

ChitEncoder 子模型输入 num_batch 个由编码后单词和字符组成的，长为 max_length 的序列，输入张量形状为 [num_batch, max_length]，输出与这个序列对应的上下文张量。为了简化代码，我们这里只使用一个最基本的 GRU 单元，没有使用可以获得更佳效果的双向RNN、注意力机制等方法。在 `__init__` 方法中我们实例化一个常用的 `GRU` 单元，并将其设置为 `return_state=True` 来获得最终的状态输出，我们首先对序列进行 GRU 操作，即将编码序列变换为 GRU 最终输出的状态 ，并将其作为代表编码序列的上下文信息 `context` ，作为模型的输出。

`ChitEncoder` 子模型具体实现如下：

.. code-block:: python

    RNN_UNIT_NUM = 512
    EMBEDDING_DIM = 512

    class ChitEncoder(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.gru = tf.keras.layers.GRU(units=RNN_UNIT_NUM,
                return_sequences=True, return_state=True)

        def call(self, inputs):
            [outputs, state] = self.gru(inputs)
            return outputs, state

ChatDecoder 子模型输入 num_batch 个编码后的一个单词或字符的 Embedding，和当前的上下文信息张量 `initial_state` 两个信息构成，输入张量形状分别为 [num_batch, 1, EMBEDDING_DIM]，和 [num_batch, RNN_UNIT_NUM]。在 `__init__` 方法中我们保存词汇表容量 `voc_size` ，实例化一个常用的 `GRU` 单元，并将其设置为输出单元状态 `return_state=True` 来获得 GRU 的状态输出，以及一个全连接层 `Dense` 单元，负责将 GRU 的输出变换为最终的单词字符分布概率，并将其作为这个上下文信息对应的单词符号序列概率分布张量，作为模型的输出，形状为[num_batch, 1, voc_size]。

`ChitDecoder` 子模型具体实现如下：

.. code-block:: python

    class ChatDecoder(tf.keras.Model):
        def __init__(self, voc_size):
            super().__init__()
            self.voc_size = voc_size
            self.gru = tf.keras.layers.GRU(units=RNN_UNIT_NUM, return_state=True)
            self.dense = tf.keras.layers.Dense(units=self.voc_size)

        def call(self, inputs, initial_state):
            outputs, state = self.gru(inputs=inputs, initial_state=[initial_state])
            outputs = self.dense(outputs)
            return outputs, state

构建 ChitChat 模型将基于上面的两个 ChitEncoder 和 ChatDecoder 子模型。在 `__init__` 方法中我们将 `Vocabulary` 中的词汇到编码字典 `word_index` 和编码到词汇字典 `index_word` ，以及词汇量 `voc_size` 保存备用，实例化一个词向量的 `Embedding` 单元，以及一个 `ChitEncoder` 子模型和对应的 `ChatDecoder` 子模型。`ChatDecoder` 子模型中需要使用词汇表尺寸，我们通过构造参数传给它。

模型的工作流程为：我们首先对输入序列通过 `Embedding` 层进行词向量转换，然后进行 Encoder 操作，即将编码序列 `inputs` 的词嵌入向量，变换为一个上下文向量 `encoder_hidden_state` 。然后，我们进入解码流程：将 START_TOKEN 词向量和 `encoder_hidden_state` 作为解码器的首次输入，解码得到解码器的输出编码张量 `decoder_outputs`，以及状态张量 `decoder_state`。接下来将 `decoder_outputs` 和 `decoder_state` 重复输入解码器，即可不断得到新的 `decoder_outputs` 即作为模型的输出，直到 `decoder_outputs` 解码出来的字符为 END_TOKEN 为止。最终输出的张量形状为[num_batch, max_length, voc_size]。

`ChitChat` 模型具体实现如下：

.. code-block:: python


    class ChitChat(tf.keras.Model):
        def __init__(self, vocabulary):
            super().__init__()
            self.word_index = vocabulary.tokenizer.word_index
            self.index_word = vocabulary.tokenizer.index_word
            self.voc_size = vocabulary.size

            self.indice_sos = self.word_index[START_TOKEN]
            self.indice_eos = self.word_index[END_TOKEN]

            self.embedding = tf.keras.layers.Embedding(
                input_dim=self.voc_size,output_dim=EMBEDDING_DIM)
            self.encoder = ChitEncoder()
            self.decoder = ChatDecoder(voc_size=self.voc_size)

        def call(self, inputs, training=False, teacher_forcing_targets=None):
            inputs = tf.convert_to_tensor(inputs)
            batch_size = tf.shape(inputs)[0]

            inputs = self.embedding(inputs)
            encoder_outputs, encoder_hidden_state = self.encoder(inputs=inputs)

            batch_sos_one_hot = tf.ones([batch_size, 1, 1]) \
                * [tf.one_hot(self.indice_sos, self.voc_size)]

            decoder_output = batch_sos_one_hot
            decoder_state = encoder_hidden_state

            outputs = tf.zeros([batch_size, 0, self.voc_size])

            for t in range(0, MAX_LEN):
                if training and teacher_forcing_targets is not None:
                    target_indice = tf.expand_dims(
                        teacher_forcing_targets[:, t], axis=-1)
                else:
                    target_indice = tf.argmax(decoder_output, axis=-1)
                decoder_inputs = self.embedding(target_indice)
                decoder_output, decoder_state = self.decoder(
                    inputs=decoder_inputs,
                    initial_state=decoder_state,
                )
                outputs = tf.concat([outputs, decoder_output], axis=1)
            return outputs

训练过程与本书的 RNN 模型训练基本一致，在此复述：

- 从DataLoader中随机取一批训练数据；
- 将这批数据送入模型，计算出模型的预测值；
- 将模型预测值与真实值进行比较，计算损失函数（loss）；
- 计算损失函数关于模型变量的导数；
- 使用优化器更新模型参数以最小化损失函数。

.. code-block:: python

    LEARNING_RATE = 1e-3
    NUM_STEP = 10000
    BATCH_SIZE = 64

    def loss_function(model, x, y):
        predictions = model(inputs=x, training=True, teacher_forcing_targets=y)
        y_without_sos = tf.concat([y[:, 1:],
            tf.expand_dims(tf.fill([BATCH_SIZE], 0.), axis=1)],axis=1)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_without_sos, logits=predictions)

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss_function(model, inputs, targets)
        return tape.gradient(loss_value, model.variables)

    data_loader = DataLoader()
    vocabulary = Vocabulary(data_loader.raw_text)
    chitchat = ChitChat(vocabulary=vocabulary)
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=chitchat)

    for batch_index in range(NUM_STEP):
        queries, responses = data_loader.get_batch(BATCH_SIZE)

        queries_sequences = vocabulary.texts_to_padded_sequences(queries)
        responses_sequences = vocabulary.texts_to_padded_sequences(responses)

        grads = grad(chitchat, queries_sequences, responses_sequences)
        optimizer.apply_gradients(grads_and_vars=zip(grads, chitchat.variables))

        print("step %d: loss %f" % (batch_index,
            loss(chitchat, queries_sequences, responses_sequences).numpy())

    checkpoint.save('./checkpoints')

训练时，可以通过输出了解模型的loss::

    step 0: loss 2.019347
    step 10: loss 1.798050
    step 20: loss 1.87050
    step 30: loss 1.758132
    step 40: loss 1.821826

模型训练完成后，我们通过 `checkpoint.save()` 函数将模型的参数存在 `./checkpoints` 目录中。最后，我们需要一个用来对话的程序，来测试实际效果。我们来给 ChitChat 增加 predict 方法：

.. code-block:: python

    class ChitChat(tf.keras.Model):
        # ... append the following code to previous code
        def predict(self, inputs):
            inputs = np.expand_dims(inputs, 0)
            outputs = self(inputs)
            outputs = tf.squeeze(outputs)
            response_indices = []
            for t in range(0, MAX_LEN):
                output = outputs[t]
                indice = tf.argmax(inputs).numpy()
                if indice == self.indice_eos:
                    break
                response_indices.append(indice)
            return response_indices

然后，我们就可以实现一个简单的 Chat 程序。具体实现如下：

.. code-block:: python

    data_loader = DataLoader()
    vocabulary = Vocabulary(data_loader.raw_text)

    chitchat = ChitChat(vocabulary)
    checkpoint = tf.train.Checkpoint(model=chitchat)
    checkpoint.restore(tf.train.latest_checkpoint('./checkpoints'))

    index_word = vocabulary.tokenizer.index_word
    word_index = vocabulary.tokenizer.word_index

    while True:
        try:
            query = input('> ').lower()
            if query == 'q' or query == 'quit':
                break
            query = data_loader.preprocess(query)

            query_sequence = vocabulary.texts_to_padded_sequences([query])[0]
            response_sequence = chitchat.predict(query_sequence)

            response_word_list = [
                index_word[indice]
                for indice in response_sequence
                if indice != 0 and indice != word_index[END_TOKEN]
            ]

            print('Bot:', ' '.join(response_word_list))

        except KeyError:
            print("OOV: Please use simple words with the ChitChat Bot!")

最终生成的对话的界面将会是这样子的::

    > how are you ?
    Bot: fine .
    > where are you ?
    Bot: i don t know .

