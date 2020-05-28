Tensorflow Seq2Seq 闲聊机器人（Huan）
===================================================

Tensorflow Python 闲聊机器人
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Tensorflow JavaScript 闲聊对话模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
本章我们将根据前述章节的 Python 版闲聊对话模型，实现一个基于 JavaScript 版的序列到序列模型（Sequence to Sequence, Seq2Seq）。它同样是基于 RNN 的 Encoder-Decoder 结构，具体基本介绍，请读者参考 Python 版闲聊对话模型的相关章节。

这里的Encoder-Decoder结构，简单的来说就是算法包含两部分，一个负责对输入的信息进行Encoding，将输入转换为向量形式；然后由Decoder对这个向量进行解码，还原为输出序列。

这个任务预测的是通过一个序列，来预测另外一个对应的序列。举例来说，常见的打招呼就是一个序列到序列的过程::

    输入：How are you ?
    输出：Fine, thank you .

这个过程的输入序列有4个 token： ``['how', 'are', 'you', '?']`` ，输出序列有5个 token： ``['fine', ',', 'thank', 'you', '.']`` 。我们希望建立这样的模型，输入长为 ``maxLength`` 的序列，输入张量形状为 ``[null, max_length]`` ，输出与这个序列对应的序列中 token 的概率分布，概率分布的维度为词汇表大小 ``vocSize`` ，输出张量形状为 ``[null, maxLength, vocSize]`` 。

首先，我们下载预先准备好数据集，将其存为 ``dataset.txt`` 。数据集的格式为每行为一对序列，分别为输入序列和输出序列，之间用 ``'\t'`` 制表符隔开。序列中的每一个 token 之间，都通过 ``' '`` 空格符号进行分割。

::

    $ wget https://github.com/huan/python-concise-chitchat/releases/download/v0.0.1/dataset.txt.gz
    dataset.txt.gz 100% [======================>] 986.60K   282KB/s    in 3.5s

    2019-03-15 22:59:00 (282 KB/s) - ‘dataset.txt.gz’ saved [1010276/1010276]

    $ gzip -d dataset.txt.gz

    $ ls -l dataset.txt
    l-rw-r--r--  1 zixia  wheel  3516695 Mar 14 13:15 dataset.txt

    $ head -3 dataset.txt 
    did you change your hair ?	no .
    no .	you might wanna think about it
    you the new guy ?	so they tell me ...

我们需要将它转换为 Tensorflow Dataset 格式：

.. code-block:: javascript

    import * as tf from '@tensorflow/tfjs'

    type Seq2seqData = {
      input: string,
      output: string,
    }

    const dataset = tf.data.csv('dataset.txt', {
        hasHeader: false,
        columnNames: ['input', 'output'],
        delimiter: '\t',
    }) as any as tf.data.Dataset<Seq2seqData>

其次，我们还需要基于 ``Dataset`` 中输入序列和输出序列中的文本数据，建立对应的词汇表 ``Vocabulary`` 来负责管理以下5项任务：

1. 将所有单词和标点符号进行编码；
2. 记录词汇表大小；
3. 建立单词到编码数字，以及编码数字到单词的映射字典；

.. code-block:: javascript

    class Vocabulary {
      private readonly tokenIndice: Map<string, number>
      private readonly indiceToken: Map<number, string>

      public maxSeqLength: number
      public size: number

      constructor () {
        this.tokenIndice = new Map<string, number>()
        this.indiceToken = new Map<number, string>()

        this.size = 1 // Including the reserved 0
        this.maxSeqLength = 0
      }

      public fitToken(token: string): void {
        if (!this.tokenIndice.has(token)) {
          this.tokenIndice.set(token, this.size)
          this.indiceToken.set(this.size, token)
          this.size++
        }
      }

      public fitText(text: string): void {
        const tokenList = [...text.split(/\s+/)]

        if (tokenList.length > this.maxSeqLength) {
          this.maxSeqLength = tokenList.length
        }
        for (const token of tokenList) {
          this.fitToken(token)
        }
      }

      public token(indice: number): string {
        return this.indiceToken.get(indice) as string
      }

      public indice (token: string): number {
        return this.tokenIndice.get(token) as number
      }

      public sequenize (
        text: string,
        length = 0,
      ): number[] {
        const tokenList = [...text.split(/\s+/)]
        const indiceList = tokenList.map(token => this.indice(token))

        if (length === -1) {
          indiceList.length = this.maxSeqLength
          if (this.maxSeqLength > tokenList.length) {
            indiceList.fill(0, tokenList.length)
          }
        }

        return indiceList
      }
    }

接下来，我们将数据集和 ``Vocabulary`` 结合起来，并对数据集进行数据向量化。

.. code-block:: javascript

    export const START_TOKEN = '\t'
    export const END_TOKEN = '\n'

    const voc = new Vocabulary()

    voc.fitToken(START_TOKEN)
    voc.fitToken(END_TOKEN)

    await dataset.forEachAsync(value => {
      voc.fitText(value.input)
      voc.fitText(value.output)
    })

    // 额外的 START_TOKEN 和 END_TOKEN
    voc.maxSeqLength += 2

    const seq2seqDataset = dataset
    .map(value => {
      const input = tf.tensor(voc.sequenize(value.input, -1))

      const decoderInputBuf = tf.buffer<tf.Rank.R1>([
        voc.maxSeqLength,
      ])
      const decoderTargetBuf = tf.buffer<tf.Rank.R2>([
        voc.maxSeqLength,
        voc.size,
      ])

      const outputIndiceList = [
        voc.indice(START_TOKEN),
        ...voc.sequenize(value.output),
        voc.indice(END_TOKEN),
      ]

      for (const [t, indice] of outputIndiceList.entries()) {
        decoeerInputBuf.set(indice, t)

        // shift left for target: not including START_OF_SEQ
        if (t > 0) {
          decoderTargetBuf.set(1, t - 1, indice)
        }
      }

      const decoderInput = decoderInputBuf.toTensor()
      const decoderTarget = decoderTargetBuf.toTensor()

      const xs = {
        seq2seqInputs: inputTensor,
        seq2seqDecoderInputs: decoderInput,
      }
      const ys = decoderTarget

      return {xs, ys}
    })

接下来进行模型的实现。我们先建立 Seq2Seq 模型所需的所有 Layers，具体实现如下：

.. code-block:: javascript

    /**
     * Encoder Layers
     */
    const encoderEmbeddingLayer = tf.layers.embedding({
      inputDim: voc.size,
      outputDim: latentDim,
    })

    const encoderRnnLayer = tf.layers.gru({
      units: latentDim,
      returnState: true,
    })

    /**
     * Decoder Layers
     */
    const decoderEmbeddingLayer = tf.layers.embedding({
      inputDim: voc.size,
      outputDim: latentDim,
    })

    const decoderRnnLayer = tf.layers.gru({
      units: latentDim,
      returnSequences: true,
      returnState: true,
    })

    const decoderDenseLayer = tf.layers.dense({
        units: voc.size,
        activation: 'softmax',
    })


然后，由这些 Layers ，来组建我们的 Seq2Seq 模型。需要注意的是我们需要共享这些 Layers 建立三个不同的模型，分别是：

* 用来训练的完整 Seq2Seq 模型： ``seq2seqModel`` 
* 用来对序列进行编码的 Encoder 模型： ``encoderModel`` 
* 用来对序列进行解码的 Decoder 模型： ``decoderModel`` 

请注意这三个模型中，只有第一个模型  ``seq2seqModel``  是用来训练参数所需要的，所以训练的的时候使用这个模型。而另外的两个模型 ``encoderModel`` 和 ``decoderModel`` ，使我们用来预测的时候需要使用的。这三个模型共享所有的 Layers 参数。

``seq2seqModel`` 模型的输入包含两个，一个是 Encoder 的输入，另外一个是 Decoder 的输入。模型的输出是我们数据集的输出。

.. code-block:: javascript

    const inputs = tf.layers.input({
      shape: [null],
      name: 'seq2seqInputs',
    })

    const encoderEmbedding = encoderEmbeddingLayer.apply(inputs) as tf.Tensor<tf.Rank.R3>

    const [, encoderState] = encoderRnnLayer.apply(encoderEmbedding) as tf.SymbolicTensor[]

    const decoderInputs = tf.layers.input({
      shape: [voc.maxSeqLength],
      name: 'seq2seqDecoderInputs',
    })

    const decoderEmbedding = decoderEmbeddingLayer.apply(decoderInputs) as tf.SymbolicTensor

    const [decoderOutputs,] = decoderRnnLayer.apply(
      [decoderEmbedding, encoderState],
      {
        returnSequences: true,
        returnState: true,
      },
    ) as tf.SymbolicTensor[]

    const decoderTargets = decoderDenseLayer.apply(decoderOutputs) as tf.SymbolicTensor

    const seq2seqModel = tf.model({
      inputs: [inputs, decoderInputs],
      outputs: decoderTargets,
      name: 'seq2seqModel',
    })

用来训练的 ``seq2seqModel`` 模型建立完毕后，即可基于模型的 ``fitDataset`` 函数进行训练：

.. code-block:: javascript
    await seq2seqModel.fitDataset(
      seq2seqDataset
      .take(10000)
      .batch(64)
      {
        epochs: 100,
      },
    )

训练大约需要几个小时的时间，才能达到比较好的效果。

::

    Epoch 1 / 20
    eta=0.0 > 
    90436ms 576025us/step - loss=4.82 
    Epoch 2 / 20
    eta=0.0 > 
    85229ms 542858us/step - loss=4.07 
    Epoch 3 / 20
    eta=0.0 > 
    81913ms 521742us/step - loss=3.77 
    Epoch 4 / 20
    eta=0.0 - loss=3.52 
    ...

然后，为了能够让我们使用训练好的模型，我们还需要基于已经训练好的模型 Layer 参数，构建独立的 ``encoderModel`` 和 ``decoderModel`` 。

Encoder子模型输入 ``numBatch`` 个由编码后单词和字符组成的，长为 ``maxLength`` 的序列，输入张量形状为 ``[numBatch, maxLength]`` ，输出与这个序列对应的上下文状态张量。

``encoderModel`` 的代码实现如下：

.. code-block:: javascript

    const encoderInputs = tf.layers.input({
      shape: [null],
      name: 'encoderInputs',
    })
    const encoderEmbedding = encoderEmbeddingLayer.apply(encoderInputs)
    const [, encoderState] = encoderRnnLayer.apply(encoderEmbedding) as tf.SymbolicTensor[]

    const encoderModel = tf.model({
      inputs: encoderInputs,
      outputs: encoderState,
    })

``deocoderModel`` 的输入有两个，分别是 t 时刻的 token indice，和对应的解码器 ``state``；输出也有两个，分别是 t+1 时刻的 token 的 voc 分布概率，和对应的解码器 ``state`` ：

``decoderModel`` 子模型具体实现如下：

.. code-block:: javascript

    const decoderInput = tf.layers.input({
      shape: [1],
      name: 'decoderInputs',
    })
    const decoderStateInput = tf.layers.input({
      shape: [latentDim],
      name: 'decoderState',
    }) as tf.SymbolicTensor

    const decoderEmbedding = decoderEmbeddingLayer.apply(decoderInput) as tf.SymbolicTensor

    const [decoderOutputs, decoderStateOutput] = decoderRnnLayer.apply(
      [decoderEmbedding, decoderStateInput],
      {
        returnState: true,
      },
    ) as tf.SymbolicTensor[]
    const decoderDenseOutputs = decoderDenseLayer.apply(decoderOutputs) as tf.SymbolicTensor

    const decoderModel = tf.model({
      inputs: [decoderInput, decoderStateInput],
      outputs: [decoderDenseOutputs, decoderStateOutput],
    })

最后，我们需要一个用来对话的程序。我们建立一个专门用来接收一句话输入，然后通过我们的模型预测，得到序列输出的函数 ``seq2seqDecoder()`` ：

.. code-block:: javascript

    export async function seq2seqDecoder (
      input: string,
      encoderModel: tf.LayersModel,
      decoderModel: tf.LayersModel,
      voc: Vocabulary,
    ): Promise<string> {
      const inputSeq = voc.sequenize(input)
      const inputTensor = tf.tensor(inputSeq)

      const batchedInput = inputTensor.expandDims(0)
      let state = encoderModel.predict(batchedInput) as tf.Tensor<tf.Rank.R2>

      let tokenIndice = voc.indice(START_TOKEN)

      let decoderOutputs: tf.Tensor<tf.Rank.R3>
      let decodedToken: string
      let decodedTokenList = []

      do {
        const decoderInputs = tf.tensor(tokenIndice).reshape([1, 1]) as tf.Tensor<tf.Rank.R2>

        ;[decoderOutputs, state] = decoderModel.predict([
          decoderInputs,
          state,
        ]) as [
          tf.Tensor<tf.Rank.R3>,
          tf.Tensor<tf.Rank.R2>,
        ]

        let decodedIndice = await decoderOutputs
                                    .squeeze()
                                    .argMax()
                                    .array() as number

        if (decodedIndice === 0) {
          // 0 for padding, should be treated as END
          decodedToken = END_TOKEN
        } else {
          decodedToken = voc.token(decodedIndice)
        }

        if (decodedToken === END_TOKEN) {
          break
        } else {
          decodedTokenList.push(decodedToken)
        }

        // save decoded data for next time step
        tokenIndice = decodedIndice

      } while (decodedTokenList.length < voc.maxSeqLength)

      return decodedTokenList.join(' ')
    }

最后，我们就可以用我们训练好的Seq2Seq模型，实现我们的 ChitChat 聊天功能了：

.. code-block:: javascript

    const input = 'how are you ?'

    const decodedOutput = await seq2seqDecoder(
      input,
      encoderModel,
      decoderModel,
      inputVoc,
      outputVoc,
    )

    console.log(`Input sentence: "${input}"`)
    console.log(`Decoded sentence: "${decodedOutput}"`)

模型每次的训练，得到的结果都会不尽相同。作者的某一次输出的内容是下面这样的：

::

    Input sentence： "how are you ?"
    Decoded setence: "good ."


注：本章节中的 JavaScript 版 ChitChat 完整代码，使用说明，和训练好的模型文件及参数，都可以在作者的 GitHub 上找到。地址： https://github.com/huan/tensorflow-handbook-javascript

TensorFlow Swift 闲聊机器人
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果时间来得及，完成 Seq2Seq 模型。
