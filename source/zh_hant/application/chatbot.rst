Tensorflow Seq2Seq 閒聊機器人（Huan）
===================================================

Tensorflow Python 閒聊機器人
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在本章，我們實現一個可以用來閒聊的對話模型。這個對話模型將基於序列到序列（Seq2Seq）來對電影台詞中的對話資料進行訓練。

序列到序列模型（Sequence to Sequence, SEQ2SEQ）是一種基於 RNN 的 Encoder-Decoder 結構，它也是現在谷歌應用於線上機器翻譯的演算法，翻譯質量已經和人類水平不相上下。

這里通過 Keras 自定義模型建立一個閒聊對話模型（Seq2Seq）。它使用 Encoder-Decoder 結構，簡單的來說就是演算法包含兩部分，一個負責對輸入的信息進行 Encoding，將輸入轉換為向量形式；然後由 Decoder 對這個向量進行解碼，還原為輸出序列。

關於 Seq2Seq 的原理和介紹，可以參考 Keras 的博客：A ten-minute introduction to sequence-to-sequence learning in Keras。地址： https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

這里，我們使用 Seq2Seq 來實現一個閒聊（ChitChat）對話機器人。除了閒聊任務（輸入一句話，輸出一句回復）之外，它也可以被直接應用於解決其他類似問題，比如：翻譯（輸入一句英文，輸出一句中文）、摘要（輸入一篇文章，輸出一份總結）、作詩（輸入幾個關鍵字，輸出一首短詩）、對對聯（輸入上聯，輸出下聯），等等。

這個任務對比與之前的RNN尼採風格文本生成，區別在於我們預測的不再是文本的連續字母機率分佈，而是通過一個序列，來預測另外一個對應的完整序列。舉例來說，針對一句常見的打招呼::

    How are you


這個句子（序列）一共有3個單詞。當我們聽到這個由3個單詞組成的句子後，根據我們的習慣，我們最傾向與回復的一句話是 "Fine thank you"。我們希望建立這樣的模型，輸入 num_batch 個由編碼後單詞和字元組成的，長為 max_length 的序列，輸入張量形狀為 [num_batch, max_length]，輸出與這個序列對應的序列（如聊天回復、翻譯等）中單詞和字元的機率分佈，機率分佈的維度為詞匯表大小 voc_size，輸出張量形狀為 [num_batch, max_length, voc_size]。

首先，還是實現一個簡單的 ``DataLoader`` 類來讀取文本，

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

其次，我們還需要基於 `DataLoader` 加載的文本資料，建立一個詞匯表 `Vocabulary` 來負責管理以下5項任務：

1. 將所有單詞和標點符號進行編碼；
2. 記錄詞匯表大小；
3. 建立單詞到編碼數字，以及編碼數字到單詞的映射字典；
4. 負責將文本句子轉化為填充後的編碼序列，形狀為[batch_size, max_length]；

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

接下來進行模型的實現。我們建立一個ChitChat模型。ChitChat 模型是一個 Seq2Seq 的模型，主要由 ChitEncoder 和 ChatDecoder 組成。

ChitEncoder 子模型輸入 num_batch 個由編碼後單詞和字元組成的，長為 max_length 的序列，輸入張量形狀為 [num_batch, max_length]，輸出與這個序列對應的上下文張量。為了簡化代碼，我們這里只使用一個最基本的 GRU 單元，沒有使用可以獲得更佳效果的雙向RNN、註意力機制等方法。在 `__init__` 方法中我們實例化一個常用的 `GRU` 單元，並將其設置為 `return_state=True` 來獲得最終的狀態輸出，我們首先對序列進行 GRU 操作，即將編碼序列變換為 GRU 最終輸出的狀態 ，並將其作為代表編碼序列的上下文信息 `context` ，作為模型的輸出。

`ChitEncoder` 子模型具體實現如下：

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

ChatDecoder 子模型輸入 num_batch 個編碼後的一個單詞或字元的 Embedding，和當前的上下文信息張量 `initial_state` 兩個信息構成，輸入張量形狀分別為 [num_batch, 1, EMBEDDING_DIM]，和 [num_batch, RNN_UNIT_NUM]。在 `__init__` 方法中我們保存詞匯表容量 `voc_size` ，實例化一個常用的 `GRU` 單元，並將其設置為輸出單元狀態 `return_state=True` 來獲得 GRU 的狀態輸出，以及一個全連接層 `Dense` 單元，負責將 GRU 的輸出變換為最終的單詞字元分佈機率，並將其作為這個上下文信息對應的單詞符號序列機率分佈張量，作為模型的輸出，形狀為[num_batch, 1, voc_size]。

`ChitDecoder` 子模型具體實現如下：

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

構建 ChitChat 模型將基於上面的兩個 ChitEncoder 和 ChatDecoder 子模型。在 `__init__` 方法中我們將 `Vocabulary` 中的詞匯到編碼字典 `word_index` 和編碼到詞匯字典 `index_word` ，以及詞匯量 `voc_size` 保存備用，實例化一個詞向量的 `Embedding` 單元，以及一個 `ChitEncoder` 子模型和對應的 `ChatDecoder` 子模型。`ChatDecoder` 子模型中需要使用詞匯表尺寸，我們通過構造參數傳給它。

模型的工作流程為：我們首先對輸入序列通過 `Embedding` 層進行詞向量轉換，然後進行 Encoder 操作，即將編碼序列 `inputs` 的詞嵌入向量，變換為一個上下文向量 `encoder_hidden_state` 。然後，我們進入解碼流程：將 START_TOKEN 詞向量和 `encoder_hidden_state` 作為解碼器的首次輸入，解碼得到解碼器的輸出編碼張量 `decoder_outputs`，以及狀態張量 `decoder_state`。接下來將 `decoder_outputs` 和 `decoder_state` 重復輸入解碼器，即可不斷得到新的 `decoder_outputs` 即作為模型的輸出，直到 `decoder_outputs` 解碼出來的字元為 END_TOKEN 為止。最終輸出的張量形狀為[num_batch, max_length, voc_size]。

`ChitChat` 模型具體實現如下：

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

訓練過程與本書的 RNN 模型訓練基本一致，在此復述：

- 從DataLoader中隨機取一批訓練資料；
- 將這批資料送入模型，計算出模型的預測值；
- 將模型預測值與真實值進行比較，計算損失函數（loss）；
- 計算損失函數關於模型變量的導數；
- 使用優化器更新模型參數以最小化損失函數。

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

訓練時，可以透過輸出了解模型的loss::

    step 0: loss 2.019347
    step 10: loss 1.798050
    step 20: loss 1.87050
    step 30: loss 1.758132
    step 40: loss 1.821826

模型訓練完成後，我們通過 `checkpoint.save()` 函數將模型的參數存在 `./checkpoints` 目錄中。最後，我們需要一個用來對話的程式，來測試實際效果。我們來給 ChitChat 增加 predict 方法：

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

然後，我們就可以實現一個簡單的 Chat 程式。具體實現如下：

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

最終生成的對話的界面將會是這樣子的::

    > how are you ?
    Bot: fine .
    > where are you ?
    Bot: i don t know .

Tensorflow JavaScript 閒聊對話模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
本章我們將根據前述章節的 Python 版閒聊對話模型，實現一個基於 JavaScript 版的序列到序列模型（Sequence to Sequence, Seq2Seq）。它同樣是基於 RNN 的 Encoder-Decoder 結構，具體基本介紹，請讀者參考 Python 版閒聊對話模型的相關章節。

這里的Encoder-Decoder結構，簡單的來說就是演算法包含兩部分，一個負責對輸入的信息進行Encoding，將輸入轉換為向量形式；然後由Decoder對這個向量進行解碼，還原為輸出序列。

這個任務預測的是通過一個序列，來預測另外一個對應的序列。舉例來說，常見的打招呼就是一個序列到序列的過程::

    輸入：How are you ?
    輸出：Fine, thank you .

這個過程的輸入序列有4個 token： ``['how', 'are', 'you', '?']`` ，輸出序列有5個 token： ``['fine', ',', 'thank', 'you', '.']`` 。我們希望建立這樣的模型，輸入長為 ``maxLength`` 的序列，輸入變數形狀為 ``[null, max_length]`` ，輸出與這個序列對應的序列中 token 的機率分佈，機率分佈的維度為詞匯表大小 ``vocSize`` ，輸出變數形狀為 ``[null, maxLength, vocSize]`` 。

首先，我們下載預先準備好資料集，將其存為 ``dataset.txt`` 。資料集的格式為每行為一對序列，分別為輸入序列和輸出序列，之間用 ``'\t'`` 製表符隔開。序列中的每一個 token 之間，都透過 ``' '`` 空格符號進行分割。

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

我們需要將它轉換為 Tensorflow Dataset 格式：

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

其次，我們還需要基於 ``Dataset`` 中輸入序列和輸出序列中的文本資料，建立對應的詞匯表 ``Vocabulary`` 來負責管理以下5項任務：

1. 將所有單詞和標點符號進行編碼；
2. 記錄詞匯表大小；
3. 建立單詞到編碼數字，以及編碼數字到單詞的對應字典；

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

接下來，我們將資料集和 ``Vocabulary`` 結合起來，並對資料集進行資料向量化。

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

    // 額外的 START_TOKEN 和 END_TOKEN
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

接下來進行模型的實現。我們先建立 Seq2Seq 模型所需的所有 Layers，具體實現如下：

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


然後，由這些 Layers ，來建立我們的 Seq2Seq 模型。需要註意的是我們需要共用這些 Layers 建立三個不同的模型，分別是：

* 用來訓練的完整 Seq2Seq 模型： ``seq2seqModel`` 
* 用來對序列進行編碼的 Encoder 模型： ``encoderModel`` 
* 用來對序列進行解碼的 Decoder 模型： ``decoderModel`` 

請註意這三個模型中，只有第一個模型  ``seq2seqModel``  是用來訓練參數所需要的，所以訓練的的時候使用這個模型。而另外的兩個模型 ``encoderModel`` 和 ``decoderModel`` ，使我們用來預測的時候需要使用的。這三個模型共用所有的 Layers 參數。

``seq2seqModel`` 模型的輸入包含兩個，一個是 Encoder 的輸入，另外一個是 Decoder 的輸入。模型的輸出是我們資料集的輸出。

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

用來訓練的 ``seq2seqModel`` 模型建立完畢後，即可基於模型的 ``fitDataset`` 函數進行訓練：

.. code-block:: javascript
    await seq2seqModel.fitDataset(
      seq2seqDataset
      .take(10000)
      .batch(64)
      {
        epochs: 100,
      },
    )

訓練大約需要幾個小時的時間，才能達到比較好的效果。

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

然後，為了能夠讓我們使用訓練好的模型，我們還需要基於已經訓練好的模型 Layer 參數，構建獨立的 ``encoderModel`` 和 ``decoderModel`` 。

Encoder子模型輸入 ``numBatch`` 個由編碼後單詞和字元組成的，長為 ``maxLength`` 的序列，輸入張量形狀為 ``[numBatch, maxLength]`` ，輸出與這個序列對應的上下文狀態張量。

``encoderModel`` 的代碼實現如下：

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

``deocoderModel`` 的輸入有兩個，分別是 t 時間的 token indice，和對應的解碼器 ``state``；輸出也有兩個，分別是 t+1 時間的 token 的 voc 分佈機率，和對應的解碼器 ``state`` ：

``decoderModel`` 子模型具體實現如下：

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

最後，我們需要一個用來對話的程式。我們建立一個專門用來接收一句話輸入，然後通過我們的模型預測，得到序列輸出的函數 ``seq2seqDecoder()`` ：

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

最後，我們就可以用我們訓練好的Seq2Seq模型，實現我們的 ChitChat 聊天功能了：

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

模型每次的訓練，得到的結果都會不盡相同。作者的某一次輸出的內容是下面這樣的：

::

    Input sentence： "how are you ?"
    Decoded setence: "good ."


註：本章節中的 JavaScript 版 ChitChat 完整程式碼，使用說明，和訓練好的模型文件及參數，都可以在作者的 GitHub 上找到。地址： https://github.com/huan/tensorflow-handbook-javascript

TensorFlow Swift 閒聊機器人
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果時間來得及，完成 Seq2Seq 模型。
