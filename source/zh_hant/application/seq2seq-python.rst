Tensorflow Python 閒聊機器人
===================================================

在本章，我們實現一個可以用來閒聊的對話模型。這個對話模型將基於序列到序列（Seq2Seq）來對電影台詞中的對話數據進行訓練。

序列到序列模型（Sequence to Sequence, SEQ2SEQ）是一種基於 RNN 的 Encoder-Decoder 結構，它也是現在谷歌應用於線上機器翻譯的算法，翻譯質量已經和人類水平不相上下。

這裡通過 Keras 自定義模型建立一個閒聊對話模型（Seq2Seq）。它使用 Encoder-Decoder 結構，簡單的來說就是算法包含兩部分，一個負責對輸入的信息進行 Encoding，將輸入轉換爲向量形式；然後由 Decoder 對這個向量進行解碼，還原爲輸出序列。

關於 Seq2Seq 的原理和介紹，可以參考 Keras 的博客：A ten-minute introduction to sequence-to-sequence learning in Keras。地址： https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

這裡，我們使用 Seq2Seq 來實現一個閒聊（ChitChat）對話機器人。除了閒聊任務（輸入一句話，輸出一句回復）之外，它也可以被直接應用於解決其他類似問題，比如：翻譯（輸入一句英文，輸出一句中文）、摘要（輸入一篇文章，輸出一份總結）、作詩（輸入幾個關鍵字，輸出一首短詩）、對對聯（輸入上聯，輸出下聯），等等。

這個任務對比與之前的RNN尼採風格文本生成，區別在於我們預測的不再是文本的連續字母概率分布，而是通過一個序列，來預測另外一個對應的完整序列。舉例來說，針對一句常見的打招呼::

    How are you


這個句子（序列）一共有3個單詞。當我們聽到這個由3個單詞組成的句子後，根據我們的習慣，我們最傾向與回復的一句話是 "Fine thank you"。我們希望建立這樣的模型，輸入 num_batch 個由編碼後單詞和字符組成的，長爲 max_length 的序列，輸入張量形狀爲 [num_batch, max_length]，輸出與這個序列對應的序列（如聊天回復、翻譯等）中單詞和字符的概率分布，概率分布的維度爲詞彙表大小 voc_size，輸出張量形狀爲 [num_batch, max_length, voc_size]。

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

其次，我們還需要基於 `DataLoader` 加載的文本數據，建立一個詞彙表 `Vocabulary` 來負責管理以下5項任務：

1. 將所有單詞和標點符號進行編碼；
2. 記錄詞彙表大小；
3. 建立單詞到編碼數字，以及編碼數字到單詞的映射字典；
4. 負責將文本句子轉化爲填充後的編碼序列，形狀爲[batch_size, max_length]；

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

ChitEncoder 子模型輸入 num_batch 個由編碼後單詞和字符組成的，長爲 max_length 的序列，輸入張量形狀爲 [num_batch, max_length]，輸出與這個序列對應的上下文張量。爲了簡化代碼，我們這裡只使用一個最基本的 GRU 單元，沒有使用可以獲得更佳效果的雙向RNN、注意力機制等方法。在 `__init__` 方法中我們實例化一個常用的 `GRU` 單元，並將其設置爲 `return_state=True` 來獲得最終的狀態輸出，我們首先對序列進行 GRU 操作，即將編碼序列變換爲 GRU 最終輸出的狀態 ，並將其作爲代表編碼序列的上下文信息 `context` ，作爲模型的輸出。

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

ChatDecoder 子模型輸入 num_batch 個編碼後的一個單詞或字符的 Embedding，和當前的上下文信息張量 `initial_state` 兩個信息構成，輸入張量形狀分別爲 [num_batch, 1, EMBEDDING_DIM]，和 [num_batch, RNN_UNIT_NUM]。在 `__init__` 方法中我們保存詞彙表容量 `voc_size` ，實例化一個常用的 `GRU` 單元，並將其設置爲輸出單元狀態 `return_state=True` 來獲得 GRU 的狀態輸出，以及一個全連接層 `Dense` 單元，負責將 GRU 的輸出變換爲最終的單詞字符分布概率，並將其作爲這個上下文信息對應的單詞符號序列概率分布張量，作爲模型的輸出，形狀爲[num_batch, 1, voc_size]。

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

構建 ChitChat 模型將基於上面的兩個 ChitEncoder 和 ChatDecoder 子模型。在 `__init__` 方法中我們將 `Vocabulary` 中的詞彙到編碼字典 `word_index` 和編碼到詞彙字典 `index_word` ，以及詞彙量 `voc_size` 保存備用，實例化一個詞向量的 `Embedding` 單元，以及一個 `ChitEncoder` 子模型和對應的 `ChatDecoder` 子模型。`ChatDecoder` 子模型中需要使用詞彙表尺寸，我們通過構造參數傳給它。

模型的工作流程爲：我們首先對輸入序列通過 `Embedding` 層進行詞向量轉換，然後進行 Encoder 操作，即將編碼序列 `inputs` 的詞嵌入向量，變換爲一個上下文向量 `encoder_hidden_state` 。然後，我們進入解碼流程：將 START_TOKEN 詞向量和 `encoder_hidden_state` 作爲解碼器的首次輸入，解碼得到解碼器的輸出編碼張量 `decoder_outputs`，以及狀態張量 `decoder_state`。接下來將 `decoder_outputs` 和 `decoder_state` 重複輸入解碼器，即可不斷得到新的 `decoder_outputs` 即作爲模型的輸出，直到 `decoder_outputs` 解碼出來的字符爲 END_TOKEN 爲止。最終輸出的張量形狀爲[num_batch, max_length, voc_size]。

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

訓練過程與本書的 RNN 模型訓練基本一致，在此複述：

- 從DataLoader中隨機取一批訓練數據；
- 將這批數據送入模型，計算出模型的預測值；
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

訓練時，可以通過輸出了解模型的loss::

    step 0: loss 2.019347
    step 10: loss 1.798050
    step 20: loss 1.87050
    step 30: loss 1.758132
    step 40: loss 1.821826

模型訓練完成後，我們通過 `checkpoint.save()` 函數將模型的參數存在 `./checkpoints` 目錄中。最後，我們需要一個用來對話的程序，來測試實際效果。我們來給 ChitChat 增加 predict 方法：

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

然後，我們就可以實現一個簡單的 Chat 程序。具體實現如下：

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

