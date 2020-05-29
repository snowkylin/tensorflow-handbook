Tensorflow JavaScript 閒聊對話模型
===================================================
 
本章我們將根據前述章節的 Python 版閒聊對話模型，實現一個基於 JavaScript 版的序列到序列模型（Sequence to Sequence, Seq2Seq）。它同樣是基於 RNN 的 Encoder-Decoder 結構，具體基本介紹，請讀者參考 Python 版閒聊對話模型的相關章節。

這里的Encoder-Decoder結構，簡單的來說就是演算法包含兩部分，一個負責對輸入的資訊進行Encoding，將輸入轉換為向量形式；然後由Decoder對這個向量進行解碼，還原為輸出序列。

這個任務預測的是通過一個序列，來預測另外一個對應的序列。舉例來說，常見的打招呼就是一個序列到序列的過程::

    輸入：How are you ?
    輸出：Fine, thank you .

這個過程的輸入序列有4個 token： ``['how', 'are', 'you', '?']`` ，輸出序列有5個 token： ``['fine', ',', 'thank', 'you', '.']`` 。我們希望建立這樣的模型，輸入長為 ``maxLength`` 的序列，輸入變數形狀為 ``[null, max_length]`` ，輸出與這個序列對應的序列中 token 的機率分佈，機率分佈的維度為詞匯表大小 ``vocSize`` ，輸出變數形狀為 ``[null, maxLength, vocSize]`` 。

首先，我們下載預先準備好資料集，將其存為 ``dataset.txt`` 。資料集的格式為每行為一對序列，分別為輸入序列和輸出序列，之間用 ``'\t'`` 製表符隔開。序列中的每一個 token 之間，都通過 ``' '`` 空格符號進行分割。

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


然後，由這些 Layers ，來組建我們的 Seq2Seq 模型。需要註意的是我們需要共用這些 Layers 建立三個不同的模型，分別是：

* 用來訓練的完整 Seq2Seq 模型： ``seq2seqModel`` 
* 用來對序列進行編碼的 Encoder 模型： ``encoderModel`` 
* 用來對序列進行解碼的 Decoder 模型： ``decoderModel`` 

請注意這三個模型中，只有第一個模型  ``seq2seqModel``  是用來訓練參數所需要的，所以訓練的的時候使用這個模型。而另外的兩個模型 ``encoderModel`` 和 ``decoderModel`` ，使我們用來預測的時候需要使用的。這三個模型共用所有的 Layers 參數。

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

然後，為了能夠讓我們使用訓練好的模型，我們還需要基於已經訓練好的模型 Layer 參數，建構獨立的 ``encoderModel`` 和 ``decoderModel`` 。

Encoder子模型輸入 ``numBatch`` 個由編碼後單詞和字元組成的，長為 ``maxLength`` 的序列，輸入變數形狀為 ``[numBatch, maxLength]`` ，輸出與這個序列對應的上下文狀態變數。

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
