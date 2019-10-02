Tensorflow JavaScript Chitchat Model
====================================
 
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
