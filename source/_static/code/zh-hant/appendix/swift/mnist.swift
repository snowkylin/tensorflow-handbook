import TensorFlow
import Python
import Foundation

import MNIST

struct MLP: Layer {
    // 定義模型的輸入、輸出資料類型
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    // 定義 flatten 層，將二維矩陣展開為一堆陣列
    var flatten = Flatten<Float>()
    // 定義全連接層，輸入為 784 個神經元，輸出為 10 個神經元
    var dense = Dense<Float>(inputSize: 784, outputSize: 10)
    
    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        var x = input
        x = flatten(x)
        x = dense(x)
        return x
    }  
}

var model = MLP()
let optimizer = Adam(for: model)

let mnist = MNIST()
let ((trainImages, trainLabels), (testImages, testLabels)) = mnist.loadData()

let imageBatch = Dataset(elements: trainImages).batched(32)
let labelBatch = Dataset(elements: trainLabels).batched(32)

for (X, y) in zip(imageBatch, labelBatch) {
    // 計算梯度
    let grads = gradient(at: model) { model -> Tensor<Float> in
        let logits = model(X)
        return softmaxCrossEntropy(logits: logits, labels: y)
    }

    // 優化器根據梯度更新模型參數
    optimizer.update(&model.self, along: grads)
}

let logits = model(testImages)
let acc = mnist.getAccuracy(y: testLabels, logits: logits)

print("Test Accuracy: \(acc)" )
