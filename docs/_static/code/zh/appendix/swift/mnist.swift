import TensorFlow
import Python
import Foundation

import MNIST

struct MLP: Layer {
    // 定义模型的输入、输出数据类型
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    // 定义 flatten 层，将二维矩阵展开为一个一维数组
    var flatten = Flatten<Float>()
    // 定义全连接层，输入为 784 个神经元，输出为 10 个神经元
    var dense = Dense<Float>(inputSize: 784, outputSize: 10)
    
    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        // 模型运行时，将 input 顺序传递给：flatten 和 dense 层，将结果返回作为模型输出。
        return input.sequenced(through: flatten, dense)
    }  
}

var model = MLP()
let optimizer = Adam(for: model)

let mnist = MNIST()
let ((trainImages, trainLabels), (testImages, testLabels)) = mnist.loadData()

let imageBatch = Dataset(elements: trainImages).batched(32)
let labelBatch = Dataset(elements: trainLabels).batched(32)

for (X, y) in zip(imageBatch, labelBatch) {
    // Caculate the gradient
    let grads = gradient(at: model) { model -> Tensor<Float> in
        let logits = model(X)
        return softmaxCrossEntropy(logits: logits, labels: y)
    }

    // Update parameters by optimizer
    optimizer.update(&model.self, along: grads)
}

let logits = model(testImages)
let acc = mnist.getAccuracy(y: testLabels, logits: logits)

print("Test Accuracy: \(acc)" )
