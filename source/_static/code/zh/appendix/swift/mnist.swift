import TensorFlow
import Python
import Foundation

/**
 * The Swift Module for MNIST Dataset:
 * https://github.com/huan/swift-MNIST
 */
import MNIST

struct MLP: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var flatten = Flatten<Float>()
    var dense = Dense<Float>(inputSize: 784, outputSize: 10)
    
    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
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
    let (_, grads) = valueWithGradient(at: model) { model -> Tensor<Float> in
        let logits = model(X)
        return softmaxCrossEntropy(logits: logits, labels: y)
    }

    // Update parameters by optimizer
    optimizer.update(&model.self, along: grads)
}

let logits = model(testImages)
let acc = mnist.getAccuracy(y: testLabels, logits: logits)

print("Test Accuracy: \(acc)" )
