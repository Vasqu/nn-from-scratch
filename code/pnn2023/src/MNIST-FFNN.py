from src.LoadDataMNIST import MNIST_DATA_FNN
from src.network.Network import Network
from src.network.SGDTrainer import SGDTrainer
from src.layer.ActivationLayer import ActivationLayer
from src.layer.FullyConnectedLayer import FullyConnectedLayer
from src.layer.LossLayer import CrossEntropyLoss
from src.layer.SoftmaxLayer import SoftmaxLayer


def create_net():
    # network fc - relu - fc - softmax - cross entropy
    net = Network()
    # images are 28x28 -> flattened into 784
    net.addLayer(FullyConnectedLayer(784, 100))
    net.addLayer(ActivationLayer('relu'))
    net.addLayer(FullyConnectedLayer(100, 10))
    net.addLayer(SoftmaxLayer())
    net.addLayer(CrossEntropyLoss())
    return net


# running through some configs to see some possible results
if __name__ == '__main__':
    train, dev, eval = MNIST_DATA_FNN()

    #lr = [0.001, 0.01, 0.1]
    lr = [0.01]
    #epochs = [1, 2, 3]
    epochs = [2]
    #iterations = 5
    iterations = 1

    #start = time.time()
    for l in lr:
        for e in epochs:
            for _ in range(iterations):
                sgd = SGDTrainer(learningRate=l, amountEpochs=e)
                net = create_net()
                eval_acc = sgd.optimize(net, train, dev, eval)
                print(f'Evaluation accuracy: {eval_acc}')
                print()
        print()
    #end = time.time()
    # ~60-70s (few seconds eval)
    #print(f'Elapsed time: {end-start}')
