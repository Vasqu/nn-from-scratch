import pickle

from src.LoadDataMNIST import MNIST_DATA_CNN
from src.network.Network import Network
from src.network.SGDTrainer import SGDTrainer
from src.layer.ActivationLayer import ActivationLayer
from src.layer.FullyConnectedLayer import FullyConnectedLayer
from src.layer.LossLayer import CrossEntropyLoss
from src.layer.SoftmaxLayer import SoftmaxLayer
from src.layer.Conv2DLayer import Conv2DLayer
from src.layer.Pooling2DLayer import AvgPooling, MaxPooling
from src.layer.FlattenLayer import FlattenLayer


def create_net():
    # 4 times 9x9 filter - relu - avg pooling on 4x4 filter with stride 4 - flatten - fc - softmax - cross entropy
    net = Network()
    net.addLayer(Conv2DLayer(9, 9, 1, 4))
    net.addLayer(ActivationLayer('relu'))
    net.addLayer(AvgPooling(filter_shape=(4, 4), stride=(4, 4)))
    net.addLayer(FlattenLayer(input_shape=(4, 5, 5)))
    # 4 x 5 x 5 = 100
    net.addLayer(FullyConnectedLayer(100, 10))
    net.addLayer(SoftmaxLayer())
    net.addLayer(CrossEntropyLoss())
    return net


if __name__ == '__main__':
    train, dev, eval = MNIST_DATA_CNN()

    #start = time.time()

    pickle_run = False
    if pickle_run:
        current_state = 'EVALUATION'
        with open('./picklenets/cnnv1.pkl', 'rb') as f:
            net = pickle.load(f)
        eval_acc = SGDTrainer().evaluate(eval, net, state='EVAL')
        print(f"Evaluation accuracy: {eval_acc}")
    else:
        #lr = [0.001, 0.01, 0.1]
        lr = [0.01]
        #epochs = [1, 2, 3]
        epochs = [1]
        #iterations = 5
        iterations = 1

        nets = []
        save_current_net = False

        for l in lr:
            for e in epochs:
                for _ in range(iterations):
                    sgd = SGDTrainer(learningRate=l, amountEpochs=e)
                    net = create_net()
                    eval_acc = sgd.optimize(net, train, dev, eval)
                    print(f"Evaluation accuracy: {eval_acc}")
                    print()

                    nets.append((net, eval_acc))

        #end = time.time()
        # ~6:50-8:50min total, 20-30s eval only
        #print(f'Elapsed time: {end-start}')

        winner = max(nets, key=lambda x: x[1])[0]
        #"""
        if save_current_net:
            with open('./picklenets/cnnv2.pkl', 'wb+') as f:
                pickle.dump(winner, f) #"""
