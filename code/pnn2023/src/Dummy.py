import numpy as np

from src.layer.ActivationLayer import ActivationLayer
from src.layer.Conv2DLayer import Conv2DLayer
from src.layer.FlattenLayer import FlattenLayer
from src.layer.FullyConnectedLayer import FullyConnectedLayer
from src.layer.InputLayer import InputLayer
from src.layer.LossLayer import MeanSquaredErrorLoss, CrossEntropyLoss
from src.layer.Pooling2DLayer import AvgPooling
from src.layer.SoftmaxLayer import SoftmaxLayer
from src.model.Tensor import Tensor
from src.network.Network import Network
from src.network.SGDTrainer import SGDTrainer


if __name__ == '__main__':
    ffnn = True

    if ffnn:
        # - FFNN -
        # high learning rate, thus small epochs
        sgd = SGDTrainer(learningRate=1, amountEpochs=3, debug=True, loss=False)

        net = Network(debug=True)
        net.addLayer(FullyConnectedLayer(3, 3))
        net.addLayer(ActivationLayer('sigmoid'))
        net.addLayer(FullyConnectedLayer(3, 2))
        net.addLayer(SoftmaxLayer())
        net.addLayer(CrossEntropyLoss())

        data = [([0.4183, 0.5209, 0.0291], [0.7095, 0.0942])]

        sgd.optimize(net, data, [], [])
    else:
        # initialises the standard gradient descent trainer with certain
        # hyperparameters and the loss flag
        # --> forces print statements during the network optimisation
        sgd = SGDTrainer(learningRate=0.01, amountEpochs=1, debug=True, loss=False)


        # initialising an empty network and adding consecutive layers
        # one after another
        net = Network()
        # convolutional layer of 9 rows, 9 columns, and 1 channel
        # while using 4 filters
        net.addLayer(Conv2DLayer(9, 9, 1, 4))
        # activation layer relu by passing the according string
        net.addLayer(ActivationLayer('relu'))
        # pooling layer that uses averaging as aggregation mechanism
        # filter of size 4x4 and a stride of 4x4
        net.addLayer(AvgPooling(filter_shape=(4, 4), stride=(4, 4)))
        # flattening layer to transform the result into 1d
        net.addLayer(FlattenLayer(input_shape=(4, 5, 5)))
        # fully connected layer of 100 rows and 10 columns
        net.addLayer(FullyConnectedLayer(100, 10))
        # softmax layer
        net.addLayer(SoftmaxLayer())
        # cross entropy loss
        net.addLayer(CrossEntropyLoss())


        # the data the net will be trained on with sgd
        # only consists of one pair (input, labels)
        data = [(np.array([[[0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.32941177, 0.72549021, 0.62352943, 0.59215689,
                             0.23529412, 0.14117648, 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.87058824, 0.99607843, 0.99607843, 0.99607843,
                             0.99607843, 0.94509804, 0.7764706 , 0.7764706 , 0.7764706 ,
                             0.7764706 , 0.7764706 , 0.7764706 , 0.7764706 , 0.7764706 ,
                             0.66666669, 0.20392157, 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.26274511, 0.44705883, 0.28235295, 0.44705883,
                             0.63921571, 0.89019608, 0.99607843, 0.88235295, 0.99607843,
                             0.99607843, 0.99607843, 0.98039216, 0.89803922, 0.99607843,
                             0.99607843, 0.54901963, 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.06666667, 0.25882354, 0.05490196, 0.26274511,
                             0.26274511, 0.26274511, 0.23137255, 0.08235294, 0.9254902 ,
                             0.99607843, 0.41568628, 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.32549021, 0.99215686,
                             0.81960785, 0.07058824, 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.08627451, 0.9137255 , 1.        ,
                             0.32549021, 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.50588238, 0.99607843, 0.93333334,
                             0.17254902, 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.23137255, 0.97647059, 0.99607843, 0.24313726,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.52156866, 0.99607843, 0.73333335, 0.01960784,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.03529412, 0.80392158, 0.97254902, 0.22745098, 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.49411765, 0.99607843, 0.71372551, 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.29411766,
                             0.98431373, 0.94117647, 0.22352941, 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.07450981, 0.86666667,
                             0.99607843, 0.65098041, 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.01176471, 0.79607844, 0.99607843,
                             0.85882354, 0.13725491, 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.14901961, 0.99607843, 0.99607843,
                             0.3019608 , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.12156863, 0.87843138, 0.99607843, 0.4509804 ,
                             0.00392157, 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.52156866, 0.99607843, 0.99607843, 0.20392157,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.23921569, 0.94901961, 0.99607843, 0.99607843, 0.20392157,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.47450981, 0.99607843, 0.99607843, 0.85882354, 0.15686275,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.47450981, 0.99607843, 0.81176472, 0.07058824, 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        , 0.        , 0.        ,
                             0.        , 0.        , 0.        ]]])
                 , [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])]


        # trains the net with the sgd trainer
        sgd.optimize(net, data, [], [])