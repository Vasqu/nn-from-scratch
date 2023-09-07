# Neural networks from scratch

## Introduction

This repository implements (restricted) FNNs (fully-connected neural nets) and CNNs (convolutional neural nets) only using numpy at its core.
The purpose behind this project is/was to learn about the smaller details behind neural nets and their functionality. 

This project was accompanied by the lecture ["Programming with neural nets" (SS2023)](https://www.informatik.uni-wuerzburg.de/is/lehre/sommersemester-2023/).


## Usage

An example for a convolutional neural network:
```python
# import necessary classes
from src.LoadDataMNIST import MNIST_DATA_CNN, MNIST_DATA_FNN
from src.network.Network import Network
from src.network.SGDTrainer import SGDTrainer
from src.layer.ActivationLayer import ActivationLayer
from src.layer.FullyConnectedLayer import FullyConnectedLayer
from src.layer.LossLayer import CrossEntropyLoss
from src.layer.SoftmaxLayer import SoftmaxLayer
from src.layer.Conv2DLayer import Conv2DLayer
from src.layer.Pooling2DLayer import AvgPooling
from src.layer.FlattenLayer import FlattenLayer
"""
other possible layers not included in the example net: 
    - MeanSquaredErrorLoss in src.layer.LossLayer
    - MaxPooling in src.layer.Pooling2DLayer
"""


# create an empty neural network with the input layer
net = Network()
# add one layer after the other in order (expects fitting shapes given by the user)
# convolution gets the number of rows, columns, channels and finally filters
net.addLayer(Conv2DLayer(9, 9, 1, 4))
# activation function are initialised by a given set of existing ones (referenced by
# their string): relu, sigmoid, tanh
net.addLayer(ActivationLayer('relu'))
# pooling layers get their filter shape as well as their stride (row, column)
net.addLayer(AvgPooling(filter_shape=(4, 4), stride=(4, 4)))
# the flattening layer expects an input shape of the incoming tensor
net.addLayer(FlattenLayer(input_shape=(4, 5, 5)))
# fully connected gets as parameters the shape of the input and the output
net.addLayer(FullyConnectedLayer(100, 10))
# softmax does not have any parameters
net.addLayer(SoftmaxLayer())
# cross entropy loss also does not have any parameters
net.addLayer(CrossEntropyLoss())


# load the preprocessed MNIST data for the CNN
train, dev, eval = MNIST_DATA_CNN()
# for the FNN likewise
# train, dev, eval = MNIST_DATA_FNN()


"""
initialise stochastic gradient descent trainer which has the following hyperparameters
   - learningRate, i.e. the learning rate for the net
   - amountEpochs, i.e. the total number of iterations through the total dataset
   - debug, i.e. a boolean flag that enables print statements during a run that
     give all states of all layers during the iterations (forward, backward, deltas..)
   - loss, i.e. also a boolean flag that enables a tqdm progress bar showing the avg 
     loss, the current accuracy of a given batch size, and the batch size used
   - loss_batch, i.e. the batch size that is used when the loss flag is set to True
   - dev, i.e. also a boolean flag that enables early endings if the improvement from the last epoch is too little
   - dev_improv, i.e. the improvement that is needed from epoch to epoch to not early break 
     (value is shifted to the right x2, e.g. improvement of 0.01 == 1)
"""
sgd = SGDTrainer(learningRate=0.01, amountEpochs=1)
# run the training via
eval_acc = sgd.optimize(net, train, dev, eval)
print(f"Evaluation accuracy: {eval_acc}")
```

In general, the usage can be examined in the <em>src.MNIST-*.py</em> (see [CNN](./code/pnn2023/src/MNIST-CNN.py) and [FFNN](./code/pnn2023/src/MNIST-FFNN.py)) scripts. Those scripts also include a main function that can be run to test a given net (which can be modified in the *create_net()* function) on MNIST. Otherwise, additional explanation can be found [here](./documents/PNN_FrameworkExplanation.pdf).



## Status

Of course, this implementation is not perfect and hence, here are some bullet points I consider to be improvable:
- Only SGD (stochastic gradient descent) is available as optimizer
- Batches as another dimension are not implemented
- Subsequent input shapes have always to be passed in the initial creation of layers
  - Should not be necessary, just calculate the necessary shapes on initialisation and pass it then and there
- Cross entropy implementation should follow the "torch" standard
  - This implementation uses softmax and a subsequent loss function --> just combine them into one layer
- Deltas are always initialised which might be unwanted especially if we only want to infer with a model
  - Eval flag or something similar?
- Convolutions are implemented in a restricted way
  - No dilation allowed
  - No selection of padding allowed (uses valid padding as standard)
  - Inefficiently implemented as it does not use the [toeplitz](https://en.wikipedia.org/wiki/Toeplitz_matrix) implementation
- Pooling layers assume stride == shape to work properly
- Utility/Normalisation layers are not implemented 
  - E.g. layer normalisation and the sort are useful to keep values "in range"
- RNNs are not implemented
- Automatic differentiation? 
- Probably a lot more I didn't even consider...