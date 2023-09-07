from typing import List
import numpy as np
from src.layer.InputLayer import InputLayer
from src.layer.Conv2DLayer import Conv2DLayer
from src.layer.FullyConnectedLayer import FullyConnectedLayer
from src.layer.ActivationLayer import ActivationLayer
from src.layer.SoftmaxLayer import SoftmaxLayer
from src.layer.LossLayer import CrossEntropyLoss, MeanSquaredErrorLoss
from src.layer.Layer import Layer
from src.model.Tensor import Tensor


class Network:
    input: InputLayer
    layers: List[Layer]
    parameters: List[Tensor]
    deltaParams: List[Tensor]
    cache = List[Tensor]
    debug: bool
    initialised: bool

    def __init__(self, debug=False):
        self.debug = debug
        self.input = InputLayer()
        self.layers = []
        self.parameters = []
        self.deltaParams = []
        self.cache = []
        self.initialised = False

    def addLayer(self, layer: Layer):
        self.layers.append(layer)

    def initTensors(self, first_data):
        # initialise input layer (will always be exchanged with new data)
        inTen = self.input.forward(first_data)
        self.cache.append(inTen)

        # first layer is expected to not be an instant loss layer --> get first output + shape
        first_layer_out_shape = self.layers[0].getOutShape(inTen[0].shape)
        second_layer_tensor = [Tensor(np.zeros(shape=first_layer_out_shape))]
        self.cache.append(second_layer_tensor)

        # iterate through and construct preset tensors according to the layer type
        current_shape = first_layer_out_shape
        for i in range(1, len(self.layers)):
            current_layer = self.layers[i]

            if isinstance(current_layer, ActivationLayer) \
                or isinstance(current_layer, SoftmaxLayer):
                next_layer_tensor = [Tensor(np.zeros(shape=current_shape))]
            elif isinstance(current_layer, CrossEntropyLoss) or isinstance(current_layer, MeanSquaredErrorLoss):
                next_layer_tensor = [Tensor(np.float64(0))]
            else:
                current_layer_out_shape = current_layer.getOutShape(current_shape)
                next_layer_tensor = [Tensor(np.zeros(shape=current_layer_out_shape))]
                current_shape = current_layer_out_shape

            self.cache.append(next_layer_tensor)

    def forward(self, input: List[any], labels: [float]):
        # if not yet initialised --> create all tensor with their according shapes and save them
        if not self.initialised: self.initTensors(input); self.initialised = True

        # setting expected output
        self.layers[-1].setLabels(np.array(labels))
        # expecting an array of floats ish as input
        inTen = self.input.forward(input)
        self.cache[0] = inTen

        if self.debug:
            print('###########################################')
            print()
            print("- Given input and expected output -")
            print()
            print("Input: " + str(self.cache[0][0].elements))
            print("Labels: " + str(self.layers[-1].labels))
            print()
            print('###########################################')
            print()

        global cache_copy
        for j in range(len(self.layers)):
            self.layers[j].forward(self.cache[j], self.cache[j+1])

        if self.debug:
            print("- Forward passing -")
            print()
            for i in range(len(self.cache)-1):
                print("Layer type: " + str(type(self.layers[i])))
                if isinstance(self.layers[i], FullyConnectedLayer):
                    print("Weights: \n" + str(self.layers[i].weightMatrix.elements))
                    print("Bias: \n" + str(self.layers[i].bias.elements))
                elif isinstance(self.layers[i], Conv2DLayer):
                    print("Filters: \n" + str(self.layers[i].filters[0].elements))
                    print("Bias: \n" + str(self.layers[i].filters[1].elements))
                print("Corresponding output: \n" + str(self.cache[i][0].elements))
                print("Corresponding shape: \n" + str(self.cache[i+1][0].shape))
                print()
            print("Loss: " + str(self.cache[-1][0].elements))
            print()
            print('###########################################')
            print()

        return self.cache[-1]

    def backprop(self, learningRate: float) -> None:
        if self.debug:
            print("- Backward passing -")
            print()

        # iterative in reverse order for backwards -> delta calc -> update
        offset = len(self.cache) - 2
        for i in range(offset, -1, -1):
            self.layers[i].backward(self.cache[i+1], self.cache[i])

            if self.debug:
                print("Layer type: " + str(type(self.layers[i])))
                print("Corresponding deltas: \n" + str(self.cache[i][0].deltas))
                print()
        if self.debug:
            print('###########################################')
            print()

        if self.debug:
            print("- Deltas -")
            print()

        for i in range(offset, -1, -1):
            self.layers[i].calculateDeltaParams(self.cache[i + 1], self.cache[i])
            self.layers[i].updateParams(learningRate)

            if self.debug:
                if isinstance(self.layers[i], FullyConnectedLayer):
                    print("Layer type: " + str(type(self.layers[i])))
                    print("Weights: \n" + str(self.layers[i].weightMatrix.elements))
                    print("Bias: \n" + str(self.layers[i].bias.elements))
                    print()
                elif isinstance(self.layers[i], Conv2DLayer):
                    print("Layer type: " + str(type(self.layers[i])))
                    print("Filters: \n" + str(self.layers[i].filters[0].elements))
                    print("Bias: \n" + str(self.layers[i].filters[1].elements))
                    print()

        if self.debug:
            print('###########################################')
