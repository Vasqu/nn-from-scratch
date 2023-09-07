import numpy as np
from typing import List, Tuple
from functools import reduce
from src.layer.Layer import Layer
from src.model.Tensor import Tensor

class FlattenLayer(Layer):
    input_shape = Tuple[int, int, int]

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, inTensors: List[Tensor], outTensors: List[Tensor]) -> None:
        for i in range(len(inTensors)):
            outTensors[i].elements = np.array([inTensors[i].elements.flatten()])

    def backward(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        for i in range(len(outTensors)):
            inTensors[i].deltas = outTensors[i].deltas.reshape(self.input_shape)

    def calculateDeltaParams(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        pass

    def updateParams(self, learningRate: float):
        pass

    def setLabels(self, labels: [float]):
        pass

    def getInShape(self):
        return self.input_shape

    def getOutShape(self, inShape=None):
        return (1, reduce(lambda x, y: x * y, self.input_shape))
