import numpy as np
from typing import List
from src.layer.Layer import Layer
from src.model.Tensor import Tensor


class ReLu:
    def apply(self, x):
        return np.float64(max(0, x))

    # technically 0 isn't defined but for safety purpose (==0)
    def derive(self, x):
        return np.float64(1 if x >= 0 else 0)


class Sigmoid:
    def apply(self, x):
        return np.float64(1 / (1 + np.e ** (-x)))

    def derive(self, x):
        return np.float64(self.apply(x) * (1 - self.apply(x)))


class TanH:
    def apply(self, x):
        return np.float64(((np.e ** (2 * x)) - 1) / ((np.e ** (2 * x)) + 1))

    def derive(self, x):
        return np.float64(1 - self.apply(x) ** 2)


relu = ReLu()
sigmoid = Sigmoid()
tanh = TanH()
functions = {
    'relu' : relu,
    'sigmoid' : sigmoid,
    'tanh' : tanh
}


class ActivationLayer(Layer):
    def __init__(self, activation):
        self.activation = functions[activation]

    def forward(self, inTensors: List[Tensor], outTensors: List[Tensor]) -> None:
        for i in range(len(inTensors)):
            # elementwise application of function
            outTensors[i].elements = np.vectorize(self.activation.apply)(inTensors[i].elements)

    def backward(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        for i in range(len(outTensors)):
            # elementwise application of derivative (*) dY
            inTensors[i].deltas = np.vectorize(self.activation.derive)(inTensors[i].elements) * outTensors[i].deltas

    def calculateDeltaParams(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        pass

    def updateParams(self, learningRate: float):
        pass

    def setLabels(self, labels: [float]):
        pass

    def getInShape(self):
        pass

    def getOutShape(self, inShape=None):
        return inShape
