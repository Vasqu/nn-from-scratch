import numpy as np
from typing import List
from src.layer.Layer import Layer
from src.model.Tensor import Tensor


class FullyConnectedLayer(Layer):
    weightMatrix: Tensor
    bias: Tensor
    inShape: int
    outShape: int

    def __init__(self, inShape, outShape):
        self.weightMatrix = Tensor(np.random.uniform(low=-1.0, high=1.0, size=(inShape, outShape)))
        self.bias = Tensor(np.zeros(shape=(1, outShape)))
        self.inShape = inShape
        self.outShape = outShape


    def forward(self, inTensors: List[Tensor], outTensors: List[Tensor]) -> None:
        for i in range(len(inTensors)):
            # X * W + b
            np.dot(inTensors[i].elements, self.weightMatrix.elements, out=outTensors[i].elements)
            outTensors[i].elements += self.bias.elements

    def backward(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        for i in range(len(outTensors)):
            # dY * W^T
            np.dot(outTensors[i].deltas, self.weightMatrix.elements.T, out=inTensors[i].deltas)

    def calculateDeltaParams(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        # shouldn't loop but just for one tensor
        for i in range(len(outTensors)):
            # X^T * dY
            np.dot(inTensors[i].elements.T, outTensors[i].deltas, out=self.weightMatrix.deltas)
            # dY
            self.bias.deltas = outTensors[i].deltas

    def updateParams(self, learningRate: float):
        # para <- para - alph * dPara
        self.weightMatrix.elements -= learningRate * self.weightMatrix.deltas
        self.bias.elements -= learningRate * self.bias.deltas

    def setLabels(self, labels: [float]):
        pass

    def getInShape(self):
        return (1, self.inShape)

    def getOutShape(self, inShape=None):
        return (1, self.outShape)
