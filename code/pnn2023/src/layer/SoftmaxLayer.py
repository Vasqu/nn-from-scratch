import numpy as np
from typing import List
from src.layer.Layer import Layer
from src.model.Tensor import Tensor


class SoftmaxLayer(Layer):
    eps = .1

    def forward(self, inTensors: List[Tensor], outTensors: List[Tensor]) -> None:
        for i in range(len(inTensors)):
            # for each i: e^i / sum (using normalization and small eps)
            exps = np.exp(inTensors[i].elements - np.max(inTensors[i].elements) + self.eps)
            np.divide(exps, np.sum(exps), out=outTensors[i].elements)

    def backward(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        for i in range(len(outTensors)):
            # calculation of the jacobi matrix in a more elegant way
            # see here for reference -> https://github.com/eliben/deep-learning-samples/blob/d5ca86c5db664fabfb302cbbc231c50ec3d6a103/softmax/softmax.py#LL21C4-L21C4
            sm = outTensors[i].elements
            jacobi = np.diag(sm.flatten()) - np.outer(sm, sm)

            # dY * jacobi
            np.dot(outTensors[i].deltas, jacobi, out=inTensors[i].deltas)

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
