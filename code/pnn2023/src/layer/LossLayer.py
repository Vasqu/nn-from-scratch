import numpy as np
from typing import List
from src.layer.Layer import Layer
from src.model.Tensor import Tensor

# turn warnings into errors
import warnings
warnings.filterwarnings("error")


class CrossEntropyLoss(Layer):
    cap = 5

    def forward(self, inTensors: List[Tensor], outTensors: List[Tensor]) -> None:
        for i in range(len(inTensors)):
            try:
                # - sum_i c_i * log(x_i)
                cse = - (self.labels @ np.vectorize(np.log)(inTensors[i].elements).T).sum()
            except RuntimeWarning:
                # cause of warning
                x_t = np.float64(int(inTensors[i].elements[0][np.argmax(self.labels)]))
                # other irrelevant values are too close to zero
                if x_t == 1:
                    cse = np.float64(0)
                # the pred itself is too close to zero
                else:
                    # cap loss at @self.cap
                    cse = np.float64(self.cap)
            outTensors[i].elements = abs(cse)

    def backward(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        for i in range(len(inTensors)):
            try:
                # - c_i / x_i
                cse_delta = - self.labels / (inTensors[i].elements)
            except RuntimeWarning:
                # cause of warning
                x_t = np.float64(int(inTensors[i].elements[0][np.argmax(self.labels)]))
                # other irrelevant values are too close to zero
                if x_t == 1:
                    cse_delta = - self.labels / np.array([np.repeat(1, self.labels.shape[0])])
                # the pred itself is too close to zero
                else:
                    # cap based on forward pass with @self.cap
                    cse_delta = - self.labels / np.array([np.repeat((np.e ** -self.cap), self.labels.shape[0])])
            inTensors[i].deltas = cse_delta

    def calculateDeltaParams(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        pass

    def updateParams(self, learningRate: float):
        pass

    def setLabels(self, labels: [float]):
        self.labels = np.array(labels)

    def getInShape(self):
        return (1, self.labels.shape)

    def getOutShape(self, inShape=None):
        return self.getInShape()


class MeanSquaredErrorLoss(Layer):
    def forward(self, inTensors: List[Tensor], outTensors: List[Tensor]) -> None:
        for i in range(len(inTensors)):
            # 1 / n * sum_i (x_i - c_i)^2
            outTensors[i].elements = ((1 / self.labels.shape[0]) * ((inTensors[i].elements - self.labels) ** 2)).sum()

    def backward(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        for i in range(len(inTensors)):
            # x_i - c_i
            inTensors[i].deltas = inTensors[i].elements - self.labels

    def calculateDeltaParams(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        pass

    def updateParams(self, learningRate: float):
        pass

    def setLabels(self, labels: [float]):
        self.labels = np.array(labels)

    def getInShape(self):
        return (1, self.labels.shape)

    def getOutShape(self, inShape=None):
        return self.getInShape()
