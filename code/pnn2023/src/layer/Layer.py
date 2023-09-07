from abc import ABC, abstractmethod
from typing import List
from src.model.Tensor import Tensor

class Layer(ABC):
    @abstractmethod
    def forward(self, inTensors: List[Tensor], outTensors: List[Tensor]) -> None:
        pass

    @abstractmethod
    def backward(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        pass

    @abstractmethod
    def calculateDeltaParams(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        pass

    @abstractmethod
    def updateParams(self, learningRate: float) -> None:
        pass

    @abstractmethod
    def setLabels(self, labels: [float]):
        pass

    @abstractmethod
    def getInShape(self):
        pass

    @abstractmethod
    def getOutShape(self, inShape=None):
        pass