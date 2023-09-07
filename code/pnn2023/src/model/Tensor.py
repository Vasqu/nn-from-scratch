import numpy as np
from typing import Tuple, List


class Tensor:
    elements: np.ndarray[float]
    shape: Tuple[int]
    deltas: np.ndarray[float]
    mask: List[Tuple[int, int, int]]

    def __init__(self, elements):
        self.elements = elements
        self.shape = elements.shape
        self.deltas = np.zeros(shape=elements.shape)
        self.mask = []
