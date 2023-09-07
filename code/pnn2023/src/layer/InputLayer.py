import numpy as np
from typing import List
from src.model.Tensor import Tensor


class InputLayer:
    # assuming that any is a float
    def forward(self, rawData: List[any]) -> List[Tensor]:
        res = np.array(rawData)
        # along x axis
        # -> [1, 2, 3] is transformed into [[1, 2, 3]] (ffnn)
        # -> [[1, 2, 3], [4, 5, 6]] is transformed into [ [[1, 2, 3], [4, 5, 6]] ] (cnn)
        if res.ndim in (1, 2):
            res = np.expand_dims(res, axis=0)

        return [Tensor(res)]
