import numpy as np
from typing import List, Tuple
from src.layer.Layer import Layer
from src.model.Tensor import Tensor


class Conv2DLayer(Layer):
    # filters [numFilters, c, x, y] x bias
    filters: Tuple[Tensor, Tensor]
    xShape: int
    yShape: int
    channels: int
    numFilters: int

    def __init__(self, x, y, channels, numFilters):
        # for testing purposes ~
        # filter = Tensor(np.ones(shape=(numFilters, channels, x, y)))

        filter = Tensor(np.random.uniform(low=-0.5, high=0.5, size=(numFilters, channels, x, y)))
        bias = Tensor(np.zeros(shape=(numFilters, 1, x, y)))
        self.filters = (filter, bias)

        self.xShape = x
        self.yShape = y
        self.channels = channels
        self.numFilters = numFilters

    def patch_generator(self, matrix, f_x, f_y):
        _, m_x, m_y = matrix.shape
        for x in range(m_x - f_x + 1):
            for y in range(m_y - f_y + 1):
                patch = matrix[:, x:(x + f_x), y:(y + f_y)]
                yield patch, x, y

    def forward(self, inTensors: List[Tensor], outTensors: List[Tensor]) -> None:
        for i in range(len(inTensors)):
            for patch, x, y in self.patch_generator(inTensors[i].elements, self.xShape, self.yShape):
                f_res = np.sum(
                    patch * self.filters[0].elements,
                    axis=(1, 2, 3)
                )
                f_bias = self.filters[1].elements.sum(axis=(1, 2, 3))
                outTensors[i].elements[:, x, y] = f_res + f_bias

    def backward(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        for i in range(len(outTensors)):
            # first transpose the filters (i.e. switch filter_dim with channel_dim)
            tf = np.transpose(self.filters[0].elements, axes=[1, 0, 2, 3])

            # then rotate each new filter 180°
            rf = np.rot90(tf, k=2, axes=(2, 3))

            # take x and y shape of kernel (for each the same)
            x_rotate = rf.shape[2]
            y_rotate = rf.shape[3]

            # calculate paddings
            p_x = x_rotate - 1
            p_y = y_rotate - 1

            # not changing channels and adding the padding on both ends of each dimension
            npad = ((0, 0), (p_x, p_x), (p_y, p_y))
            deltas = np.pad(outTensors[i].deltas, pad_width=npad, mode='constant')

            # see @forward, same stuff but different filter and starting 'image'
            for patch, x, y in self.patch_generator(deltas, rf.shape[2], rf.shape[3]):
                # each filter summed up over all channels
                f_res = np.sum(
                    patch * rf,
                    axis=(1, 2, 3)
                )
                inTensors[i].deltas[:, x, y] = f_res

    def calculateDeltaParams(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        for i in range(len(outTensors)):
            for filter_index in range(self.numFilters):
                # copied specific channel (c==f) of delta total filter amount of times
                deltas_filter = np.repeat(np.expand_dims(outTensors[i].deltas[filter_index], axis=0), self.channels,
                                          axis=0)

                for patch, x, y in self.patch_generator(inTensors[i].elements, deltas_filter.shape[1],
                                                        deltas_filter.shape[2]):
                    # summed over so that the result for a channel is in its corresponding index
                    c_res = np.sum(
                        patch * deltas_filter,
                        axis=(1, 2)
                    )

                    # assign filter deltas to correct channels
                    for c in range(self.channels):
                        self.filters[0].deltas[filter_index][c, x, y] = c_res[c]

                # bias is usually one number, but we use a 2d-tensor --> equal distribution on all entries
                # (don't ask why I did this useless thing :-:)
                self.filters[1].deltas[filter_index] = np.full((1, self.xShape, self.yShape),
                                                               deltas_filter[0].sum() / (self.xShape * self.yShape),
                                                               dtype=np.float64)

    def updateParams(self, learningRate: float):
        # para <- para - alph * dPara
        self.filters[0].elements -= learningRate * self.filters[0].deltas
        self.filters[1].elements -= learningRate * self.filters[1].deltas

    def setLabels(self, labels: [float]):
        pass

    def getInShape(self):
        return (self.channels, self.xShape, self.yShape)

    def getOutShape(self, inShape=None):
        return (self.numFilters, inShape[1] - self.xShape + 1, inShape[2] - self.yShape + 1)
