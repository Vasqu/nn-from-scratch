import math
import numpy as np
from typing import List, Tuple
from src.layer.Layer import Layer
from src.model.Tensor import Tensor


class MaxPooling(Layer):
    filter_shape: Tuple[int, int]
    stride: Tuple[int, int]

    def __init__(self, filter_shape, stride):
        self.filter_shape = filter_shape
        self.stride = stride

    def patch_generator(self, matrix, f_x, f_y, s_x, s_y):
        _, m_x, m_y = matrix.shape
        x_o = math.floor((m_x - f_x) / s_x) + 1
        y_o = math.floor((m_y - f_y) / s_y) + 1

        for x in range(x_o):
            for y in range(y_o):
                patch = matrix[:, (x * s_x):(x * s_x + f_x), (y * s_y):(y * s_y + f_y)]
                yield patch, x, y

    def get_by_channel_max(self, idx, matrix):
        pos = (idx,) + np.unravel_index(np.argmax(matrix[idx]), matrix.shape[1:])
        return matrix[pos], pos

    def forward(self, inTensors: List[Tensor], outTensors: List[Tensor]) -> None:
        for i in range(len(inTensors)):
            # keep track of the offset of the original matrix
            x_offset, y_offset = 0, 0
            y_limit = inTensors[i].shape[2]
            for patch, x, y in self.patch_generator(inTensors[i].elements, self.filter_shape[0], self.filter_shape[1], self.stride[0], self.stride[1]):
                for c in range(inTensors[i].elements.shape[0]):
                    # max of patch and the local position in the patch
                    max, pos = self.get_by_channel_max(c, patch)
                    # save channel result at position
                    outTensors[i].elements[c, x, y] = max
                    # adjust position by offset for mask
                    pos = (int(pos[0]), int(pos[1] + x_offset), int(pos[2] + y_offset))
                    # saving the complete position as tuple for convenience
                    inTensors[i].mask.append(pos)

                # iterating through columns first, see @patch_generator
                if y_offset + self.stride[1] + patch.shape[2] - 1 < y_limit:
                    y_offset += self.stride[1]
                else:
                    y_offset = 0
                    x_offset += self.stride[0]

    def backward(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        for i in range(len(outTensors)):
            # deltas have same size as original matrix
            inTensors[i].deltas = np.zeros(shape=inTensors[i].shape)

            # iterate in same order as in forward pass to assign right value to right position
            m = 0
            for _, x, y in self.patch_generator(outTensors[i].deltas, 1, 1, 1, 1):
                for c in range(outTensors[i].deltas.shape[0]):
                    delta_y = outTensors[i].deltas[c][x][y]

                    pos = inTensors[i].mask[m]
                    inTensors[i].deltas[pos[0]][pos[1]][pos[2]] = delta_y

                    m += 1

    def calculateDeltaParams(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        pass

    def updateParams(self, learningRate: float):
        pass

    def setLabels(self, labels: [float]):
        pass

    def getInShape(self):
        pass

    def getOutShape(self, inShape=None):
        x_o = math.floor((inShape[1] - self.filter_shape[0]) / self.stride[0]) + 1
        y_o = math.floor((inShape[2] - self.filter_shape[1]) / self.stride[1]) + 1

        return (inShape[0], x_o, y_o)


class AvgPooling(Layer):
    filter_shape: Tuple[int, int]
    stride: Tuple[int, int]

    def __init__(self, filter_shape, stride):
        self.filter_shape = filter_shape
        self.stride = stride

    def patch_generator(self, matrix, f_x, f_y, s_x, s_y):
        _, m_x, m_y = matrix.shape
        x_o = math.floor((m_x - f_x) / s_x) + 1
        y_o = math.floor((m_y - f_y) / s_y) + 1

        for x in range(x_o):
            for y in range(y_o):
                patch = matrix[:, (x * s_x):(x * s_x + f_x), (y * s_y):(y * s_y + f_y)]
                yield patch, x, y

    def get_by_channel_avg(self, idx, matrix):
        return np.mean(matrix[idx])

    def forward(self, inTensors: List[Tensor], outTensors: List[Tensor]) -> None:
        for i in range(len(inTensors)):
            # keep track of the offset of the original matrix
            x_offset, y_offset = 0, 0
            y_limit = inTensors[i].shape[2]
            for patch, x, y in self.patch_generator(inTensors[i].elements, self.filter_shape[0], self.filter_shape[1], self.stride[0], self.stride[1]):
                for c in range(inTensors[i].elements.shape[0]):
                    # avg of patch and the local position in the patch
                    avg = self.get_by_channel_avg(c, patch)
                    # save channel result at position
                    outTensors[i].elements[c, x, y] = avg
                    # adjust position by offset for mask
                    pos = (c, x_offset, y_offset)
                    # saving the complete position as tuple for convenience
                    inTensors[i].mask.append(pos)

                # iterating through columns first, see @patch_generator
                if y_offset + self.stride[1] + patch.shape[2] - 1 < y_limit:
                    y_offset += self.stride[1]
                else:
                    y_offset = 0
                    x_offset += self.stride[0]

    def backward(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        for i in range(len(outTensors)):
            # deltas have same size as original matrix
            inTensors[i].deltas = np.zeros(shape=inTensors[i].shape)

            # iterate in same order as in forward pass to assign right value to right position
            m = 0
            for _, x, y in self.patch_generator(outTensors[i].deltas, 1, 1, 1, 1):
                for c in range(outTensors[i].deltas.shape[0]):
                    delta_y = outTensors[i].deltas[c][x][y] / (self.filter_shape[0] * self.filter_shape[1])

                    # assign equal distribution according to filter size
                    pos = inTensors[i].mask[m]
                    for k in range(self.filter_shape[0]):
                        for l in range(self.filter_shape[1]):
                            inTensors[i].deltas[pos[0]][pos[1] + k][pos[2] + l] = delta_y

                    m += 1

    def calculateDeltaParams(self, outTensors: List[Tensor], inTensors: List[Tensor]) -> None:
        pass

    def updateParams(self, learningRate: float):
        pass

    def setLabels(self, labels: [float]):
        pass

    def getInShape(self):
        pass

    def getOutShape(self, inShape=None):
        x_o = math.floor((inShape[1] - self.filter_shape[0]) / self.stride[0]) + 1
        y_o = math.floor((inShape[2] - self.filter_shape[1]) / self.stride[1]) + 1

        return (inShape[0], x_o, y_o)
