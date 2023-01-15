# -*- coding: utf-8 -*-
# @Author  : Tian Hongrong
# @Time    : 2023/1/15 13:12
# @Function: input data pre-processing, binarize iris dataset

import numpy as np
from sklearn.datasets import load_iris

iris_data = load_iris()
input_data = iris_data.data
binary_data = input_data

"""
    use 2-bit representation to represent iris flower's 4 features, hence expand the original
    4 input to 8 input, then the input matrix is (data_num *  8) = (150 * 8)
"""

new_2bit_data = np.zeros(shape=(150 * 8))
SeLen = input_data[:, 0]
SeWidth = input_data[:, 1]
PeLen = input_data[:, 2]
PeWidth = input_data[:, 3]


def sort(feature):
    MAX = np.max(feature)
    MIN = np.min(feature)
    A = (MAX - MIN) / 4

    i = 0
    for i in range(150):
        if feature[i] < MIN + A and feature[i] >= MIN:
            feature[i] = 0
        elif feature[i] < MIN + 2 * A and feature[i] >= A + MIN:
            feature[i] = 1
        elif feature[i] < MIN + 3 * A and feature[i] >= 2 * A + MIN:
            feature[i] = 2
        elif feature[i] <= MAX and feature[i] >= 3 * A + MIN:
            feature[i] = 3
    return (feature)


SeLen_sort = sort(SeLen)
SeWidth_sort = sort(SeWidth)
PeLen_sort = sort(PeLen)
PeWidth_sort = sort(PeWidth)


def binaries(feature):
    b = np.zeros(shape=(150, 2))
    i = 0
    j = 0
    for i in range(150):
        for j in range(2):
            if feature[i] == 0:
                b[i][0] = -1
                b[i][1] = -1
            elif feature[i] == 1:
                b[i][0] = 1
                b[i][1] = -1
            elif feature[i] == 2:
                b[i][0] = -1
                b[i][1] = 1
            elif feature[i] == 3:
                b[i][0] = 1
                b[i][1] = 1
    return b


SeLen_binary = binaries(SeLen_sort)
SeWidth_binary = binaries(SeWidth_sort)
PeLen_binary = binaries(PeLen_sort)
PeWidth_binary = binaries(PeWidth_sort)

new_binary_data = np.c_[SeLen_binary, SeWidth_binary, PeLen_binary, PeWidth_binary]
np.save('binary_input_data.npy', new_binary_data)