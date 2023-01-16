# -*- coding: utf-8 -*-
# @Author  : Tian Hongrong
# @Time    : 2023/1/15 20:35
# @Function:

# -*- coding: utf-8 -*-
# @Author  : Tian Hongrong
# @Time    : 2023/1/15 20:07
# @Function:

# -*- coding: utf-8 -*-
# %matplotlib inline
import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.utils import np_utils
import numpy as np
import keras.backend as K
from keras.utils import np_utils
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from BNN_model.binary_layers import BinaryDense
from BNN_model.binary_ops import binary_tanh as binary_tanh_op

class DropoutNoScale(Dropout):
    '''Keras Dropout does scale the input in training phase, which is undesirable here.
    '''
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed) * (1 - self.rate)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs


iris_data = datasets.load_iris()
input_data = iris_data.data

# convert 150*4 input input 150*12 since we use 3-bit representation now and binaries it
new_binary_data = np.zeros(shape=(150 * 12))
SeLen = input_data[:, 0]
SeWidth = input_data[:, 1]
PeLen = input_data[:, 2]
PeWidth = input_data[:, 3]


def sort(feature):
    MAX = np.max(feature)
    MIN = np.min(feature)
    A = (MAX - MIN) / 16
    for i in range(150):
        if feature[i] < MIN + A and feature[i] >= MIN:
            feature[i] = 0
        elif feature[i] < MIN + 2 * A and feature[i] >= A + MIN:
            feature[i] = 1
        elif feature[i] < MIN + 3 * A and feature[i] >= 2 * A + MIN:
            feature[i] = 2
        elif feature[i] < MIN + 4 * A and feature[i] >= 3 * A + MIN:
            feature[i] = 3
        elif feature[i] < MIN + 5 * A and feature[i] >= 4 * A + MIN:
            feature[i] = 4
        elif feature[i] < MIN + 6 * A and feature[i] >= 5 * A + MIN:
            feature[i] = 5
        elif feature[i] < MIN + 7 * A and feature[i] >= 6 * A + MIN:
            feature[i] = 6
        elif feature[i] < MIN + 8 * A and feature[i] >= 7 * A + MIN:
            feature[i] = 7
        elif feature[i] < MIN + 9 * A and feature[i] >= 8 * A + MIN:
            feature[i] = 8
        elif feature[i] < MIN + 10 * A and feature[i] >= 9 * A + MIN:
            feature[i] = 9
        elif feature[i] < MIN + 11 * A and feature[i] >= 10 * A + MIN:
            feature[i] = 10
        elif feature[i] < MIN + 12 * A and feature[i] >= 11 * A + MIN:
            feature[i] = 11
        elif feature[i] < MIN + 13 * A and feature[i] >= 12 * A + MIN:
            feature[i] = 12
        elif feature[i] < MIN + 14 * A and feature[i] >= 13 * A + MIN:
            feature[i] = 13
        elif feature[i] < MIN + 15 * A and feature[i] >= 14 * A + MIN:
            feature[i] = 14
        elif feature[i] <= MAX and feature[i] >= 15 * A + MIN:
            feature[i] = 15
    return feature


def binary_tanh(x):
    return binary_tanh_op(x)


SeLen_sort = sort(SeLen)
SeWidth_sort = sort(SeWidth)
PeLen_sort = sort(PeLen)
PeWidth_sort = sort(PeWidth)


def binaries(feature):
    B = np.zeros(shape=(150, 4))
    for i in range(150):
        for j in range(3):
            if feature[i] == 0:
                B[i][0] = -1
                B[i][1] = -1
                B[i][2] = -1
                B[i][3] = -1
            elif feature[i] == 1:
                B[i][0] = 1
                B[i][1] = -1
                B[i][2] = -1
                B[i][3] = -1
            elif feature[i] == 2:
                B[i][0] = -1
                B[i][1] = 1
                B[i][2] = -1
                B[i][3] = -1
            elif feature[i] == 3:
                B[i][0] = 1
                B[i][1] = 1
                B[i][2] = -1
                B[i][3] = -1
            elif feature[i] == 4:
                B[i][0] = -1
                B[i][1] = -1
                B[i][2] = 1
                B[i][3] = -1
            elif feature[i] == 5:
                B[i][0] = 1
                B[i][1] = -1
                B[i][2] = 1
                B[i][3] = -1
            elif feature[i] == 6:
                B[i][0] = -1
                B[i][1] = 1
                B[i][2] = 1
                B[i][3] = -1
            elif feature[i] == 7:
                B[i][0] = 1
                B[i][1] = 1
                B[i][2] = 1
                B[i][3] = -1
            elif feature[i] == 8:
                B[i][0] = -1
                B[i][1] = -1
                B[i][2] = -1
                B[i][3] = 1
            elif feature[i] == 9:
                B[i][0] = -1
                B[i][1] = -1
                B[i][2] = -1
                B[i][3] = 1
            elif feature[i] == 10:
                B[i][0] = -1
                B[i][1] = 1
                B[i][2] = -1
                B[i][3] = 1
            elif feature[i] == 11:
                B[i][0] = 1
                B[i][1] = 1
                B[i][2] = -1
                B[i][3] = 1
            elif feature[i] == 12:
                B[i][0] = -1
                B[i][1] = -1
                B[i][2] = 1
                B[i][3] = 1
            elif feature[i] == 13:
                B[i][0] = 1
                B[i][1] = -1
                B[i][2] = 1
                B[i][3] = 1
            elif feature[i] == 14:
                B[i][0] = -1
                B[i][1] = 1
                B[i][2] = 1
                B[i][3] = 1
            elif feature[i] == 15:
                B[i][0] = 1
                B[i][1] = 1
                B[i][2] = 1
                B[i][3] = 1

    return B


SeLen_binary = binaries(SeLen_sort)
SeWidth_binary = binaries(SeWidth_sort)
PeLen_binary = binaries(PeLen_sort)
PeWidth_binary = binaries(PeWidth_sort)

new_binary_data = np.c_[SeLen_binary, SeWidth_binary, PeLen_binary, PeWidth_binary]
np.save('binary_input_data.npy', new_binary_data)

input_data = new_binary_data
correct = iris_data.target
n_data = len(correct)

nb_classes = 3
# one-hot encoding and constraint to -1, 1 for hinge loss
correct_data = np_utils.to_categorical(correct, nb_classes)*2 - 1

index = np.arange(n_data)
index_train = index[index % 2 == 0]
index_test = index[index % 2 != 0]

input_train = input_data[index_train, :]
correct_train = correct_data[index_train, :]
input_test = input_data[index_test, :]
correct_test = correct_data[index_test, :]

n_train = input_train.shape[0]
n_test = input_test.shape[0]


batch_size = 8
epochs = 1000
nb_classes = 3

# learning rate schedule
lr_start = 1e-4
lr_end = 1e-5
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9
stddev = 10
H = 'Glorot'
#kernel_lr_multiplier = keras.initializers.random_normal(mean=0,stddev=stddev)
#H = 'Glorot'
kernel_lr_multiplier = 'Glorot'
#H = 0.5
#kernel_lr_multiplier = 10

# network
num_unit = 100
num_hidden = 1
use_bias = True
stddev = 100
bias_initializer = 'zeros' # without bias
# bias_initializer = keras.initializers.RandomNormal(mean=0,stddev=stddev)

# dropout
drop_in = 0.2
drop_hidden = 0.5

model = Sequential()
model.add(DropoutNoScale(drop_in, input_shape=(16,), name='dropIn'))

# middle layer
model.add(BinaryDense(num_unit, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,
                      name='dense1', bias_initializer=bias_initializer))
# model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1)))
model.add(Activation(binary_tanh, name='act1'))
model.add(DropoutNoScale(drop_hidden, name='drop1'))

# output layer
model.add(BinaryDense(3, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,
          name='denseOut', bias_initializer=bias_initializer))


opt = Adam(lr=lr_start)
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])

lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(input_train, correct_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=0, validation_data=(input_test,correct_test),
                    callbacks=[lr_scheduler])


weight = model.get_weights()
middle_weight = weight[0]
np.save('middle_weight_100hid_4bit.npy', middle_weight)
output_weight = weight[2]
np.save('output_weight_100hid_4bit.npy', output_weight)


score0 = model.evaluate(input_test, correct_test, verbose=0)
score1 = model.evaluate(input_train, correct_train, verbose=0)


print('Train score:', score1[0])
print('Train accuracy:', score1[1])
print('Test score:', score0[0])
print('Test accuracy:', score0[1])

