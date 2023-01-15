# -*- coding: utf-8 -*-
# @Author  : Tian Hongrong
# @Time    : 2023/1/10 10:42
# @Function: BNN training program


from __future__ import print_function
import numpy as np
from keras.utils import np_utils
from sklearn.datasets import load_iris
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from BNN_model.binary_layers import BinaryDense
from BNN_model.binary_ops import binary_tanh as binary_tanh_op



# np.random.seed(1337)  # for reproducibility


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


def binary_tanh(x):
    return binary_tanh_op(x)


batch_size = 8
epochs = 1000
nb_classes = 3


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

# learning rate schedule
lr_start = 1e-4
lr_end = 1e-5
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
drop_in = 0.2
drop_hidden = 0.5

iris_data = load_iris()
input_data = iris_data.data

## binaries the inputs

binary_data = input_data

# convert 150*4 input to 150*8 and binaries
new_2bit_data = np.zeros(shape=(150 * 8))
SeLen = input_data[:, 0]
SeWidth = input_data[:, 1]
PeLen = input_data[:, 2]
PeWidth = input_data[:, 3]


def sort(feature):
    MAX = np.max(feature)
    MIN = np.min(feature)
    A = (MAX - MIN) / 4

    for i in range(150):
        if feature[i] < MIN + A and feature[i] >= MIN:
            feature[i] = 0
        elif feature[i] < MIN + 2 * A and feature[i] >= A + MIN:
            feature[i] = 1
        elif feature[i] < MIN + 3 * A and feature[i] >= 2 * A + MIN:
            feature[i] = 2
        elif feature[i] <= MAX and feature[i] >= 3 * A + MIN:
            feature[i] = 3
    return feature


SeLen_sort = sort(SeLen)
SeWidth_sort = sort(SeWidth)
PeLen_sort = sort(PeLen)
PeWidth_sort = sort(PeWidth)


def binaries(feature):
    b = np.zeros(shape=(150, 2))

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
# print(new_binary_data.shape)


input_data = new_binary_data
correct = iris_data.target
n_data = len(correct)

nb_classes = 3
# one-hot encoding and constraint to -1, 1 for hinge loss
correct_data = np_utils.to_categorical(correct, nb_classes)*2 - 1

# data processing, divide into training and test dataset
index = np.arange(n_data)
index_train = index[index % 2 == 0]
index_test = index[index % 2 != 0]

input_train = input_data[index_train, :]
correct_train = correct_data[index_train, :]
input_test = input_data[index_test, :]
correct_test = correct_data[index_test, :]

model = Sequential()
model.add(DropoutNoScale(drop_in, input_shape=(8,), name='dropIn'))

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
np.save('middle_weight_100hid_2bit.npy', middle_weight)
output_weight = weight[2]
np.save('output_weight_100hid_2bit.npy', output_weight)


score0 = model.evaluate(input_test, correct_test, verbose=0)
score1 = model.evaluate(input_train, correct_train, verbose=0)


print('Train score:', score1[0])
print('Train accuracy:', score1[1])
print('Test score:', score0[0])
print('Test accuracy:', score0[1])

