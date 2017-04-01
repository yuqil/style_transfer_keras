#import MNIST
from keras.datasets import mnist
import matplotlib.pyplot as plt

#import Keras Libs
import keras
from keras.models import Model
from keras.layers import Input, merge, BatchNormalization, Activation, Deconvolution2D, Convolution2D, Flatten, Dense
from keras.utils import np_utils

#import numpy
import numpy as np


# Residual Block
def residual_block(x):
    shortcut = x
    x = Convolution2D(128, 3, 3, activation='linear', border_mode='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Convolution2D(128, 3, 3, activation='linear', border_mode='same')(x)
    x = BatchNormalization(axis=1)(x)
    m = merge([x, shortcut], mode='sum')
    return m


def get_transfer_net():

    #x = Input(shape=(28, 28, 1))
    x = Input(shape=(256, 256, 3))

    c1 = Convolution2D(32, 9, 9, activation='linear', border_mode='same')(x)
    c1 = BatchNormalization(axis=1)(c1)
    c1 = Activation('relu')(c1)


    c2 = Convolution2D(64, 3, 3, activation='linear', border_mode='same',
		       subsample=(2, 2))(c1)
    c2 = BatchNormalization(axis=1)(c2)
    c2 = Activation('relu')(c2)


    c3 = Convolution2D(128, 3, 3, activation='linear', border_mode='same',
		       subsample=(2, 2))(c2)
    c3 = BatchNormalization(axis=1)(c3)
    c3 = Activation('relu')(c3)

    r1 = residual_block(c3)
    r2 = residual_block(r1)
    r3 = residual_block(r2)
    r4 = residual_block(r3)
    r5 = residual_block(r4)

    d1 = Deconvolution2D(64, 3, 3, activation='linear', border_mode='same',
			 subsample=(2, 2), output_shape=(1, 128, 128))(r5)

    d2 = Deconvolution2D(32, 3, 3, activation='linear', border_mode='same',
			 subsample=(2, 2), output_shape=(1, 256, 256))(d1)

    c4 = Convolution2D(3, 9, 9, activation='tanh', border_mode='same')(d2)
    #c4 = Convolution2D(3, 9, 9, activation='linear', border_mode='same')(d2)

    #for MNIST
    #f1 = Flatten()(r1)
    #out = Dense(10, input_shape=(3, ), activation='softmax')(f1)

    m = Model(x, out)
    return m

'''
model = get_transfer_net()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

for layer in model.layers:
	print(layer.output_shape)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
input_shape = (28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.fit(x_train, y_train, batch_size=128, nb_epoch=100,
                      verbose=2, validation_data=(x_test, y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy;', score[1])
'''
