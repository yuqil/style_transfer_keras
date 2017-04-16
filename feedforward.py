import keras
import h5py
import numpy as np
from keras import backend as K
from keras.applications import vgg16
from keras.layers import BatchNormalization, Activation, Deconvolution2D
from keras.layers import Input
from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Lambda
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import math
import model
import sys

WIDTH = 256
HEIGHT = 256
TV_WEIGHT = math.pow(10, -6)

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


def process_image(image_path):
    '''
    Preprocess image for VGG 16
    subtract mean pixel value and resize to 256*256
    '''
    img = load_img(image_path, target_size=(WIDTH, HEIGHT))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


def get_content_loss(args):
    new_activation, content_activation = args[0], args[1]
    return K.mean(K.square(new_activation - content_activation))


def gram_matrix(activation):
    assert K.ndim(activation) == 3
    shape = K.shape(activation)
    shape = (shape[0] * shape[1], shape[2])
    # reshape to (H*W, C)
    activation = K.reshape(activation, shape)
    return K.dot(K.transpose(activation), activation) / (shape[0] * shape[1])


def get_style_loss(args):
    new_activation, style_activation = args[0], args[1]
    original_gram_matrix = gram_matrix(style_activation[0])

    new_gram_matrix = gram_matrix(new_activation[0])
    return K.sum(K.square(original_gram_matrix - new_gram_matrix))


def get_TV(new_gram_matrix):
    x_diff = K.square(new_gram_matrix[:, :WIDTH - 1, :HEIGHT - 1, :] - new_gram_matrix[:, 1:, :HEIGHT - 1, :])
    y_diff = K.square(new_gram_matrix[:, :WIDTH - 1, :HEIGHT - 1, :] - new_gram_matrix[:, :WIDTH - 1, 1:, :])
    return TV_WEIGHT * K.mean(K.sum(K.pow(x_diff + y_diff, 1.25)))


def get_vgg_activation(tensor, layer_name):
    input_tensor = Input(tensor=tensor, shape=tensor.shape)
    model = vgg16.VGG16(input_tensor=input_tensor, input_shape=(256, 256, 3), weights='imagenet', include_top=False)
    outputs_dict = {}
    for layer in model.layers:
        outputs_dict[layer.name] = layer.output
        layer.trainable = False
    return outputs_dict[layer_name]


def dummy_loss_function(y_true, y_pred):
    return y_pred


def zero_loss_function(y_true, y_pred):
    return K.variable(np.zeros(1,))

def get_loss_model():
    input = Input(shape=(WIDTH, HEIGHT, 3))

    c1 = Convolution2D(32, 9, 9, activation='linear', border_mode='same')(input)
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

    c4 = Convolution2D(3, 9, 9, activation='tanh', name='output', border_mode='same')(d2)


    model = Model([input], [c4])

    return model

if len(sys.argv) != 2:
    print "python feedforward.py weight path"


trained_model = model.get_loss_model()
trained_model.load_weights(sys.argv[1])
model = get_loss_model()

print "trained model"
for layer in trained_model.layers:
    print layer.name, layer.output_shape

print
print "new model"
for layer in model.layers:
    print layer.name, layer.output_shape

print ""
print "start"

trained_model_layers = [layer for layer in trained_model.layers]
model_layers = [layer for layer in model.layers]

for i in xrange(0, len(model_layers)):
    new_layer = model_layers[i]
    old_layer = trained_model_layers[i]
    print new_layer.name, old_layer.name
    new_layer.set_weights(old_layer.get_weights())

