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
from keras.preprocessing.image import load_img, img_to_array
import math

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


def save_weights(model, style_name, directory=None):
    '''
    Saves the weights of the FastNet model.
    It creates a temporary save file having the weights of FastNet + VGG,
    loads the weights into just the FastNet model and then deletes the
    FastNet + VGG weights.
    Args:
        style_name: style image name
        directory: base directory of saved weights
    '''
    import os

    if directory is not None:
        if not os.path.exists(directory):
            os.makedirs(directory)

        full_weights_fn = directory + "style_transfer_full_model_single_%s.h5" % style_name
    else:
        full_weights_fn = "style_transfer_full_model_single_%s.h5" % style_name

    model.save_weights(filepath=full_weights_fn, overwrite=True)
    # f = h5py.File(full_weights_fn)
    #
    # layer_names = [name for name in f.attrs['layer_names']]
    #
    #
    # for i, layer in enumerate(fastnet_model.layers):
    #     g = f[layer_names[i]]
    #     weights = [g[name] for name in g.attrs['weight_names']]
    #     layer.set_weights(weights)
    #
    # if directory is not None:
    #     weights_fn = directory + "style_transfer_net_%s.h5" % style_name
    # else:
    #     weights_fn = "style_transfer_net_%s.h5" % style_name
    #
    # fastnet_model.save_weights(weights_fn, overwrite=True)
    #
    # f.close()
    # os.remove(full_weights_fn)  # The full weights aren't needed anymore since we only need 1 forward pass
    # # through the fastnet now.
    print("Saved transfer net weights for style : %s.h5" % style_name)


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

    content_activation = Input(shape=(128, 128, 128))
    style_activation1 = Input(shape=(256, 256, 64))
    style_activation2 = Input(shape=(128, 128, 128))
    style_activation3 = Input(shape=(64, 64, 256))
    style_activation4 = Input(shape=(32, 32, 512))

    total_variation_loss = Lambda(get_TV, output_shape=(1,), name='tv')(c4)

    x = Convolution2D(64, 3, 3, activation='relu', name='block1_conv1', border_mode='same')(c4)
    x = Convolution2D(64, 3, 3, activation='relu', name='block1_conv2', border_mode='same')(x)
    style_loss1 = Lambda(get_style_loss, output_shape=(1,), name='style1')([x, style_activation1])

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Convolution2D(128, 3, 3, activation='relu', name='block2_conv1', border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', name='block2_conv2', border_mode='same')(x)
    content_Loss = Lambda(get_content_loss, output_shape=(1,), name='content')([x, content_activation])
    style_loss2 = Lambda(get_style_loss, output_shape=(1,), name='style2')([x, style_activation2])

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Convolution2D(256, 3, 3, activation='relu', name='block3_conv1', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', name='block3_conv2', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', name='block3_conv3', border_mode='same')(x)
    style_loss3 = Lambda(get_style_loss, output_shape=(1,), name='style3')([x, style_activation3])
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Convolution2D(512, 3, 3, activation='relu', name='block4_conv1', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='block4_conv2', border_mode='same')(x)

    x = Convolution2D(512, 3, 3, activation='relu', name='block4_conv3', border_mode='same')(x)
    style_loss4 = Lambda(get_style_loss, output_shape=(1,), name='style4')([x, style_activation4])
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Convolution2D(512, 3, 3, activation='relu', name='block5_conv1', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='block5_conv2', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='block5_conv3', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Model(
        [input, content_activation, style_activation1, style_activation2, style_activation3, style_activation4],
        [content_Loss, style_loss1, style_loss2, style_loss3, style_loss4, total_variation_loss, c4])
    model_layers = {layer.name: layer for layer in model.layers}
    original_vgg = vgg16.VGG16(weights='imagenet', include_top=False)
    original_vgg_layers = {layer.name: layer for layer in original_vgg.layers}

    # load weight
    for layer in original_vgg.layers:
        if layer.name in model_layers:
            print layer.name, model_layers[layer.name].output_shape
            model_layers[layer.name].set_weights(original_vgg_layers[layer.name].get_weights())
            model_layers[layer.name].trainable = False

    print "VGG model built successfully!"
    return model