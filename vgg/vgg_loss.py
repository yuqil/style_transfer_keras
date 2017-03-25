import keras
from keras.models import Sequential
from keras.layers import merge
from keras.layers.core import Flatten, Dense, Lambda
from keras.layers import Conv2D, Input
from keras.models import Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.applications import vgg16
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array


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
    C = K.shape(activation).eval()[2]
    w = K.shape(activation).eval()[0]
    h = K.shape(activation).eval()[1]
    shape = (w * h, C)
    # reshape to (C, H*W)
    activation = K.reshape(activation, shape)
    return K.dot(K.transpose(activation), activation)

def get_style_loss(args):
    new_activation, style_activation = args[0], args[1]
    original_gram_matrix = gram_matrix(style_activation)
    new_gram_matrix = gram_matrix(new_activation)
    return K.sum(K.square(original_gram_matrix - new_gram_matrix))



WIDTH = 256
HEIGHT = 256

content_layers = 'block4_conv2'
style_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']

def get_loss_model():
    input = Input(shape=(WIDTH, HEIGHT, 3))
    content_activation = Input(shape=(1, 32, 32, 512))
    # style_activation1 = Input(shape=(1, 256, 256, 64))

    # style_activation2 = Input(shape=(128, 128, 128))
    # style_activation3 = Input(shape=(64, 64, 256))
    # style_activation4 = Input(shape=(32, 32, 512))
    # style_activation5 = Input(shape=(16, 16, 512))

    x = Convolution2D(64, 3, 3, activation='relu', name='block1_conv1', border_mode='same')(input)
    # style_loss1 = Lambda(get_style_loss, output_shape=(1,), name='style1')([x, style_activation1])

    x = Convolution2D(64, 3, 3, activation='relu', name='block1_conv2', border_mode='same')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name = 'block1_pool')(x)

    x = Convolution2D(128, 3, 3, activation='relu', name='block2_conv1', border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', name='block2_conv2', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name = 'block2_pool')(x)

    x = Convolution2D(256, 3, 3, activation='relu', name='block3_conv1', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', name='block3_conv2', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', name='block3_conv3', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name = 'block3_pool')(x)

    x = Convolution2D(512, 3, 3, activation='relu', name='block4_conv1', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='block4_conv2', border_mode='same')(x)

    content_Loss = Lambda(get_content_loss, output_shape=(1,), name='content')([x, content_activation])

    x = Convolution2D(512, 3, 3, activation='relu', name='block4_conv3', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name = 'block4_pool')(x)

    x = Convolution2D(512, 3, 3, activation='relu', name='block5_conv1', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='block5_conv2', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='block5_conv3', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name = 'block5_pool')(x)

    model = Model([input, content_activation], [x, content_Loss])

    model_layers = {layer.name : layer for layer in model.layers}
    original_vgg = vgg16.VGG16(weights='imagenet', include_top=False)
    original_vgg_layers = {layer.name : layer for layer in original_vgg.layers}
    for layer in original_vgg.layers:
        if layer.name in model_layers:
            print layer.name, model_layers[layer.name].output_shape
            model_layers[layer.name].set_weights(original_vgg_layers[layer.name].get_weights())
            model_layers[layer.name].trainable = False
    print "VGG model built successfully!"

    return model

def get_activation(tensor, layer_name):
    model = vgg16.VGG16(input_tensor=tensor, weights='imagenet', include_top=False)
    outputs_dict = {}
    for layer in model.layers:
        outputs_dict[layer.name] = layer.output
        layer.trainable = False
    return outputs_dict[layer_name]

# input image
content = process_image("./image/baby.jpg")
style = process_image('./image/style.jpg')
transfer = process_image('./image/tranfered.jpg')

content_tensor = K.variable(content)
style_tensor = K.variable(style)
transfer_tensor = K.variable(transfer)

content_activation = get_activation(content_tensor, 'block4_conv2')
print content_activation.eval().shape

model = get_loss_model()
c = model([transfer_tensor, content_activation])
print c[0].eval()
print c[1].eval()

