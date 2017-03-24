import keras
from keras.models import Sequential
from keras.layers import merge
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.applications import vgg16
import cv2
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
from keras import backend as K
import re
import multiprocessing, time
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model


WIDTH = 256
HEIGHT = 256

def process_image(image_path):
    '''
    Preprocess image for VGG 16
    subtract mean pixel value and resize to 256*256
    '''
    img = load_img(image_path, target_size=(256, 256))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


def get_content_loss(original_activation, new_activation):
    '''
    Get content loss between original activation and our new image activation
    Returns: a tensor variable representing content loss
    '''
    shape = K.shape(original_activation).eval()
    size = shape[0] * shape[1] * shape[2]  # W * H * C
    return K.sum(K.square(original_activation - new_activation)) / float(size)


def gram_matrix(activation):
    C = K.shape(activation).eval()[2]
    w = K.shape(activation).eval()[0]
    h = K.shape(activation).eval()[1]
    shape = (C, w * h)
    # reshape to (C, H*W)
    activation = K.reshape(activation, shape)
    return K.dot(activation, K.transpose(activation))


def get_style_loss(original_gram_matrix, new_gram_matrix):
    shape = K.shape(original_gram_matrix).eval()
    size = shape[0] * shape[1]  # W * H * C
    return K.sum(K.square(original_gram_matrix - new_gram_matrix))  / float(size)

# input image
content = process_image("./image/baby.jpg")
style = process_image('./image/style.jpg')
transfer = process_image('./image/tranfered.jpg')
content_tensor = K.variable(content)
style_tensor = K.variable(style)
transfer_tensor = K.variable(transfer)

x = merge([content_tensor, style_tensor, transfer_tensor], mode='concat', concat_axis=0)
model = vgg16.VGG16(input_tensor= x, weights='imagenet', include_top=False)
print('Model loaded.')

outputs_dict = {}
for layer in model.layers:
    print layer.name, layer.output
    outputs_dict[layer.name] = layer.output
    layer.trainable = False

content_act = outputs_dict['block1_conv2'][0]
style_act = outputs_dict['block1_conv2'][1]
transfer_act = outputs_dict['block1_conv2'][2]

G = gram_matrix(style_act)
G2 = gram_matrix(content_act)
G3 = gram_matrix(transfer_act)

print G.eval().shape

print "style \n"
print get_style_loss(G, G3).eval()
print get_style_loss(G2, G3).eval()

print "content \n"
print get_content_loss(content_act, transfer_act).eval()
print get_content_loss(transfer_act, style_act).eval()