import argparse
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.layers import Input

def get_vgg_activation(tensor, layer_name):
    input_tensor = Input(tensor=tensor, shape=tensor.shape)
    model = vgg16.VGG16(input_tensor=input_tensor, input_shape=(256, 256, 3), weights='imagenet', include_top=False)
    outputs_dict = {}
    for layer in model.layers:
        outputs_dict[layer.name] = layer.output
        layer.trainable = False
    return outputs_dict[layer_name]

parser = argparse.ArgumentParser(description='COCO dataset loader')

parser.add_argument("training_data_path", type=str, help="Path to training images")

args = parser.parse_args()

num_epoch = 1
num_itr = 5
training_batch_size = 1
img_width = 256
img_height = 256

content_layers = ['block2_conv2']
style_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3']

iteration = 0

datagen = ImageDataGenerator(rescale=1. / 255)

print args.training_data_path

for i in range(num_epoch):
    for x in datagen.flow_from_directory(args.training_data_path, class_mode=None, batch_size=training_batch_size,
                                         target_size=(img_width, img_height), shuffle=False):
        print x.shape
        tensor = K.variable(x)
        activation = get_vgg_activation(tensor, content_layers[0])

        iteration += training_batch_size
        print "Iteration: ", iteration

        if iteration >= num_itr:
            break