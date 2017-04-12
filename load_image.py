import argparse
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser(description='COCO dataset loader')

parser.add_argument("training_data_path", type=str, help="Path to training images")

args = parser.parse_args()

num_epoch = 1
num_images = 1
training_batch_size = 1
img_width = 256
img_height = 256

iteration = 0

datagen = ImageDataGenerator(rescale=1. / 255)

print args.training_data_path

for i in range(num_epoch):
    for x in datagen.flow_from_directory(args.training_data_path, class_mode=None, batch_size=training_batch_size,
                                         target_size=(img_width, img_height), shuffle=False):
        print x
        iteration += training_batch_size

        if iteration >= num_epoch:
            break