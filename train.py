import numpy as np
import argparse
import time
import math
import model

import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.layers import Input

NUM_EPOCH = 1
NUM_ITR = 1
TRAINING_BATCH_SIZE = 4
WIDTH = 256
HEIGHT = 256
TV_WEIGHT = math.pow(10, -6)

dummy_input = np.array([0])

content_layers = ['block2_conv2']
style_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3']

parser = argparse.ArgumentParser(description='COCO dataset loader')
parser.add_argument("training_data_path", type=str, help="Path to training images")
parser.add_argument("style_image_path", type=str, help="Path to the style image")
args = parser.parse_args()

style_tensor = K.variable(model.process_image(args.style_image_path))
style_act_1 = model.get_vgg_activation(style_tensor, style_layers[0])
style_act_2 = model.get_vgg_activation(style_tensor, style_layers[1])
style_act_3 = model.get_vgg_activation(style_tensor, style_layers[2])
style_act_4 = model.get_vgg_activation(style_tensor, style_layers[3])

# define a model
training_model = model.get_loss_model()
training_model.compile(loss={'content': model.dummy_loss_function, 'style1': model.dummy_loss_function, 'style2': model.dummy_loss_function,
                    'style3': model.dummy_loss_function, 'style4': model.dummy_loss_function, 'tv': model.dummy_loss_function,
                    'output': model.zero_loss_function},
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              loss_weights=[100, 1, 1, 1, 1, 1, 0])

t_total_1 = time.time()
prev_improvement = -1
iteration = 0
interrupted = False

datagen = ImageDataGenerator(rescale=1. / 255)

for i in range(NUM_EPOCH):
    print("Epoch : %d" % (i + 1))

    for x in datagen.flow_from_directory(args.training_data_path, class_mode=None, batch_size=TRAINING_BATCH_SIZE,
                                         target_size=(WIDTH, HEIGHT), shuffle=False):
        try:
            t1 = time.time()

            content_tensor = K.variable(x)
            content_activation = model.get_vgg_activation(content_tensor, content_layers[0])

            res = training_model.fit([x, content_activation.eval(), style_act_1.eval(), style_act_2.eval(),
                       style_act_3.eval(), style_act_4.eval()],
                      [dummy_input, dummy_input, dummy_input, dummy_input, dummy_input, dummy_input, x])

            iteration += TRAINING_BATCH_SIZE
            loss = res.history['loss'][0]

            if prev_improvement == -1:
                prev_improvement = loss

            improvement = (prev_improvement - loss) / prev_improvement * 100
            prev_improvement = loss

            t2 = time.time()

            print("Iter : %d / %d, Time elapsed : %0.2f seconds, Loss : %d, Improvement : %0.2f percent." %
                  (iteration, NUM_ITR, t2 - t1, loss, improvement))

            if iteration >= NUM_ITR:
                break
        except KeyboardInterrupt:
            print("Keyboard interrupt. Training suspended. Saving weights...")
            interrupted = True
            break

    iteration = 0
    if interrupted:
        break

model.save_weights(training_model, args.style_image_path, "weights/")