import numpy as np
import tensorflow as tf

import vgg16
import utils

def get_activations(layer_name, batch):
	#uncomment for gpu
	# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:

	with tf.device('/cpu:0'):
		with tf.Session() as sess:
			images = tf.placeholder("float", [len(batch), 224, 224, 3])
			feed_dict = {images: batch}

			vgg = vgg16.Vgg16()
			with tf.name_scope("content_vgg"):
				vgg.build(images)
			
			map_layer_name = {}
			map_layer_name['conv1_1'] = vgg.conv1_1
			map_layer_name['conv1_2'] = vgg.conv1_2
			map_layer_name['conv2_1'] = vgg.conv2_1
			map_layer_name['conv2_2'] = vgg.conv2_2
			map_layer_name['conv3_1'] = vgg.conv3_1
			map_layer_name['conv3_2'] = vgg.conv3_2
			map_layer_name['conv3_3'] = vgg.conv3_3
			map_layer_name['conv4_1'] = vgg.conv4_1
			map_layer_name['conv4_2'] = vgg.conv4_2
			map_layer_name['conv4_3'] = vgg.conv4_3
			map_layer_name['conv5_1'] = vgg.conv5_1
			map_layer_name['conv5_2'] = vgg.conv5_2
			map_layer_name['conv5_3'] = vgg.conv5_3
			map_layer_name['fc6'] = vgg.fc6
			map_layer_name['fc7'] = vgg.fc7
			map_layer_name['fc8'] = vgg.fc8

			activations = sess.run(map_layer_name[layer_name], feed_dict=feed_dict)
			return activations

#test call
if __name__ == '__main__':

	img1 = utils.load_image("./test_data/tiger.jpeg")
	img2 = utils.load_image("./test_data/puzzle.jpeg")

	batch1 = img1.reshape((1, 224, 224, 3))
	batch2 = img2.reshape((1, 224, 224, 3))

	batch = np.concatenate((batch1, batch2), 0)

	print(get_activations('conv1_1', batch))