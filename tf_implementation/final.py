import argparse
import tensorflow as tf
import time
import sys
import cv2
import numpy as np
import scipy

import process_data
from transfer_net import transfer_net
import residualnet
import vgg_model
import loss_function
from process_data import data_set
import functools

# Basic model parameters as external flags.
FLAGS = None
CHANNELS = 3
WIDTH = 256
HEIGHT = 256
LEARNING_RATE = 0.001

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  is_training = tf.placeholder(dtype=tf.bool)
  content_placeholder = tf.placeholder(tf.float32, shape=(batch_size,  WIDTH, HEIGHT, CHANNELS))
  return content_placeholder, is_training


def fill_feed_dict(data_set, X_content, is_training, style_placeholder, iteration, batch_size):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed = data_set.get_training_image(iteration, batch_size)
  feed_dict = {
      X_content: images_feed,
      is_training: False
  }
  return feed_dict



def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images for training, validation
  dataset = data_set(FLAGS.training_dir, FLAGS.style_image_path, FLAGS.test_image_dir, FLAGS.test_image_path, FLAGS.batch_size)
  style_shape = dataset.style_image.shape
  print style_shape


  style_activations = {}
  with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
      style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
      style_image_pre = vgg_model.preprocess(style_image)
      net = vgg_model.net(FLAGS.vgg_path, style_image_pre)
      for layer in STYLE_LAYERS:
          features = net[layer].eval(feed_dict={style_image: dataset.style_image})
          style_activations[layer] = loss_function.gram_matrix_ndarray(features)
  print np.sum(style_activations['relu1_1'])


  with tf.Graph().as_default(), tf.Session() as sess:
      X_content = tf.placeholder(tf.float32, shape=(1,256,256,3), name="X_content")
      is_training = tf.placeholder(dtype=tf.bool)

      # precompute content features
      content_vgg_image = vgg_model.preprocess(X_content)
      content_net = vgg_model.net(FLAGS.vgg_path, content_vgg_image)
      content_activations= {}
      content_activations[CONTENT_LAYER] = content_net[CONTENT_LAYER]


      # transferred_image = residualnet.net(X_content / 255.0)
      transferred_image = transfer_net(X_content / 255.0, is_training=is_training)

      transferred_image_vgg = vgg_model.preprocess(transferred_image)
      net = vgg_model.net(FLAGS.vgg_path, transferred_image_vgg)

      content_loss = loss_function.content_loss(content_activations[CONTENT_LAYER], net[CONTENT_LAYER])

      style_losses = []
      weight = 1.0
      for style_layer in STYLE_LAYERS:
          style_losses.append(loss_function.style_loss(style_activations[style_layer], net[style_layer], weight))
          weight -= 0.1
      style_loss =  tf.add_n(style_losses)

      tv_loss = loss_function.tv_loss(transferred_image)

      loss = content_loss + style_loss + tv_loss
      train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
      init = tf.global_variables_initializer()
      sess = tf.Session()
      sess.run(init)

      # Start the training loop.
      for iteration in xrange(FLAGS.iterations):
          start_time = time.time()

          feed_dict = {
              X_content: dataset.get_training_image(iteration, batch_size=1),
              is_training: True
          }

          # Run one step of the model.
          image, loss_value, content_loss_value, style_loss_value, tv_loss_value\
              = sess.run([transferred_image, loss, content_loss, style_loss, tv_loss], feed_dict=feed_dict)
          sess.run(train_op, feed_dict=feed_dict)

          # save test image
          image = image[0]
          image = image.reshape((256,256,3))
          img = np.clip(image, 0, 255).astype(np.uint8)
          scipy.misc.imsave('train.jpg', img)

          duration = time.time() - start_time
          print "loss: ", loss_value, "training time: ", duration
          print content_loss_value, style_loss_value, tv_loss_value

          if (iteration % 1000  == 0):
              saver = tf.train.Saver()
              saver.save(sess, 'my-model')
              print "save weights done"

              feed_dict = {
                  X_content: dataset.get_training_image(0, batch_size=1),
                  is_training: True
              }

              image, loss_value, content_loss_value, style_loss_value, tv_loss_value \
                  = sess.run([transferred_image, loss, content_loss, style_loss, tv_loss], feed_dict=feed_dict)

              # save test image
              image = image[0]
              image = image.reshape((256, 256, 3))
              img = np.clip(image, 0, 255).astype(np.uint8)
              scipy.misc.imsave('test'+str(iteration)+'.jpg', img)


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--training_dir',
      type=str,
      default='/home/ec2-user/train2014/',
      help='Directory to put the input data.'
  )

  parser.add_argument(
      '--style_image_path',
      type=str,
      default='./style_data/rain.jpg',
      help='Directory to put the log data.'
  )

  parser.add_argument(
      '--batch_size',
      type=int,
      default=1,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )

  parser.add_argument(
      '--test_image_path',
      type=str,
      default='./test_image/test.jpg',
      help='Path of test iamge'
  )

  parser.add_argument(
      '--test_image_dir',
      type=str,
      default='./test_image/output/',
      help='Directory to put the transfer result of test iamge'
  )

  parser.add_argument(
      '--iterations',
      type=int,
      default=160000,
      help='Number of iteations to run trainer.'
  )

  parser.add_argument(
      '--log_dir',
      type=str,
      default='./logs/',
      help='Directory to put the log data.'
  )

  parser.add_argument(
      '--vgg_path',
      type=str,
      default='./data/imagenet-vgg-verydeep-19.mat',
      help='Directory to vgg weight.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  print (FLAGS)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



