import argparse
import tensorflow as tf
import time
import sys
import cv2
import numpy as np
import scipy

import process_data
import residualnet
import vgg_model
import loss_function
from process_data import data_set

# Basic model parameters as external flags.
FLAGS = None
CHANNELS = 3
WIDTH = 256
HEIGHT = 256
LEARNING_RATE = 0.001

CONTENT_LAYER = 'relu2_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


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
  content_placeholder = tf.placeholder(tf.float32, shape=(batch_size,  WIDTH, HEIGHT, CHANNELS))
  style_placeholder = tf.placeholder(tf.float32, shape=(1, WIDTH, HEIGHT, CHANNELS))
  return content_placeholder, style_placeholder


def fill_feed_dict(data_set, content_placeholder, style_placeholder, iteration, batch_size):
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
  style_feed = data_set.style_image

  feed_dict = {
      style_placeholder: style_feed,
      content_placeholder: images_feed,
  }
  return feed_dict



def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images for training, validation
  dataset = data_set(FLAGS.training_dir, FLAGS.style_image_path, FLAGS.test_image_dir, FLAGS.test_image_path)

  with tf.Graph().as_default():
      # get input for calculationi
      content_placeholder, style_placeholder = placeholder_inputs(FLAGS.batch_size)

      transferred_image_result = residualnet.net(content_placeholder/255.0)

      # get vgg activation
      transferred_image = vgg_model.preprocess(transferred_image_result)
      content_vgg_input = vgg_model.preprocess(content_placeholder)
      style_vgg_input= vgg_model.preprocess(style_placeholder)
      content_activation_dict = vgg_model.net(FLAGS.vgg_path, content_vgg_input)
      style_activation_dict = vgg_model.net(FLAGS.vgg_path, style_vgg_input)
      transferred_activation_dict = vgg_model.net(FLAGS.vgg_path, transferred_image)

      # get content loss
      content_activation = content_activation_dict[CONTENT_LAYER]
      transferred_activation = transferred_activation_dict[CONTENT_LAYER]
      content_loss = loss_function.content_loss(content_activation, transferred_activation)

      # get style loss
      style_losses = []
      for layer in STYLE_LAYERS:
          style_activation = style_activation_dict[layer]
          transferred_activation = transferred_activation_dict[layer]
          style_losses.append(loss_function.style_loss(style_activation, transferred_activation))
      style_loss = tf.add_n(style_losses, name=None)

      # get tv loss
      tv_loss = loss_function.tv_loss(transferred_image)

      loss = content_loss + style_loss + tv_loss

      train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
      init = tf.global_variables_initializer()
      sess = tf.Session()
      sess.run(init)

      # Start the training loop.
      for iteration in xrange(FLAGS.iterations):
          start_time = time.time()

          # Fill a feed dictionary with the actual set of images
          feed_dict = fill_feed_dict(dataset, content_placeholder, style_placeholder, iteration, FLAGS.batch_size)

          # Run one step of the model.
          image, loss_value, content_loss_value, style_loss_value, tv_loss_value\
              = sess.run([transferred_image_result, loss, content_loss, style_loss, tv_loss], feed_dict=feed_dict)
          sess.run(train_op, feed_dict=feed_dict)

          # save test image
          image = image.reshape((256,256,3))
          img = np.clip(image, 0, 255).astype(np.uint8)
          scipy.misc.imsave('test.jpg', img)

          duration = time.time() - start_time
          print "loss: ", loss_value, "training time: ", duration
          print content_loss_value, style_loss_value, tv_loss_value


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
      default='./training_data/',
      help='Directory to put the input data.'
  )

  parser.add_argument(
      '--style_image_path',
      type=str,
      default='./style_data/style.jpg',
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
      default=1000,
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
