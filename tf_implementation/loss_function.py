import math
import tensorflow as tf


CHANNELS = 3
WIDTH = 256
HEIGHT = 256

# FIXED WEIGHT CANNOT CHANGE
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 0.01
TV_WEIGHT = 1


def content_loss(content_activation, transferred_activation):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  # return CONTENT_WEIGHT * tf.reduce_sum((content_activation - transferred_activation) ** 2) / (WIDTH * HEIGHT * CHANNELS)
  return CONTENT_WEIGHT * (2 * tf.nn.l2_loss(content_activation - transferred_activation) /  (WIDTH * HEIGHT * CHANNELS))

def gram_matrix(activation):
  shape = activation.get_shape()
  phi = tf.reshape(activation, [shape[0].value, shape[1].value * shape[2].value, shape[3].value])
  phi_t = tf.transpose(phi, perm=[0,2,1])
  gram = tf.matmul(phi_t, phi) / (shape[1].value * shape[2].value * shape[3].value)
  return gram, shape[0].value



def style_loss(style_activation, transferred_activation):
  # gram_style, filters = gram_matrix(style_activation)
  # gram_transferred, filters = gram_matrix(transferred_activation)
  # return STYLE_WEIGHT * tf.reduce_sum((gram_style - gram_transferred) ** 2) / (filters ** 2)
  bs, height, width, filters = map(lambda i: i.value, transferred_activation.get_shape())

  size = height * width * filters
  feats = tf.reshape(transferred_activation, (bs, height * width, filters))
  feats_T = tf.transpose(feats, perm=[0, 2, 1])
  grams = tf.matmul(feats_T, feats) / size

  feats1 = tf.reshape(style_activation, (bs, height * width, filters))
  feats_T1 = tf.transpose(feats, perm=[0, 2, 1])
  grams1 = tf.matmul(feats_T1, feats1) / size
  print grams.get_shape()
  return (2 * tf.nn.l2_loss(grams - grams1) / (filters * filters))


def tv_loss(transferred_image):
  y_tv = tf.nn.l2_loss(transferred_image[:, 1:, :, :] - transferred_image[:, :WIDTH - 1, :, :])
  x_tv = tf.nn.l2_loss(transferred_image[:, :, 1:, :] - transferred_image[:, :, :HEIGHT - 1, :])
  loss = TV_WEIGHT * (x_tv + y_tv) / (CHANNELS * WIDTH * HEIGHT)
  return loss


