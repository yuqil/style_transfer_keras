import math
import tensorflow as tf
import numpy as np

CHANNELS = 3
WIDTH = 256
HEIGHT = 256

# FIXED WEIGHT CANNOT CHANGE
CONTENT_WEIGHT = 10
STYLE_WEIGHT = 100
TV_WEIGHT = 200


def content_loss(content_activation, transferred_activation):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  return CONTENT_WEIGHT * (tf.nn.l2_loss(content_activation - transferred_activation) /  (WIDTH * HEIGHT * CHANNELS))


def gram_matrix(activation):
  shape = activation.get_shape()
  phi = tf.reshape(activation, [shape[0].value, shape[1].value * shape[2].value, shape[3].value])
  phi_t = tf.transpose(phi, perm=[0,2,1])
  gram = tf.matmul(phi_t, phi) / (shape[1].value * shape[2].value * shape[3].value)
  return gram, shape[0].value


def gram_matrix_ndarray(activation):
  shape = activation.shape
  phi = np.reshape(activation, [shape[0] * shape[1] * shape[2], shape[3]])
  phi_t = np.transpose(phi)
  gram = np.matmul(phi_t, phi) / (shape[1] * shape[2] * shape[3])
  return gram


def style_loss(style_activation, transferred_activation, weight = 1.0):
  gram_transferred, filters = gram_matrix(transferred_activation)
  print style_activation.shape
  print gram_transferred.get_shape()
  size = style_activation.size
  return weight * STYLE_WEIGHT * tf.nn.l2_loss(gram_transferred - style_activation) / size


def tv_loss(transferred_image):
  y_tv = tf.nn.l2_loss(transferred_image[:, 1:, :, :] - transferred_image[:, :WIDTH - 1, :, :])
  x_tv = tf.nn.l2_loss(transferred_image[:, :, 1:, :] - transferred_image[:, :, :HEIGHT - 1, :])
  loss = TV_WEIGHT * (x_tv + y_tv) / (CHANNELS * WIDTH * HEIGHT)
  return loss
