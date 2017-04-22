import math
import tensorflow as tf


CHANNELS = 3
WIDTH = 256
HEIGHT = 256

# FIXED WEIGHT CANNOT CHANGE
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 0.05
TV_WEIGHT = 0.01


def content_loss(content_activation, transferred_activation):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  square_error = tf.square(tf.subtract(content_activation, transferred_activation, name=None))
  return CONTENT_WEIGHT * tf.reduce_mean(square_error, name='content_loss')


def gram_matrix(activation):
  shape = tf.shape(activation)
  phi = tf.reshape(activation, [shape[0], shape[1] * shape[2], shape[3]])
  phi_t = tf.transpose(phi, perm=[0,2,1])
  gram = tf.matmul(phi_t, phi) / WIDTH / HEIGHT / CHANNELS
  return gram


def style_loss(style_activation, transferred_activation):
  gram_style = gram_matrix(style_activation)
  gram_transferred = gram_matrix(transferred_activation)

  square_error = tf.square(tf.subtract(gram_style, gram_transferred, name=None))
  return STYLE_WEIGHT * tf.reduce_mean(square_error, name='content_loss')



def tv_loss(transferred_image):
  y_tv = tf.nn.l2_loss(transferred_image[:, 1:, :, :] - transferred_image[:, :WIDTH - 1, :, :])
  x_tv = tf.nn.l2_loss(transferred_image[:, :, 1:, :] - transferred_image[:, :, :HEIGHT - 1, :])
  loss = TV_WEIGHT * (x_tv + y_tv) / (CHANNELS * WIDTH * HEIGHT)
  return loss


