import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_init_vars(x, filter_size, out_channels, transpoes=False):
    _, rows, cols, in_channels = [i.value for i in x.get_shape()]
    print (_, rows, cols, in_channels)
    if not transpoes:
        W = weight_variable([filter_size, filter_size, in_channels, out_channels])
        b = bias_variable([out_channels])
    else:
        W = weight_variable([filter_size, filter_size, out_channels, in_channels])
        b = bias_variable([in_channels])
    return W, b

def conv_layer(x, filter_size, out_channels, stride, relu=True, is_training=True):
    W, b = conv_init_vars(x, filter_size, out_channels)
    stride_shape = [1, stride, stride, 1]
    print('before', x.get_shape())
    x = batch_norm(
            tf.nn.conv2d(x, W, strides=stride_shape, padding='SAME') + b,
            is_training
        )
    print('end', x.get_shape())
    if relu:
        x = tf.nn.relu(x)
    return x

def tranpose_conv(x, filter_size, out_channels, stride):
    W, b = conv_init_vars(x=x, filter_size=filter_size, out_channels=out_channels, transpoes=True)
    batch_size, rows, cols, in_channels = [i.value for i in x.get_shape()]
    new_rows, new_cols = int(rows * stride), int(cols * stride)

    #new_shape = [batch_size, new_rows, new_cols, out_channels]
    new_shape = [tf.shape(x)[0], new_rows, new_cols, out_channels]

    print ('new_shape = ', new_shape)
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, stride, stride, 1]

    print('before ', x.get_shape())
    deconv = tf.nn.conv2d_transpose(x, W, tf_shape, strides_shape, padding='SAME')
    print('after', deconv.get_shape())

    return tf.nn.relu(deconv)

def batch_norm(x, is_training):
    x_shape = x.get_shape().as_list()

    gamma = tf.Variable(tf.truncated_normal(shape=[x_shape[-1]]))
    beta  = tf.Variable(tf.constant(0., shape=[x_shape[-1]]))
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def residual_block(x, filter_size=3, is_training=True):
    shortcut = x
    x = conv_layer(x, filter_size=filter_size, out_channels=128, stride=1, relu=True, is_training=is_training)
    x = conv_layer(x, filter_size=filter_size, out_channels=128, stride=1, relu=False, is_training=is_training)
    return shortcut + x

# working version - transfer net for MNIST
# def transfer_net(x_image, is_training=True):
#     h_conv1   = conv_layer(x_image, filter_size=9, out_channels=32, stride=1, is_training=is_training)
#     h_conv2   = conv_layer(h_conv1, filter_size=3, out_channels=64, stride=2, is_training=is_training)
#     h_conv3   = conv_layer(h_conv2, filter_size=3, out_channels=128, stride=2, is_training=is_training)
#     h_res1    = residual_block(h_conv3, filter_size=3, is_training=is_training)
#     h_res2    = residual_block(h_res1, filter_size=3, is_training=is_training)
#     h_res3    = residual_block(h_res2, filter_size=3, is_training=is_training)
#     h_res4    = residual_block(h_res3, filter_size=3, is_training=is_training)
#     h_res5    = residual_block(h_res4, filter_size=3, is_training=is_training)
#     h_conv_t1 = tranpose_conv(h_res5, filter_size=3, out_channels=64, stride=2)
#     h_conv_t1 = tf.reshape(h_conv_t1, [-1, 14, 14, 64])
#     h_conv_t2 = tranpose_conv(h_conv_t1, filter_size=3, out_channels=32, stride=2)
#     h_conv_t2 = tf.reshape(h_conv_t2, [-1, 28, 28, 32])
#     h_conv4   = conv_layer(h_conv_t2, filter_size=9, out_channels=3, stride=1, is_training=is_training)
#
#     return h_conv4

def transfer_net(x_image, is_training=True):
    h_conv1   = conv_layer(x_image, filter_size=9, out_channels=32, stride=1, is_training=is_training)
    h_conv2   = conv_layer(h_conv1, filter_size=3, out_channels=64, stride=2, is_training=is_training)
    h_conv3   = conv_layer(h_conv2, filter_size=3, out_channels=128, stride=2, is_training=is_training)
    h_res1    = residual_block(h_conv3, filter_size=3, is_training=is_training)
    h_res2    = residual_block(h_res1, filter_size=3, is_training=is_training)
    h_res3    = residual_block(h_res2, filter_size=3, is_training=is_training)
    h_res4    = residual_block(h_res3, filter_size=3, is_training=is_training)
    h_res5    = residual_block(h_res4, filter_size=3, is_training=is_training)
    h_conv_t1 = tranpose_conv(h_res5, filter_size=3, out_channels=64, stride=2)
    h_conv_t1 = tf.reshape(h_conv_t1, [-1, 128, 128, 64])
    h_conv_t2 = tranpose_conv(h_conv_t1, filter_size=3, out_channels=32, stride=2)
    h_conv_t2 = tf.reshape(h_conv_t2, [-1, 256, 256, 32])
    h_conv4   = conv_layer(h_conv_t2, filter_size=9, out_channels=3, stride=1, is_training=is_training)
    output    = tf.nn.tanh(h_conv4) * 150 + 255./2

    return output


def dense_layer(x, ouput_size, out_channels, keep_prob):
    W_fc1 = weight_variable([ouput_size*ouput_size*out_channels, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(x, [-1, ouput_size*ouput_size*out_channels])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv

def main(_):
    # import data
    mnist = input_data.read_data_sets(FLAGS.data_dir ,one_hot=True)

    # Create the model
    BATCH_SIZE = 50
    is_training = tf.placeholder(dtype=tf.bool)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv_t2 = transfer_net(x_image, is_training)
    #
    keep_prob = tf.placeholder(tf.float32)
    y_conv = dense_layer(h_conv_t2, 28, 3, keep_prob)

    # Define loss and optimizer
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_)
    )
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss=cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    batches = []
    for i in range(100000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, is_training: True})

        if i % 100 == 0:
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            train_accuracy = accuracy.eval(
                feed_dict={
                    x:batch[0],
                    y_: batch[1],
                    keep_prob: 1.0,
                    is_training: False,
                }
            )
            #train_acc /= 50
            #batches.clear()
            print("step %d, training accuracy %g" % (i, train_accuracy))
            if i != 0:
                print ('start testing')
                test_accuracy = accuracy.eval(
                    feed_dict={
                        x: mnist.test.images,
                        y_: mnist.test.labels,
                        keep_prob: 1.0,
                        is_training: False,
                    }
                )
                print("test accuracy %g" % test_accuracy)
        '''
        test_acc = 0.
        for j in range(10000):
            test_accuracy = accuracy.eval(
                feed_dict={
                    x: mnist.test.images[j],
                    y_: mnist.test.labels[j],
                    keep_prob: 1.0,
                    is_training: False,
                }
            )
            test_acc += test_accuracy
        test_acc /= 10000
        print("test accuracy %g" % test_acc)
        '''
    # Test
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
