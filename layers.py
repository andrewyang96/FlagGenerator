"""Tensorflow layer helpers with variable summaries."""

import tensorflow as tf

def variable_summaries(var):
    """Attach summaries to a tensor for Tensorboard visualization."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def weight_variable(shape, stddev=0.1):
    """Make weight variable wrapper."""
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def bias_variable(shape, init_val=0.1):
    """Make bias variable wrapper."""
    return tf.Variable(tf.constant(init_val, shape=shape))


def fc_layer(x, W, b, act=tf.nn.relu):
    """Fully connected layer wrapper, with bias and activation.

    Also attaches histogram summary on preactivations.
    """
    with tf.name_scope('fc_layer'):
        x = tf.reshape(x, [-1, W.get_shape().as_list()[0]])
        with tf.name_scope('Wx_plus_b'):
            x = tf.add(tf.matmul(x, W), b)
            tf.summary.histogram('pre_activations', x)
    return act(x)
