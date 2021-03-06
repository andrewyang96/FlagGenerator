"""The variational autoencoder model."""

import argparse
from data_utils import load_data
from data_utils import partition_data
from data_utils import select_random_rows
from PIL import Image
from layers import bias_variable
from layers import fc_layer
from layers import variable_summaries
from layers import weight_variable
import numpy as np
import sys
import tensorflow as tf
# from tensorflow.python import debug as tf_debug

# parameters
DISPLAY_STEP = 10
SUMMARY_STEP = 100
MODEL_PATH = '/tmp/flag_generator_vae.ckpt'

# network parameters
INPUT_WIDTH = 48
INPUT_HEIGHT = 32
NUM_CHANNELS = 3


def has_inf_or_nan(datum, tensor):
    return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))


def encoder(x):
    """Create encoder given placeholder input tensor."""
    # Encoding layer 1
    with tf.name_scope('encoder1'):
        with tf.name_scope('weights'):
            weights1 = weight_variable(
                [INPUT_WIDTH * INPUT_HEIGHT * NUM_CHANNELS, 2048], stddev=0.01)
            variable_summaries(weights1)
        with tf.name_scope('biases'):
            biases1 = bias_variable([2048], init_val=0.01)
        layer1 = fc_layer(x, weights1, biases1)

    # Encoding layer 2
    with tf.name_scope('encoder2'):
        with tf.name_scope('weights'):
            weights2 = weight_variable([2048, 512], stddev=0.01)
            variable_summaries(weights1)
        with tf.name_scope('biases'):
            biases2 = bias_variable([512], init_val=0.01)
        layer2 = fc_layer(layer1, weights2, biases2)

    # Mu encoder layer
    with tf.name_scope('mu_encoder'):
        with tf.name_scope('weights'):
            weights_mu = weight_variable([512, 64], stddev=0.1)
            variable_summaries(weights_mu)
        with tf.name_scope('biases'):
            biases_mu = bias_variable([64], init_val=0.1)
        mu_encoder = fc_layer(layer2, weights_mu, biases_mu)

    # Log(sigma) encoder layer
    with tf.name_scope('log_sigma_encoder'):
        with tf.name_scope('weights'):
            weights_log_sigma = weight_variable([512, 64], stddev=0.1)
            variable_summaries(weights_log_sigma)
        with tf.name_scope('biases'):
            biases_log_sigma = bias_variable([64], init_val=0.1)
        log_sigma_encoder = fc_layer(
            layer2, weights_log_sigma, biases_log_sigma)

    # Sample epsilon, a truncated normal tensor
    epsilon = tf.truncated_normal(tf.shape(log_sigma_encoder))

    # Sample latent variables
    with tf.name_scope('latent_layer'):
        std_encoder = tf.exp(log_sigma_encoder)
        z = tf.add(mu_encoder, tf.multiply(std_encoder, epsilon))
        variable_summaries(z)
    return mu_encoder, log_sigma_encoder, epsilon, z


def decoder(x):
    """Create decoder given placeholder input tensor."""
    # Decoding layer 1
    with tf.name_scope('decoder1'):
        with tf.name_scope('weights'):
            weights1 = weight_variable([64, 512], stddev=0.1)
            variable_summaries(weights1)
        with tf.name_scope('biases'):
            biases1 = bias_variable([512], init_val=0.1)
        layer1 = fc_layer(x, weights1, biases1)

    # Decoding layer 2
    with tf.name_scope('decoder2'):
        with tf.name_scope('weights'):
            weights2 = weight_variable([512, 2048], stddev=0.01)
            variable_summaries(weights1)
        with tf.name_scope('biases'):
            biases2 = bias_variable([2048], init_val=0.01)
        layer2 = fc_layer(layer1, weights2, biases2)

    # Decoding layer 3
    with tf.name_scope('decoder3'):
        with tf.name_scope('weights'):
            weights3 = weight_variable(
                [2048, INPUT_WIDTH * INPUT_HEIGHT * NUM_CHANNELS], stddev=0.01)
            variable_summaries(weights2)
        with tf.name_scope('biases'):
            biases3 = bias_variable(
                [INPUT_WIDTH * INPUT_HEIGHT * NUM_CHANNELS], init_val=0.01)
        layer3 = fc_layer(layer2, weights3, biases3)
    return layer3


def get_feed_dict(x, train_data, test_data, is_train=True):
    """Make the feed_dict dictionary for sess.run given placeholder."""
    if is_train:
        return {x: select_random_rows(train_data, num_rows=32)}
    else:
        return {x: test_data}


def train():
    """Train method."""
    norm_func = np.vectorize(lambda value: value / 255)
    data = load_data(
        data_dir=FLAGS.data_dir,
        input_size=INPUT_WIDTH * INPUT_HEIGHT * NUM_CHANNELS,
        norm_func=norm_func)
    train_data, test_data = partition_data(data)
    sess = tf.InteractiveSession()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # Tensorflow graph inputs
    x = tf.placeholder(
        tf.float32, [None, INPUT_WIDTH * INPUT_HEIGHT * NUM_CHANNELS])

    mu_layer, log_sigma_layer, epsilon_layer, latent_layer = encoder(x)
    reconstructed_layer = decoder(latent_layer)

    with tf.name_scope('loss'):
        cost = tf.reduce_mean(tf.pow(reconstructed_layer - x, 2))
    tf.summary.scalar('loss', cost)

    with tf.name_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate, global_step, FLAGS.decay_steps,
            FLAGS.decay_rate, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        train_step = tf.train.RMSPropOptimizer(
            learning_rate=FLAGS.learning_rate).minimize(
            cost, global_step=global_step)

    # Merge all summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    # Run train iterations
    for i in range(FLAGS.max_steps):
        if i % DISPLAY_STEP == 0:  # Record summaries and test-set loss
            summary, loss = sess.run(
                [merged, cost], feed_dict=get_feed_dict(
                    x, train_data, test_data, is_train=False))
            test_writer.add_summary(summary, i)
            print('Loss at step {0}: {1}'.format(i, loss))
        else:  # Train and ecord train set summaries
            if i % SUMMARY_STEP == SUMMARY_STEP - 1:
                # Record execution stats
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run(
                    [merged, train_step], feed_dict=get_feed_dict(
                        x, train_data, test_data, is_train=True),
                    options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:
                # Record a summary
                summary, _ = sess.run(
                    [merged, train_step], feed_dict=get_feed_dict(
                        x, train_data, test_data, is_train=True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()

    # Save model
    sess.graph.add_to_collection('x', x)
    sess.graph.add_to_collection('mu_layer', mu_layer)
    sess.graph.add_to_collection('log_sigma_layer', log_sigma_layer)
    sess.graph.add_to_collection('epsilon_layer', epsilon_layer)
    sess.graph.add_to_collection('latent_layer', latent_layer)
    sess.graph.add_to_collection('output_tensor', reconstructed_layer)
    sess.graph.add_to_collection('loss_tensor', cost)
    saver = tf.train.Saver()
    saver.save(sess, MODEL_PATH)
    print('Model saved in', MODEL_PATH)


def main(_):
    """Main method. Helps clean up old log files."""
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_steps', type=int, default=2000,
        help='Number of steps to run trainer.')
    parser.add_argument(
        '--learning_rate', type=float, default=0.0001,
        help='Initial learning rate')
    parser.add_argument(
        '--decay_rate', type=float, default=0.95,
        help='Learning rate decay')
    parser.add_argument(
        '--decay_steps', type=float, default=500,
        help='Apply learning_decay every decay_steps.')
    parser.add_argument(
        '--dropout', type=float, default=0.9,
        help='Keep probability for training dropout.')
    parser.add_argument(
        '--data_dir', type=str, default='data',
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir', type=str,
        default='/tmp/tensorflow/mnist/logs/flag_generator_vae',
        help='Summaries log directory')
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
