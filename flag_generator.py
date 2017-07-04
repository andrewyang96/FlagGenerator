"""Generate flags from trained model."""

import argparse
import numpy as np
import os
from PIL import Image
import tensorflow as tf
import time

MODEL_PATH = '/tmp/flag_generator_vae.ckpt'
INPUT_WIDTH = 48
INPUT_HEIGHT = 32
NUM_CHANNELS = 3

pixel_value_func = np.vectorize(lambda value: round(value * 255))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_images', type=int, default=100,
        help='Number of images to generate.')
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.exists('generated_flags'):
        os.mkdir('generated_flags')

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(MODEL_PATH + '.meta')
        saver.restore(sess, MODEL_PATH)

        latent_layer = sess.graph.get_collection('latent_layer')[0]
        output_tensor = sess.graph.get_collection('output_tensor')[0]

        for i in range(FLAGS.num_images):
            print('Generating flag', i + 1, 'of', FLAGS.num_images)
            feed_dict = {latent_layer: np.random.normal(0, 1, (1, latent_layer.get_shape()[1]))}
            output = sess.run([output_tensor], feed_dict=feed_dict)[0]
            output = pixel_value_func(output).reshape((INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNELS))
            # print(output.max(), output.min(), output.mean())
            im = Image.fromarray(output, mode='RGB')
            filename = os.path.join('generated_flags', '{0}.png'.format(i))
            im.save(filename)
