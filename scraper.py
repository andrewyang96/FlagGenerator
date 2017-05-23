"""Scraper script."""

import argparse
import io
import os
from PIL import Image
import praw
import requests


def process_submission(sub, outfile_name):
    """Extract image from submission. If it exists, processes and saves it."""
    if not sub.url.endswith(('.jpg', '.jpeg', '.png')):
        print('Skipping', sub.url)
        return
    print('Processing', sub.url)

    try:
        file = io.BytesIO(requests.get(sub.url).content)
    except requests.exceptions.ConnectionError:
        print("Couldn't retrieve", sub.url)
        return
    im = Image.open(file).convert('RGB')

    width, height = im.size
    aspect_ratio = width / height
    im = im.resize(
        (int(aspect_ratio * FLAGS.output_height), FLAGS.output_height))
    im.save(os.path.join(FLAGS.output_dir, outfile_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--client_id', type=str,
        help="Client ID, retrieved from Reddit's developer console.")
    parser.add_argument(
        '--client_secret', type=str,
        help="Client secret, retrieved from Reddit's developer console.")
    parser.add_argument(
        '--user_agent', type=str, default='flag_generator_scraper',
        help='User agent for Reddit requests.')
    parser.add_argument(
        '--output_dir', type=str, default='data',
        help='Output directory for collected data.')
    parser.add_argument(
        '--output_height', type=int, default=180,
        help='Output image height. Width can be variable.')
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    r = praw.Reddit(
        client_id=FLAGS.client_id, client_secret=FLAGS.client_secret,
        user_agent=FLAGS.user_agent)
    oc_subs = r.subreddit('vexillology').search("flair:'OC'", limit=1000)
    count = 0
    for sub in oc_subs:
        # TODO: Don't skip Imgur images and albums
        process_submission(sub, '{0}.png'.format(count))
        count += 1
