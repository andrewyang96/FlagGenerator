"""Scraper script."""

import argparse
from datetime import datetime
from datetime import timezone
import io
import os
from PIL import Image
import praw
import pyimgur
import requests
from urllib.parse import urlparse


def download_image(url, outfile_name):
    """Download image into output_dir."""
    print('Downloading', url)
    try:
        file = io.BytesIO(requests.get(url).content)
    except requests.exceptions.ConnectionError:
        print("Couldn't download", url)
        return
    im = Image.open(file).convert('RGB')
    im = im.resize((FLAGS.output_width, FLAGS.output_height))
    im.save(os.path.join(FLAGS.output_dir, '{0}.png'.format(outfile_name)))


def process_imgur_submission(imgur, url, outfile_name):
    """Extract image(s) from Imgur submission."""
    if '/a/' in url:
        # is album
        print(url, 'is an Imgur album')
        album_id = urlparse(url).path.split('/')[-1]
        album = imgur.get_album(album_id)
        for idx, image in enumerate(album.images):
            download_image(image.link, '{0}-{1}'.format(outfile_name, idx))
    else:
        # is single image
        print(url, 'is an Imgur image')
        filename = urlparse(url).path.split('/')[-1]
        download_image('http://i.imgur.com/{0}.png'.format(filename), outfile_name)


def process_submission(imgur, sub, outfile_name):
    """Extract image from submission. If it exists, processes and saves it."""
    if '/imgur.com/' in sub.url:
        # printing sub.url can cause a UnicodeEncodeError or UnicodeDecdeError
        return process_imgur_submission(imgur, sub.url, outfile_name)
    if not sub.url.endswith(('.jpg', '.jpeg', '.png')):
        print('Skipping', sub.url)
        return
    download_image(sub.url, outfile_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reddit_client_id', type=str,
        help='Reddit client ID.')
    parser.add_argument(
        '--reddit_client_secret', type=str,
        help='Reddit client secret.')
    parser.add_argument(
        '--imgur_client_id', type=str,
        help='Imgur client ID.')
    parser.add_argument(
        '--user_agent', type=str, default='flag_generator_scraper',
        help='User agent for Reddit requests.')
    parser.add_argument(
        '--output_dir', type=str, default='data',
        help='Output directory for collected data.')
    parser.add_argument(
        '--output_width', type=int, default=48,
        help='Output image width.')
    parser.add_argument(
        '--output_height', type=int, default=32,
        help='Output image height.')
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    r = praw.Reddit(
        client_id=FLAGS.reddit_client_id,
        client_secret=FLAGS.reddit_client_secret,
        user_agent=FLAGS.user_agent)
    imgur = pyimgur.Imgur(FLAGS.imgur_client_id)

    start_timestamp = 1279756800 # July 22, 2010
    end_timestamp = datetime.utcnow().timestamp()
    num_iterations = 100

    delta = (end_timestamp - start_timestamp) / num_iterations
    vexillology = r.subreddit('vexillology')
    count = 0

    for i in range(num_iterations):
        start_range = int(start_timestamp + i * delta)
        end_range = int(start_timestamp + (i + 1) * delta)
        print('Iteration {0}/{1}: Processing between {2} and {3}'.format(
            i + 1, num_iterations, datetime.utcfromtimestamp(start_range),
            datetime.utcfromtimestamp(end_range)))
        oc_subs = vexillology.search(
            "(and flair:'OC' timestamp:{0}..{1})".format(
                start_range, end_range), syntax='cloudsearch', limit=None)
        for sub in oc_subs:
            try:
                process_submission(imgur, sub, count)
            except OSError as e:
                print('Skipping due to error:', e)
            count += 1
        print('Count:', count)
