"""Scraper script."""

import argparse
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
        '--output_width', type=int, default=240,
        help='Output image width.')
    parser.add_argument(
        '--output_height', type=int, default=180,
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
    oc_subs = r.subreddit('vexillology').search("flair:'OC'", limit=1000)
    count = 0
    for sub in oc_subs:
        process_submission(imgur, sub, count)
        count += 1
