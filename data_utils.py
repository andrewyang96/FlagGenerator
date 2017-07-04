"""Functions for loading and handling data."""

import glob
from PIL import Image
import numpy as np

def load_data(data_dir, input_size, norm_func):
    """Load data into an array of 3-D numpy arrays."""
    image_files = glob.glob('{0}/*.png'.format(data_dir))
    data = np.zeros((len(image_files), input_size), dtype=np.float32)
    for idx, image_file in enumerate(image_files):
        im = Image.open(image_file).convert('RGB')
        data[idx] = norm_func(np.array(im).flatten())
    return data


def partition_data(data, test_split=0.2):
    """Partition data into training and testing split.

    Returns (train data, test data).
    """
    idxs = np.arange(data.shape[0])
    np.random.shuffle(idxs)
    boundary_idx = int(data.shape[0] * test_split)
    return data[idxs[boundary_idx:]], data[idxs[:boundary_idx]]


def select_random_rows(data, num_rows):
    """Select and return random rows from data without replacement."""
    idxs = np.random.choice(data.shape[0], num_rows, replace=False)
    return data[idxs]
