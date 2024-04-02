import gzip
import pickle
import os
import torch


def save_gzip_pickle(data, filename):
    """
    Save the given data using gzip/pickle.
    """
    with gzip.open(filename, "w") as f:
        pickle.dump(data, f)


def load_gzip_pickle(filename):
    """
    Load from file using gzip/pickle.
    """
    with gzip.open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def load_pickle(filename):
    """
    Load from file using pickle.
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def make_dir(directory):
    """
    If the provided directory does not exist, make it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)