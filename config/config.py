""" 
# read config file
from config.config import config
cfg = config()
"""

import tomllib as toml
import os

def config():
    """
    Reads all TOML files in the config directory and merges their contents into a single dictionary.
    Later files override earlier ones for overlapping keys.
    """
    config_dir = os.path.dirname(__file__)
    config_dict = {}
    for fname in os.listdir(config_dir):
        if fname.endswith('.toml'):
            fpath = os.path.join(config_dir, fname)
            with open(fpath, "rb") as f:
                data = toml.load(f)
                config_dict.update(data)
    return config_dict
