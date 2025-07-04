
""" 
# read config file
from config.config import config
cfg = config()
"""

import tomllib as toml

config_path = "config/config.toml"

def config(path=config_path):
    """
    Reads a TOML file and returns its contents as a dictionary.
    """
    with open(path, "rb") as f:
        return toml.load(f)
