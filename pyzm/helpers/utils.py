"""
utils
======
Set of utility functions
"""

from pyzm.helpers.Base import ConsoleLog
from configparser import ConfigParser

def read_config(file):
    config_file = ConfigParser(interpolation=None)
    config_file.read(file)
    return config_file

def get(key=None, section=None, conf=None):
    if conf.has_option(section, key):
        return conf.get(section, key)
    else:
        return None