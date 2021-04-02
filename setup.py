#!/usr/bin/python3

import io
import os
import re
import codecs

from setuptools import setup

#Package meta-data.
NAME = 'pyzm'
DESCRIPTION = 'ZoneMinder API, Logger and other base utilities for python programmers'
URL = 'https://github.com/pliablepixels/pyzm/'
AUTHOR_EMAIL = 'pliablepixels@gmail.com'
AUTHOR = 'Pliable Pixels'
LICENSE = 'GPL'
INSTALL_REQUIRES=[
    'SQLAlchemy>=1.3.20,<1.4.0', 
    'mysql-connector-python>=8.0.16', 
    'requests>=2.18.4', 
    'dateparser>=1.0.0',
    'websocket-client>=0.57.0',
    'progressbar2 >=3.53.1',
    'portalocker>=2.0.0',
    'imutils >=0.5.3',
    'Shapely >=1.7.0',
    'numpy >=1.13.3',
    'Pillow',
    'psutil >=5.7.3'
    
    ]


here = os.path.abspath(os.path.dirname(__file__))
# read the contents of your README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    f.close()

def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        data = fp.read()
        fp.close()
        return data

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(name = NAME,
      python_requires='>=3.0.0',
      version = find_version('pyzm','__init__.py'),
      description = DESCRIPTION,
      long_description = long_description,
      long_description_content_type='text/markdown',
      author = AUTHOR,
      author_email = AUTHOR_EMAIL,
      url = URL,
      license = LICENSE,
      install_requires=INSTALL_REQUIRES,
      py_modules = [
                    'pyzm.api',
                    'pyzm.helpers.Base',
                    'pyzm.helpers.Monitors',
                    'pyzm.helpers.Monitor',
                    'pyzm.helpers.Events',
                    'pyzm.helpers.Event',
                    'pyzm.helpers.States',
                    'pyzm.helpers.State',
                    'pyzm.helpers.Configs',
                    'pyzm.helpers.Media',
                    'pyzm.helpers.utils',
                    'pyzm.helpers.globals',

                    'pyzm.ZMLog',
                    'pyzm.ZMEventNotification',
                    'pyzm.ZMMemory',

                    'pyzm.ml.alpr',
                    'pyzm.ml.face',
                    'pyzm.ml.face_dlib',
                    'pyzm.ml.face_tpu',
                    'pyzm.ml.face_train_dlib',
                    'pyzm.ml.object',
                    'pyzm.ml.coral_edgetpu',
                    'pyzm.ml.hog',
                    'pyzm.ml.yolo',
                    'pyzm.ml.detect_sequence'
                    ]
      )

