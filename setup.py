#!/usr/bin/env python3

import os
import re
import codecs

from setuptools import setup

# Package meta-data.
NAME = 'neo-pyzm'
DESCRIPTION = 'baudneo-FORKED: ZoneMinder API, Logger and other base utilities for python programmers'
URL = 'https://github.com/baudneo/pyzm/'
AUTHOR_EMAIL = 'baudneo@protonmail.com'
AUTHOR = 'Pliable Pixels forked by baudneo'
LICENSE = 'GPL'
INSTALL_REQUIRES = [
    'pyyaml',
    'requests_toolbelt',
    'SQLAlchemy>=1.3.20,<1.4.0',
    'mysql-connector-python>=8.0.16',
    'requests>=2.18.4',
    'dateparser>=1.0.0',
    'websocket-client>=0.57.0',
    'progressbar2 >=3.53.1',
    'portalocker>=2.3.0',
    'imutils >=0.5.3',
    'Shapely >=1.7.0',
    'numpy >=1.13.3',
    'Pillow',
    'boto3',
    'psutil >=5.7.3',
    'python-dotenv',
    'scikit-learn',
    'imageio',
    'imageio-ffmpeg',
    'pygifsicle',
    'configupdater',
    'paho-mqtt >= 1.5.1',
    'cryptography',
]

here = os.path.abspath(os.path.dirname(__file__))
# read the contents of your README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        data = fp.read()
    return data


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name=NAME,
      python_requires='>=3.6',
      version=find_version('pyzm', '__init__.py'),
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      url=URL,
      license=LICENSE,
      install_requires=INSTALL_REQUIRES,
      py_modules=[
          # Helpers
          'pyzm.helpers.Monitors',
          'pyzm.helpers.Monitor',
          'pyzm.helpers.Events',
          'pyzm.helpers.Event',
          'pyzm.helpers.States',
          'pyzm.helpers.State',
          'pyzm.helpers.Zones',
          'pyzm.helpers.Zone',
          'pyzm.helpers.Configs',
          'pyzm.helpers.Media',
          'pyzm.helpers.globals',
          'pyzm.helpers.mqtt',
          'pyzm.helpers.pyzm_utils',
          'pyzm.helpers.new_yaml',
          'pyzm.helpers.mlapi_db',

          # Base ZMES
          'pyzm.api',
          'pyzm.ZMLog',
          'pyzm.ZMEventNotification',
          'pyzm.ZMMemory',
          'pyzm.interface',

          # ML
          'pyzm.ml.alpr',
          'pyzm.ml.face',
          'pyzm.ml.face_dlib',
          'pyzm.ml.face_tpu',
          'pyzm.ml.face_train_dlib',
          'pyzm.ml.object',
          'pyzm.ml.coral_edgetpu',
          'pyzm.ml.hog',
          'pyzm.ml.yolo',
          'pyzm.ml.aws_rekognition',
          'pyzm.ml.detect_sequence',
      ]
      )
