#!/usr/bin/env python
from distutils.core import setup, Command

PACKAGE_NAME = 'ci'

requires = [
  'feedparser==5.2.0',
  'pillow',
]

setup(name=PACKAGE_NAME,
      version="0.0.1",
      description='Collection of scripts used for Collective Intelligence',
      author='Ben Emery',
      url='https://github.com/benemery/%s' % PACKAGE_NAME,
      packages=[PACKAGE_NAME, ],
      install_requires=requires,
)