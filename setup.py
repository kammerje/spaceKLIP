#!/usr/bin/env python

from setuptools import setup, find_packages
import os
import re

def resource(*args):
    return os.path.join(os.path.abspath(os.path.join(__file__, os.pardir)),
                        *args)

with open('requirements.txt') as f:
    REQUIRES = f.read().splitlines()
    
with open(resource('spaceKLIP', '__init__.py')) as version_file:
    version_file = version_file.read()
    VERSION = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                        version_file, re.M)
    VERSION = VERSION.group(1)
    
setup(name='spaceKLIP',
      version=VERSION,
      description='Reduction pipeline for JWST Coronagraphy',
      author='ERS-1386 Collaboration',
      author_email='aarynn.carter@gmail.com',
      license='MIT',
      install_requires=REQUIRES,
      packages=find_packages()
     )
