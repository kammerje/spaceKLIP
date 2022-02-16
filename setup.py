#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    REQUIRES = f.read().splitlines()


setup(name='pyKLIP_for_Webb',
      version='0.0.1',
      description='Reduction pipeline for JWST Coronagraphy',
      author='ERS-1386 Collaboration',
      author_email='aarynn.carter@gmail.com',
      license='MIT',
      install_requires=REQUIRES,
      packages=find_packages()
    )