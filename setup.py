#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io

from os.path import dirname
from os.path import join

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    file_path = join(dirname(__file__), *names)
    with io.open(file_path, encoding=kwargs.get('encoding', 'utf8')) as fh:
        return fh.read()


setup(
    name='nf',
    version='0.1.0',
    license='MIT license',
    description='Neue Fische - hh-2020-ds1',
    long_description=read('README.md'),
    author='first-name last-name',
    author_email='your-GitHub-email-address',
    url='GitHub-repo-url',
    zip_safe=False, install_requires=['pandas']
)
