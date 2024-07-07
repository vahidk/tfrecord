import os
import sys

from distutils.core import setup
from setuptools import find_packages


# List of runtime dependencies required by this built package
install_requires = []
if sys.version_info <= (2, 7):
    install_requires += ['future', 'typing']
install_requires += ['numpy', 'protobuf', 'crc32c']

# read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    name='tfrecord',
    version='1.14.5',
    description='TFRecord reader',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Vahid Kazemi',
    author_email='vkazemi@gmail.com',
    url='https://github.com/vahidk/tfrecord',
    packages=find_packages(),
    license='MIT',
    install_requires=install_requires,
    extras_require={
        'torch': ['torch'],
    },
    test_suite='tests',
)
