import sys

from distutils.core import setup
from setuptools import find_packages


# List of runtime dependencies required by this built package
install_requires = []
if sys.version_info <= (2, 7):
    install_requires += ['future', 'typing']
install_requires += ["numpy", "protobuf"]

setup(
    name="tfrecord",
    version="1.10",
    description="TFRecord reader",
    author="Vahid Kazemi",
    author_email="vkazemi@gmail.com",
    url="https://github.com/vahidk/tfrecord",
    packages=find_packages(),
    license="MIT",
    install_requires=install_requires
)
