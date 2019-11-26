#!/usr/bin/env python

from distutils.core import setup

setup(name="tfrecord",
      version="1.0.1",
      description="TFRecord reader.",
      author="Vahid Kazemi",
      author_email="vkazemi@gmail.com",
      license="MIT",
      url="https://github.com/vahidk/tfrecord",
      packages=["tfrecord"],
      install_requires=["numpy", "protobuf"])
