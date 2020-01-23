#!/usr/bin/env python

from distutils.core import setup

setup(name="tfrecord",
      version="1.8",
      description="TFRecord reader.",
      author="Vahid Kazemi",
      author_email="vkazemi@gmail.com",
      license="MIT",
      url="https://github.com/vahidk/tfrecord",
      packages=["tfrecord", "tfrecord.tools", "tfrecord.torch"],
      install_requires=["numpy", "protobuf"])
