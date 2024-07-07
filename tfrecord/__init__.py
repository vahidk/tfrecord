from tfrecord import tools
try:
    from tfrecord import torch
except ImportError:
    pass

from tfrecord import example_pb2
from tfrecord import iterator_utils
from tfrecord import reader
from tfrecord import writer

from tfrecord.iterator_utils import *
from tfrecord.reader import *
from tfrecord.writer import *
