import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestProtobufCompatibility(unittest.TestCase):
    def run_python(self, source):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_import_does_not_load_tensorflow(self):
        self.run_python("""
            import importlib.abc
            import sys

            attempts = []

            class BlockTensorFlow(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'tensorflow' or fullname.startswith('tensorflow.'):
                        attempts.append(fullname)
                        raise ImportError('TensorFlow must not be imported')

            sys.meta_path.insert(0, BlockTensorFlow())
            import tfrecord
            assert not attempts, attempts
            assert 'tensorflow' not in sys.modules
        """)

    def test_descriptor_pool_is_isolated_in_both_import_orders(self):
        for tfrecord_first in (False, True):
            with self.subTest(tfrecord_first=tfrecord_first):
                self.run_python(f"""
                    from google.protobuf import descriptor_pb2, descriptor_pool

                    if {tfrecord_first!r}:
                        import tfrecord

                    # Model TensorFlow registering its own definitions in the
                    # default pool, without requiring TensorFlow to be installed.
                    proto = descriptor_pb2.FileDescriptorProto(
                        name='tensorflow/core/example/feature.proto',
                        package='tensorflow', syntax='proto3',
                    )
                    proto.message_type.add(name='BytesList')
                    descriptor_pool.Default().AddSerializedFile(proto.SerializeToString())

                    import tfrecord
                    assert tfrecord.example_pb2.DESCRIPTOR.pool is not descriptor_pool.Default()
                    message = tfrecord.example_pb2.BytesList(value=[b'test'])
                    assert list(message.value) == [b'test']
                    descriptor = descriptor_pool.Default().FindMessageTypeByName('tensorflow.BytesList')
                    assert descriptor.file.name == proto.name
                """)

    def test_public_message_types_and_pickle(self):
        self.run_python("""
            import importlib
            import pickle
            from tfrecord import example_pb2 as pb

            names = (
                'BytesList', 'FloatList', 'Int64List', 'Feature', 'Features',
                'FeatureList', 'FeatureLists', 'Example', 'SequenceExample',
            )
            assert set(pb.DESCRIPTOR.message_types_by_name) == set(names)
            for name in names:
                cls = getattr(pb, name)
                message = cls()
                assert pickle.loads(pickle.dumps(message)) == message
            assert pb.Features.FeatureEntry.DESCRIPTOR.full_name == 'tensorflow.Features.FeatureEntry'
            assert pb.FeatureLists.FeatureListEntry.DESCRIPTOR.full_name == 'tensorflow.FeatureLists.FeatureListEntry'
            assert importlib.reload(pb) is pb
            assert pickle.loads(pickle.dumps(pb.BytesList(value=[b'test']))).value == [b'test']
        """)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_tensorflow_wire_compatibility_in_both_import_orders(self):
        for tfrecord_first in (False, True):
            with self.subTest(tfrecord_first=tfrecord_first):
                self.run_python(f"""
                    import tempfile

                    if {tfrecord_first!r}:
                        import tfrecord
                        import tensorflow as tf
                    else:
                        import tensorflow as tf
                        import tfrecord

                    from tfrecord import example_pb2 as pb

                    features = pb.Features(feature={{
                        'bytes': pb.Feature(bytes_list=pb.BytesList(value=[b'a', b'b'])),
                        'float': pb.Feature(float_list=pb.FloatList(value=[1.25, 2.5])),
                        'int': pb.Feature(int64_list=pb.Int64List(value=[-1, 42])),
                    }})
                    example = pb.Example(features=features)
                    tf_example = tf.train.Example.FromString(example.SerializeToString())
                    assert list(tf_example.features.feature['bytes'].bytes_list.value) == [b'a', b'b']
                    assert list(tf_example.features.feature['float'].float_list.value) == [1.25, 2.5]
                    assert list(tf_example.features.feature['int'].int64_list.value) == [-1, 42]
                    tf_example.features.feature['int'].int64_list.value.append(99)
                    parsed = pb.Example.FromString(tf_example.SerializeToString())
                    assert list(parsed.features.feature['int'].int64_list.value) == [-1, 42, 99]

                    sequence = pb.SequenceExample(
                        context=features,
                        feature_lists=pb.FeatureLists(feature_list={{
                            'steps': pb.FeatureList(feature=[features.feature['int']]),
                        }}),
                    )
                    tf_sequence = tf.train.SequenceExample.FromString(sequence.SerializeToString())
                    assert list(tf_sequence.feature_lists.feature_list['steps'].feature[0].int64_list.value) == [-1, 42]
                    tf_sequence.context.feature['int'].int64_list.value.append(99)
                    parsed = pb.SequenceExample.FromString(tf_sequence.SerializeToString())
                    assert list(parsed.context.feature['int'].int64_list.value) == [-1, 42, 99]

                    with tempfile.TemporaryDirectory() as directory:
                        path = directory + '/example.tfrecord'
                        writer = tfrecord.TFRecordWriter(path)
                        writer.write({{'value': ([42], 'int')}})
                        writer.close()
                        raw = next(tf.compat.v1.io.tf_record_iterator(path))
                        assert tf.train.Example.FromString(raw).features.feature['value'].int64_list.value == [42]
                        with tf.io.TFRecordWriter(path) as writer:
                            writer.write(tf_example.SerializeToString())
                        record = next(tfrecord.example_loader(path, None))
                        assert record['int'].tolist() == [-1, 42, 99]
                """)
