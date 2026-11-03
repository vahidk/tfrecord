import gzip
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from tfrecord import example_pb2, reader
from tfrecord.tools.tfrecord2idx import create_index
from tfrecord.writer import TFRecordWriter


class TestRecordIntegrity(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "data.tfrecord"
        self.index = self.root / "data.index"

    def write(self, records):
        writer = TFRecordWriter(str(self.path))
        try:
            for record in records:
                writer.write(record)
        finally:
            writer.close()

    def test_binary_byte_arrays_round_trip(self):
        original = [b"a\x00", b"b\x00\x00", b"", b"\x00"]
        self.write([{"value": (original, "byte")}])
        decoded = next(reader.example_loader(str(self.path), None))["value"]
        self.assertEqual(decoded.tolist(), original)
        serialized = TFRecordWriter.serialize_tf_example({"value": (decoded, "byte")})
        rewritten = example_pb2.Example.FromString(serialized)
        self.assertEqual(list(rewritten.features.feature["value"].bytes_list.value), original)

    def test_raw_records_remain_stable_after_iteration(self):
        self.write([{"value": (1, "int")}, {"value": (2, "int")}])
        records = list(reader.tfrecord_iterator(str(self.path)))
        self.assertTrue(all(isinstance(record, memoryview) for record in records))
        values = [
            example_pb2.Example.FromString(record).features.feature["value"].int64_list.value[0]
            for record in records
        ]
        self.assertEqual(values, [1, 2])

    def test_empty_and_single_record_indexes(self):
        for count in (0, 1):
            with self.subTest(count=count):
                self.write([{"value": (42, "int")}] * count)
                create_index(str(self.path), str(self.index))
                records = list(reader.example_loader(str(self.path), str(self.index)))
                self.assertEqual(len(records), count)
                shards = [
                    list(reader.example_loader(str(self.path), str(self.index), shard=(worker, 4)))
                    for worker in range(4)
                ]
                self.assertEqual(sum(map(len, shards)), count)

    def test_empty_worker_shard_terminates_infinite_sampling(self):
        self.write([{"value": (1, "int")}, {"value": (2, "int")}])
        create_index(str(self.path), str(self.index))
        original_loader = reader.tfrecord_loader
        calls = 0

        def guarded_loader(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 3:
                self.fail("Empty worker shard was restarted instead of removed")
            return original_loader(*args, **kwargs)

        with mock.patch.object(reader, "tfrecord_loader", side_effect=guarded_loader):
            records = list(reader.multi_tfrecord_loader(
                str(self.path), str(self.index), {"unused": 1.0}, shard=(0, 4)
            ))
        self.assertEqual(records, [])

    def test_header_and_payload_checksums_are_checked(self):
        self.write([{"value": (42, "int")}])
        original = self.path.read_bytes()
        for position in (0, 8, len(original) - 5, len(original) - 1):
            with self.subTest(position=position):
                damaged = bytearray(original)
                damaged[position] ^= 1
                self.path.write_bytes(damaged)
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    list(reader.tfrecord_iterator(str(self.path)))
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    create_index(str(self.path), str(self.index))
                self.assertFalse(self.index.exists())

    def test_truncated_records_are_rejected_without_replacing_index(self):
        self.write([{"value": (42, "int")}])
        original = self.path.read_bytes()
        self.index.write_text("previous index\n")
        for removed in (1, 4, 5, len(original) - 8, len(original) - 1):
            with self.subTest(removed=removed):
                self.path.write_bytes(original[:-removed])
                with self.assertRaises(RuntimeError):
                    list(reader.tfrecord_iterator(str(self.path)))
                with self.assertRaises(RuntimeError):
                    create_index(str(self.path), str(self.index))
                self.assertEqual(self.index.read_text(), "previous index\n")
        self.assertEqual(
            sorted(path.name for path in self.root.iterdir()), ["data.index", "data.tfrecord"]
        )

    def test_index_is_not_published_when_a_later_record_is_corrupt(self):
        self.write([{"value": (1, "int")}, {"value": (2, "int")}])
        self.path.write_bytes(self.path.read_bytes()[:-1])
        with self.assertRaises(RuntimeError):
            create_index(str(self.path), str(self.index))
        self.assertFalse(self.index.exists())

    def test_empty_payload_is_a_record(self):
        length = struct.pack("<Q", 0)
        self.path.write_bytes(
            length + TFRecordWriter.masked_crc(length) + TFRecordWriter.masked_crc(b"")
        )
        records = list(reader.tfrecord_iterator(str(self.path)))
        self.assertEqual([bytes(record) for record in records], [b""])
        create_index(str(self.path), str(self.index))
        self.assertEqual(list(reader.example_loader(str(self.path), str(self.index))), [{}])

    def test_gzip_reading_with_and_without_index(self):
        self.write([{"value": (1, "int")}, {"value": (2, "int")}])
        create_index(str(self.path), str(self.index))
        compressed = self.root / "data.tfrecord.gz"
        compressed.write_bytes(gzip.compress(self.path.read_bytes()))
        for index in (None, str(self.index)):
            with self.subTest(index=index):
                records = list(reader.example_loader(str(compressed), index, compression_type="gzip"))
                self.assertEqual(sorted(record["value"].item() for record in records), [1, 2])
        damaged = bytearray(self.path.read_bytes())
        damaged[-5] ^= 1
        compressed.write_bytes(gzip.compress(damaged))
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            list(reader.tfrecord_iterator(str(compressed), compression_type="gzip"))
