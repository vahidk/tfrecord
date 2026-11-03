"""Shared TFRecord framing and integrity checks."""

import struct

import crc32c


def masked_crc(data):
    crc = crc32c.crc32c(data)
    masked = (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF
    return struct.pack("<I", masked)


def read_record(file, end_offset=None):
    """Read one validated payload, or return None at a clean end of file."""
    offset = file.tell()
    length_bytes = file.read(8)
    if not length_bytes:
        if end_offset is not None and offset < end_offset:
            raise RuntimeError(f"Failed to read the record size at byte {offset}.")
        return None
    if len(length_bytes) != 8:
        raise RuntimeError(f"Failed to read the record size at byte {offset}.")
    length_crc = file.read(4)
    if len(length_crc) != 4:
        raise RuntimeError(f"Failed to read the length checksum at byte {offset}.")
    if length_crc != masked_crc(length_bytes):
        raise RuntimeError(f"Invalid length checksum at byte {offset}.")

    (length,) = struct.unpack("<Q", length_bytes)
    if end_offset is not None and offset + length + 16 > end_offset:
        raise RuntimeError(f"Record at byte {offset} extends beyond the end of its data range.")
    payload = file.read(length)
    if len(payload) != length:
        raise RuntimeError(f"Failed to read the record at byte {offset}.")
    payload_crc = file.read(4)
    if len(payload_crc) != 4:
        raise RuntimeError(f"Failed to read the payload checksum at byte {offset}.")
    if payload_crc != masked_crc(payload):
        raise RuntimeError(f"Invalid payload checksum at byte {offset}.")
    return payload
