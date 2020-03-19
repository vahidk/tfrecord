from __future__ import print_function

import sys
import struct


def create_index(tfrecord_file: str, index_file: str) -> None:
    """Create index from the tfrecords file.

    Stores starting location (byte) and length (in bytes) of each
    serialized record.

    Params:
    -------
    tfrecord_file: str
        Path to the TFRecord file.

    index_file: str
        Path where to store the index file.
    """
    infile = open(tfrecord_file, "rb")
    outfile = open(index_file, "w")

    while True:
        current = infile.tell()
        try:
            byte_len = infile.read(8)
            if len(byte_len) == 0:
                break
            infile.read(4)
            proto_len = struct.unpack("q", byte_len)[0]
            infile.read(proto_len)
            infile.read(4)
            outfile.write(str(current) + " " + str(infile.tell() - current) + "\n")
        except:
            print("Failed to parse TFRecord.")
            break
    infile.close()
    outfile.close()


def main():
    if len(sys.argv) < 3:
        print("Usage: tfrecord2idx <tfrecord path> <index path>")
        sys.exit()

    create_index(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
