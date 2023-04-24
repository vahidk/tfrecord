from __future__ import print_function

import os
import sys
import struct
import glob


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
            outfile.write(str(current) + " " +
                          str(infile.tell() - current) + "\n")
        except:
            print("Failed to parse TFRecord.")
            break
    infile.close()
    outfile.close()


def create_indices(tfrecord_dir: str) -> None:
    """Create indices for all tfrecord files in the directory.

    Params:
    -------
    tfrecord_dir: str
        Path to the directory containing TFRecord files.
    """

    for tfrecord_file in glob.glob(os.path.join(tfrecord_dir, "*.tfrecord")):
        index_file = os.path.splitext(tfrecord_file)[0] + ".tfindex"
        create_index(tfrecord_file, index_file)


def main():
    if len(sys.argv) not in [2, 3]:
        print("""Usage:
        To create index for individual tfrecord file:
        tfrecord2idx <tfrecord path> <index path>

        To create index for all tfrecord files in a directory:
        tfrecord2idx <tfrecord dir>
        
        This will search for all "*.tfrecord" files and create corresponding
        "*.tfindex" files.
        """)
        sys.exit()

    if os.path.isdir(sys.argv[1]):
        create_indices(sys.argv[1])
    else:
        create_index(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
