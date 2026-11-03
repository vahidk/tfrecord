from __future__ import print_function

import glob
import os
import sys
import tempfile

from tfrecord import _record


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
    temporary_path = None
    try:
        with open(tfrecord_file, "rb") as infile:
            end_offset = os.fstat(infile.fileno()).st_size
            with tempfile.NamedTemporaryFile(
                mode="w", dir=os.path.dirname(os.path.abspath(index_file)),
                prefix=".tfrecord-index-", delete=False,
            ) as outfile:
                temporary_path = outfile.name
                while infile.tell() < end_offset:
                    current = infile.tell()
                    _record.read_record(infile, end_offset)
                    outfile.write(f"{current} {infile.tell() - current}\n")
            os.replace(temporary_path, index_file)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


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
        print(
            """Usage:
        To create index for individual tfrecord file:
        tfrecord2idx <tfrecord path> <index path>

        To create index for all tfrecord files in a directory:
        tfrecord2idx <tfrecord dir>
        
        This will search for all "*.tfrecord" files and create corresponding
        "*.tfindex" files.
        """
        )
        sys.exit()

    if os.path.isdir(sys.argv[1]):
        create_indices(sys.argv[1])
    else:
        create_index(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
