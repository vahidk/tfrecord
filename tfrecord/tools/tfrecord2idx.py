from __future__ import print_function

import sys
import struct
import os

def create_index(tfrecord_file: str, index_file: str) -> None:
    """
    Create index from the tfrecords file.
    Stores starting location (byte) and length (in bytes) of each
    serialized record.
    If a direcotry of tfrecords is provided, a corresponding index
    will be created in place. 

    Params:
    -------
    tfrecord_file: str
        Path to the TFRecord file.
        OR
        Path to directory of TFRecord file ending with .tfrec or .tfrecord.
    index_file: str
        Path where to store the index file.
        OR
        Path to directory of index files.
    """
    if os.path.isdir(tfrecord_file):
      tfrecord_files = []
      for dirpath,_,filenames in os.walk(tfrecord_file):
        for file in filenames:
          if file.endswith(".tfrec") or file.endswith(".tfrecord"):
            tfrecord_files.append(os.path.abspath(os.path.join(dirpath, file)))

      index_files = [index_file + os.path.splitext(os.path.basename(x))[0] + ".index" for x in tfrecord_files]
      for i, (tf, idx) in enumerate(zip(tfrecord_files, index_files)):
        print(f"CREATED {i+1} of {len(tfrecord_files)} index files at {idx}")
        create_index(tf, idx)

    else:
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
