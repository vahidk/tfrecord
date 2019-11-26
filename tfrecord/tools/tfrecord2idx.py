#!/usr/bin/env python
import sys
import struct


def main():
    if len(sys.argv) < 3:
        print("Usage: tfrecord2idx <tfrecord path> <index path>")
        exit()

    infile = open(sys.argv[1], "rb")
    outfile = open(sys.argv[2], "w")

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


if __name__ == "__main__":
    main()
