#!/usr/bin/python
"""Filter valgrind output to only include stuff that's compiled directly from C sources."""

import fileinput
import re

START = re.compile("in loss record")
STOP = re.compile(r"^==\d+== $")
GOOD = re.compile(r"py21cmfast\.c_21cmfast\.c:\d+")


def main():
    """Find only lines with specified GOOD strings in them."""
    in_line = False
    current = []
    for line in fileinput.input():
        if in_line:
            in_line = not STOP.search(line)
        else:
            in_line = START.search(line)

        if in_line:
            current.append(line)
        else:
            match = GOOD.findall("".join(current))
            if len(match) > 0:
                print("".join(current))
            current = []


if __name__ == "__main__":
    main()
