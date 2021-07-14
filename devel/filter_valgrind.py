#!/usr/bin/python
"""Filter valgrind output to only include stuff that's compiled directly from C sources."""

import fileinput
import re

START = re.compile(r"^==\d+== \w")
STOP = re.compile(r"^==\d+== $")
ANY = re.compile(r"^==\d+==")
GOOD = [
    re.compile(r"py21cmfast\.c_21cmfast\.c:\d+"),
    re.compile(r"SUMMARY"),
]


def main():
    """Find only lines with specified GOOD strings in them."""
    in_line = False
    current = []
    for line in fileinput.input():
        if not ANY.search(line):
            print(line, end="")
            continue

        in_line = not STOP.search(line) if in_line else START.search(line)
        if in_line:
            current.append(line)
        else:
            for g in GOOD:
                match = g.findall("".join(current))

                if match:
                    print("".join(current))
                    continue

            current = []


if __name__ == "__main__":
    main()
