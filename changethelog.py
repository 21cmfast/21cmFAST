#!/usr/bin/python
import sys
from datetime import datetime

if __name__ == "__main__":
    newversion = sys.argv[1]

    with open("CHANGELOG.rst", "r") as fl:
        lines = fl.readlines()

    for i, line in enumerate(lines):
        if line == "dev-version\n":
            break
    else:
        raise IOError("Couldn't Find 'dev-version' tag")

    lines.insert(i + 2, "----------------------\n")
    lines.insert(i + 2, f'v{newversion} [{datetime.now().strftime("%d %b %Y")}]\n')
    lines.insert(i + 2, "\n")

    with open("CHANGELOG.rst", "w") as fl:
        fl.writelines(lines)
