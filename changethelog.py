#!/usr/bin/python
"""Update the changelog automatically."""

import sys
from datetime import UTC, datetime
from pathlib import Path

if __name__ == "__main__":
    newversion = sys.argv[1]
    pth = Path("CHANGELOG.rst")
    with pth.open() as fl:
        lines = fl.readlines()

    for i, line in enumerate(lines):
        if line == "dev-version\n":
            lines.insert(i + 2, "----------------------\n")
            lines.insert(
                i + 2, f"v{newversion} [{datetime.now(tz=UTC).strftime('%d %b %Y')}]\n"
            )
            lines.insert(i + 2, "\n")
            break
    else:
        raise OSError("Couldn't Find 'dev-version' tag")

    with pth.open("w") as fl:
        fl.writelines(lines)
