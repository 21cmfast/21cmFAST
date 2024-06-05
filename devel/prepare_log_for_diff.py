"""Run over output of logging to prepare for diffing."""

import sys

fname = sys.argv[1]

if len(sys.argv) > 2:
    REMOVE_LINE_NUMBERS = True
    print("REMOVING LINE NUMBERS ")
else:
    REMOVE_LINE_NUMBERS = False


with open(fname) as fl:
    lines = fl.readlines()
    out_lines = []
    for line in lines:
        bits = line.split("|")

        # Get rid of time.
        if len(bits) > 2:
            line = "|".join(bits[1:])

        # get rid of pid
        if len(bits) > 2:
            pid_bit = line.split("[pid=")[-1].split("]")[0]
            line = line.replace(pid_bit, "")

        if REMOVE_LINE_NUMBERS and len(bits) > 2:
            line_no = line.split(":")[1].split("[")[0].strip()
            line = line.replace(line_no, "")

        out_lines.append(line)

with open(fname + ".out", "w") as fl:
    fl.writelines(out_lines)
