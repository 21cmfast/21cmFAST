"""Given a log file from Github Actions, takes away stuff that *should* change between runs.

Useful for preparing it to be diff'ed against another log to check where things went wrong.
"""

import sys

fname = sys.argv[1]

if len(sys.argv) > 2:
    ignores = sys.argv[2:]
else:
    ignores = []

pids = set()
threads = set()

pid_map = {}
with open(fname) as fl:
    lines = fl.readlines()

    # Get to the start of the testing
    for i, line in enumerate(lines):
        if "==== test session starts ====" in line:
            break

    print(f"Starting on line {i}")

    for line in lines[i:]:
        pids.add(line.split("pid=")[1].split("/")[0] if "pid=" in line else "other")
        threads.add(line.split("thr=")[1].split("]")[0] if "thr=" in line else "other")

    for ii, pid in enumerate(sorted(pids)):
        pid_map[pid] = ii

    out = {p: {t: [] for t in threads} for p in pid_map}

    for line in lines[(i + 1) :]:
        if "======" in line:
            break

        if any(ign in line for ign in ignores):
            continue

        ln = "|".join(line.split("|")[1:])
        pid = ln.split("pid=")[1].split("/")[0] if "pid=" in ln else "other"
        thread = ln.split("thr=")[1].split("]")[0] if "thr=" in ln else "other"

        ln = ln.replace(f"pid={pid}", f"pid={pid_map[pid]}")

        out[pid][thread].append(ln)


with open(fname + ".processed", "w") as fl:
    for key in sorted(pids):
        for t in sorted(threads):
            fl.writelines(out[key][t])
