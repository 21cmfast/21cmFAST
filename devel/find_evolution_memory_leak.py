"""Run a quick simulation to test memory leaks."""

import os
import tracemalloc
from collections import deque

import psutil

import py21cmfast as p21c

tracemalloc.start()
snapshot = tracemalloc.take_snapshot()
PROCESS = psutil.Process(os.getpid())
oldmem = 0


def trace_print():
    """Print a trace of memory leaks."""
    global snapshot
    global oldmem

    snapshot2 = tracemalloc.take_snapshot()
    snapshot2 = snapshot2.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
            tracemalloc.Filter(False, tracemalloc.__file__),
        )
    )

    if snapshot is not None:
        thismem = PROCESS.memory_info().rss / 1024**2
        diff = thismem - oldmem
        print(
            f"===================== Begin Trace (TOTAL MEM={thismem:1.4e} MB... [{diff:+1.4e} MB]):"
        )
        top_stats = snapshot2.compare_to(snapshot, "lineno", cumulative=True)
        for stat in top_stats[:4]:
            print(stat)
        print("End Trace ===========================================")
        oldmem = thismem

    snapshot = snapshot2


inputs = p21c.InputParameters.from_template(
    ["Munoz21", "medium"], random_seed=0, MINIMIZE_MEMORY=True, ZPRIME_STEP_FACTOR=1.2
)

print(inputs)

coevals = deque(maxlen=3)

for coeval, _ in p21c.generate_coeval(
    inputs=inputs, cache=p21c.OutputCache("EvolutionMemoryLeakCache"), write=True
):
    print(coeval.redshift)
    trace_print()

    print("---------------")
    coevals.append(coeval)
    for ostruct in coevals[0].output_structs.values():
        print(ostruct.__class__.__name__)
        for name, ary in ostruct.arrays.items():
            print(f"   {name}: {ary.state}")
    print("---------------")
