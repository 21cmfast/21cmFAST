"""Simple script to simulate an MCMC-like routine to check for memory leaks."""

import os
import tracemalloc

import numpy as np
import psutil

from py21cmfast import initial_conditions, perturb_field, run_coeval

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


NITER = 50

init = initial_conditions(
    matter_params={"HII_DIM": 50, "BOX_LEN": 125.0}, regenerate=True
)
perturb = (
    perturb_field(redshift=7, init_boxes=init),
    perturb_field(redshift=8, init_boxes=init),
    perturb_field(redshift=9, init_boxes=init),
)

rng = np.random.default_rng()
astro_params = {"HII_EFF_FACTOR": rng.normal(30, 0.1)}

for _i in range(NITER):
    trace_print()

    coeval = run_coeval(
        redshift=[7, 8, 9],
        astro_params=astro_params,
        init_box=init,
        perturb=perturb,
        regenerate=True,
        random_seed=init.random_seed,
        write=False,
    )
