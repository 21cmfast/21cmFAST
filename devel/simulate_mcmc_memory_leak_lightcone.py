"""Simple script to simulate an MCMC-like routine to check for memory leaks."""
import numpy as np
import os
import psutil
import tracemalloc

from py21cmfast import initial_conditions, perturb_field, run_lightcone

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
        thismem = PROCESS.memory_info().rss / 1024 ** 2
        diff = thismem - oldmem
        print(
            "===================== Begin Trace (TOTAL MEM={:1.4e} MB... [{:+1.4e} MB]):".format(
                thismem, diff
            )
        )
        top_stats = snapshot2.compare_to(snapshot, "lineno", cumulative=True)
        for stat in top_stats[:4]:
            print(stat)
        print("End Trace ===========================================")
        oldmem = thismem

    snapshot = snapshot2


trace_print()

run_lightcone(
    redshift=15,
    user_params={
        "USE_INTERPOLATION_TABLES": True,
        "N_THREADS": 1,
        "DIM": 100,
        "HII_DIM": 25,
        "PERTURB_ON_HIGH_RES": True,
        "USE_FFTW_WISDOM": False,
    },
    flag_options={
        "USE_MASS_DEPENDENT_ZETA": True,
        "INHOMO_RECO": True,
        "USE_TS_FLUCT": True,
    },
    direc="_cache_%s" % (os.path.basename(__file__)[:-3]),
    random_seed=1993,
)
