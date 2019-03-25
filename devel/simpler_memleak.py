import gc
# ===================Stuff to track memory usage.===============================
import tracemalloc
from concurrent.futures import ProcessPoolExecutor

import numpy as np

tracemalloc.start()
snapshot = tracemalloc.take_snapshot()


def trace_print():
    global snapshot
    snapshot2 = tracemalloc.take_snapshot()
    snapshot2 = snapshot2.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
        tracemalloc.Filter(False, tracemalloc.__file__)
    ))

    if snapshot is not None:
        print("================================== Begin Trace:")
        top_stats = snapshot2.compare_to(snapshot, 'lineno', cumulative=True)
        for stat in top_stats[:10]:
            if stat.size_diff != 0:
                print(stat)
    snapshot = snapshot2


# ==============================================================================

class LCC:
    def __init__(self, ):
        self.a = 2

        self.big_array = np.zeros(1000000)

        # self-ref
        self._selfref = self

    def __call__(self, x):
        trace_print()

        gc.collect()
        return x * np.sum(self.big_array)


if __name__ == "__main__":
    nthreads = 1
    p = ProcessPoolExecutor(max_workers=nthreads)

    lnprobfn = LCC()  # (MemoryLeaker(), CoreOne())

    for i in range(20):
        p.map(lnprobfn, range(4))
