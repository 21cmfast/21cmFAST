from py21cmmc import mcmc
import numpy as np
import multiprocessing
import ctypes

# ======== USER ADJUSTABLE VARIABLES
MODEL_NAME = "power_only"
CONT = False
THREADS = 1
WALK_RATIO = 2
ITER = 1
BURN = 0
# ===================================

# Stuff to track memory usage.
import tracemalloc
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
            print(stat)
    snapshot = snapshot2


class MyPrinterCore(mcmc.CoreCoevalModule):
    def setup(self):
        super().setup()

        shared_array_base = multiprocessing.Array(ctypes.c_double, 1000000)
        self.shared_array = np.frombuffer(shared_array_base.get_obj())


#        self.big_array = np.zeros(1000000)

    def build_model_data(self, ctx):
        trace_print()
        super().build_model_data(ctx)


core = MyPrinterCore(
    redshift=[9],
    user_params=dict(HII_DIM=50, BOX_LEN=125.0),
    flag_options=dict(USE_MASS_DEPENDENT_ZETA=False),
    do_spin_temp=False,
    z_step_factor=1.2,
    regenerate=True,   # ensure each run is started fresh
    initial_conditions_seed=1234  # ensure each run is exactly the same.
)

# Now the likelihood...
datafiles = ["data/power_mcmc_data_%s.npz" % z for z in core.redshift]
power_spec = mcmc.Likelihood1DPowerCoeval(
    datafile=datafiles,
    noisefile=None,
    logk=False, min_k=0.1, max_k=1.0,
    simulate=True
)


params = dict(
    HII_EFF_FACTOR=[30.0, 10.0, 50.0, 3.0],
    ION_Tvir_MIN=[4.7, 4, 6, 0.1])

pool = multiprocessing.Pool(8)

chain = mcmc.run_mcmc(
    core, power_spec, datadir='data', model_name=MODEL_NAME,
    params=params,
    log_level_21CMMC='WARNING',
    walkersRatio=WALK_RATIO, burninIterations=BURN,
    sampleIterations=ITER, threadCount=THREADS, continue_sampling=CONT
)