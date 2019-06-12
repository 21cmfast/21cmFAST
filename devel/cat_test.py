from py21cmmc import mcmc
import logging


# ======== USER ADJUSTABLE VARIABLES
MODEL_NAME = "power_only"
CONT = False
THREADS = 8
WALK_RATIO = 2
ITER = 1
BURN = 0
OLD_PARAMETERISATION = True
# ===================================

core = mcmc.CoreCoevalModule(
    redshift=[9],
    user_params=dict(HII_DIM=50, BOX_LEN=125.0),
    flag_options=dict(USE_MASS_DEPENDENT_ZETA=not OLD_PARAMETERISATION),
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

# OLD PARAMATRISATION - for code development and proof of concepts
if OLD_PARAMETERISATION:
    params = dict(
        HII_EFF_FACTOR=[30.0, 10.0, 50.0, 3.0],
        ION_Tvir_MIN=[4.7, 4, 6, 0.1])
else:
    params = dict(
        F_STAR10=[-1.301029996, -3, 0, 0.1],
        ALPHA_STAR=[0.5, -0.5, 1, 0.05],
        F_ESC10=[-1, -3, 0, 0.1],
        ALPHA_ESC=[-0.5, -1, 0.5, 0.05],
        M_TURN=[8.698970004, 8, 10, 0.1],
        t_STAR=[0.5, 0, 1, 0.05],
        L_X=[40.5, 38, 42, 0.15],
        NU_X_THRESH=[500, 100, 1500, 50]
    )

chain = mcmc.run_mcmc(
    core, power_spec, datadir='data', model_name=MODEL_NAME,
    params=params,
    log_level_21CMMC='DEBUG',
    walkersRatio=WALK_RATIO, burninIterations=BURN,
    sampleIterations=ITER, threadCount=THREADS, continue_sampling=CONT
)