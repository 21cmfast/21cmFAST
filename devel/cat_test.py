import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from py21cmmc.mcmc import analyse
from py21cmmc import mcmc

####### USER ADJUSTABLE VARIABLES ########
MODEL_NAME = "power_only"
CONT = False
THREADS = 8
WALK_RATIO = 8
ITER = 10
BURN = 0

print('initialising core')
core = mcmc.CoreCoevalModule(
    redshift=[7, 8, 9], user_params=dict(HII_DIM=50, BOX_LEN=125.0),
    flag_options=dict(USE_MASS_DEPENDENT_ZETA=True, USE_TS_FLUCT=False, OUTPUT_AVE=True), regenerate=False)
# core = mcmc.CoreCoevalModule( redshift = [7, 8, 9], user_params = dict( HII_DIM = 50, BOX_LEN = 125.0 ), regenerate =False)
# Now the likelihood...
datafiles = ["data/power_mcmc_data_%s.npz" % z for z in core.redshift]

print('initialising likelihood')
power_spec = mcmc.Likelihood1DPowerCoeval(datafile=datafiles, noisefile=None, logk=False, min_k=0.1, max_k=1.0,
                                          simulate=True)

print('setting off mcmc chain')
# OLD PARAMATRISATION - for code development and proof of concepts
# chain = mcmc.run_mcmc(core, power_spec, datadir='data', model_name=MODEL_NAME, params=dict( HII_EFF_FACTOR = [30.0, 10.0, 50.0, 3.0], ION_Tvir_MIN = [4.7, 4, 6, 0.1],), walkersRatio=WALK_RATIO, burninIterations=BURN, sampleIterations=ITER, threadCount=THREADS, continue_sampling=CONT)
# NEW PARAMS - most up to date and physically motivated parametrisation
chain = mcmc.run_mcmc(core, power_spec, datadir='data', model_name=MODEL_NAME,
                      params=dict(F_STAR10=[-1.301029996, -3, 0, 0.1], ALPHA_STAR=[0.5, -0.5, 1, 0.05],
                                  F_ESC10=[-1, -3, 0, 0.1], ALPHA_ESC=[-0.5, -1, 0.5, 0.05],
                                  M_TURN=[8.698970004, 8, 10, 0.1], t_STAR=[0.5, 0, 1, 0.05], L_X=[40.5, 38, 42, 0.15],
                                  NU_X_THRESH=[500, 100, 1500, 50]), walkersRatio=WALK_RATIO, burninIterations=BURN,
                      sampleIterations=ITER, threadCount=THREADS, continue_sampling=CONT)
print('plotting')
samples1 = chain.samples
samples2 = analyse.get_samples(chain)
samples3 = analyse.get_samples("data/%s" % MODEL_NAME)

print(samples1 is samples3)
del samples3;
samples = samples1

niter = samples.size
nwalkers, nparams = samples.shape

print(samples.param_names)
print([samples.param_guess[k] for k in samples.param_names])

samples.accepted, np.mean(samples.accepted / niter)
samples.blob_names

analyse.trace_plot(samples, include_lnl=True, start_iter=0, thin=1, colored=False, show_guess=True);
out_str = 'trace_' + MODEL_NAME + '_threads' + str(THREADS) + '_iter' + str(ITER) + '_cont' + str(CONT) + '_z7_8.png'
plt.savefig(out_str)

samples.param_guess
analyse.corner_plot(samples);
out_str = 'corner_' + MODEL_NAME + '_threads' + str(THREADS) + '_iter' + str(ITER) + '_cont' + str(CONT) + '_z7_8.png'
plt.savefig(out_str)
