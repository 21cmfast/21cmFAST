import matplotlib
matplotlib.use('Agg')
from py21cmmc import mcmc

print('initialising core')
core = mcmc.CoreCoevalModule(redshift = [7,8,9], user_params = dict(HII_DIM = 50, BOX_LEN = 125.0 ),
                             cosmo_params={"SIGMA_8":0.9},
                             regenerate=False)

# Now the likelihood...
datafiles = ["data/simple_mcmc_data_%s.npz"%z for z in core.redshift]

print('initialising likelihood')
likelihood = mcmc.Likelihood1DPowerCoeval(
    datafile = datafiles, noisefile= None, logk=False, min_k=0.1, max_k=1.0, simulate = True
)


model_name = "SimpleTest"
print('setting off mcmc chain')

chain = mcmc.run_mcmc(
    core, likelihood, datadir='data', model_name=model_name,
    params=dict( HII_EFF_FACTOR = [30.0, 10.0, 50.0, 3.0], ION_Tvir_MIN = [4.7, 2, 8, 0.1],),
    walkersRatio=2, burninIterations=0, sampleIterations=5, threadCount=2, continue_sampling=False
)
