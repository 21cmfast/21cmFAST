from py21cmmc import mcmc

nthreads = 2

class MyCore(mcmc.CoreCoevalModule):
    def prepare_storage(self, ctx, storage):
        storage['bt'] = ctx.get("brightness_temp")[0].brightness_temp[:,0,0]

core = MyCore(
    redshifts=[9],
    user_params=dict(
        HII_DIM=70,
        BOX_LEN=150.0
    ),
    regenerate=False,
)

likelihood = mcmc.Likelihood1DPowerCoeval(
    datafile="simple_mcmc_data.txt",
    logk=False
)

params = {"HII_EFF_FACTOR": [30.0, 20.0, 40.0, 3.0]}

model_name = "SimpleTest"

chain = mcmc.run_mcmc(
    core, likelihood,
    datadir='data', model_name=model_name,
    params=params,
    walkersRatio=2, burninIterations=0, sampleIterations=3, threadCount=nthreads, continue_sampling=False
)
