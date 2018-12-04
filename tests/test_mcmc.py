from py21cmmc import mcmc
from py21cmmc.mcmc import analyse
import pytest
import numpy as np
import os

@pytest.fixture(scope="module")
def core():
    return mcmc.CoreCoevalModule(redshift=9, user_params={"HII_DIM": 35, "DIM": 70},
                                 cache_ionize=False, cache_init=False)

@pytest.fixture(scope="module")
def likelihood_coeval(tmpdirec):
    return mcmc.Likelihood1DPowerCoeval(simulate=True, datafile=os.path.join(tmpdirec.strpath, "likelihood_coeval"))


def test_core_coeval_not_setup():
    core = mcmc.CoreCoevalModule(redshift=9)

    with pytest.raises(mcmc.NotAChain):
        core.chain


def test_core_coeval_setup(core, likelihood_coeval):
    with pytest.raises(ValueError): # If simulate is not true, and no datafile given...
        lk = mcmc.Likelihood1DPowerCoeval()
        mcmc.build_computation_chain(core, lk)

    mcmc.build_computation_chain(core, likelihood_coeval)

    assert isinstance(core.chain, mcmc.cosmoHammer.LikelihoodComputationChain)
    assert core.initial_conditions is not None
    assert core.initial_conditions_seed is not None

    ctx = core.chain.createChainContext()
    core.simulate_data(ctx)

    assert ctx.get("xHI") is not None
    assert ctx.get("brightness_temp") is not None

    assert not np.all(ctx.get("xHI")==0)
    assert not np.all(ctx.get("brightness_temp")==0)

def test_mcmc(core, likelihood_coeval, tmpdirec):
    chain = mcmc.run_mcmc(
        core, likelihood_coeval, model_name="TEST",continue_sampling=False, datadir=tmpdirec.strpath,
        params=dict(HII_EFF_FACTOR=[30.0, 10.0, 50.0, 3.0], ION_Tvir_MIN=[4.7, 2, 8, 0.1]),
        walkersRatio=2, burninIterations=0, sampleIterations=10, threadCount=1
    )

    samples_from_chain = analyse.get_samples(chain)
    samples_from_file = analyse.get_samples(os.path.join(tmpdirec.strpath, "TEST"))

    # make sure reading from file is the same as the chain.
    assert samples_from_chain.iteration == samples_from_file.iteration
    assert np.all(samples_from_file.accepted == samples_from_chain.accepted)
    assert np.all(samples_from_file.get_chain() == samples_from_chain.get_chain())

    assert all([c in ['HII_EFF_FACTOR', "ION_Tvir_MIN"] for c in samples_from_chain.param_names])
    assert samples_from_chain.has_blobs
    assert samples_from_chain.param_guess['HII_EFF_FACTOR'] == 30.0
    assert samples_from_chain.param_guess['ION_Tvir_MIN'] == 4.7
