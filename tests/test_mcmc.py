from py21cmmc import mcmc
import pytest
import numpy as np

def test_core_coeval_not_setup():
    core = mcmc.CoreCoevalModule(redshift=9)

    with pytest.raises(mcmc.NotAChain):
        core.chain


def test_core_coeval_setup():
    core = mcmc.CoreCoevalModule(redshift=9, user_params={"HII_DIM": 35, "DIM":70},
                                 cache_ionize=False, cache_init=False)

    with pytest.raises(ValueError): # If simulate is not true, and no datafile given...
        lk = mcmc.Likelihood1DPowerCoeval()
        mcmc.build_computation_chain(core, lk)

    lk = mcmc.Likelihood1DPowerCoeval(simulate=True)
    mcmc.build_computation_chain(core, lk)

    assert isinstance(core.chain, mcmc.cosmoHammer.LikelihoodComputationChain)
    assert core.initial_conditions is not None
    assert core.initial_conditions_seed is not None

    ctx = core.chain.createChainContext()
    core.simulate_data(ctx)

    assert ctx.get("xHI") is not None
    assert ctx.get("brightness_temp") is not None

    assert not np.all(ctx.get("xHI")==0)
    assert not np.all(ctx.get("brightness_temp")==0)
