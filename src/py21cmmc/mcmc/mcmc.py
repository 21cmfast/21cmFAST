# from .core import CoreCoevalModule
import sys
from os import path, mkdir
from .cosmoHammer import CosmoHammerSampler, LikelihoodComputationChain, HDFStorageUtil, Params


def build_computation_chain(core_modules, likelihood_modules, params=None):
    if not hasattr(core_modules, "__len__"):
        core_modules = [core_modules]

    if not hasattr(likelihood_modules, "__len__"):
        likelihood_modules = [likelihood_modules]

    chain = LikelihoodComputationChain(params)

    for cm in core_modules:
        chain.addCoreModule(cm)

    for lk in likelihood_modules:
        chain.addLikelihoodModule(lk)

    chain.setup()
    return chain


def run_mcmc(core_modules, likelihood_modules, params,
             datadir='.', model_name='21CMMC',
             reuse_burnin=True, continue_sampling=True,
             **mcmc_options):

    file_prefix = path.join(datadir, model_name)

    try:
        mkdir(datadir)
    except:
        pass

    # Setup parameters.
    if not isinstance(params, Params):
        params = Params(*[(k, v) for k, v in params.items()])

    # # Setup the Core Module
    # core_module = Core21cmFastModule(
    #     params.keys,
    #     store = store,
    #     box_dim = box_dim, flag_options=flag_options,
    #     astro_params=astro_params, cosmo_params=cosmo_params
    # )

    # # Write the parameter names to a file, might be useful later.
    # param_names = [v[0] for v in parameters.values()]
    # with open(path.join(datadir, model_name+"_parameter_names.txt"), 'w') as f:
    #     f.write("\n".join(param_names)+'\n')

    # # Get all the likelihood modules.
    # likelihoods = []
    # for lk in likelihood_modules:
    #     if isinstance(lk, tuple):
    #         if isinstance(lk[0], str):
    #             likelihoods += [getattr(sys.modules['py21cmmc.likelihood'],
    #                                     "Likelihood%s" % lk[0])(box_dim = box_dim,
    #                                                             astro_params=astro_params,
    #                                                             cosmo_params=cosmo_params,
    #                                                             flag_options=flag_options,
    #                                                             **lk[1])]
    #         else:
    #             likelihoods += [lk[0](
    #                 box_dim = box_dim,
    #                 astro_params=astro_params,
    #                 cosmo_params=cosmo_params,
    #                 flag_options=flag_options,**lk[1]
    #             )]
    #
    #     else:
    #         likelihoods += [lk]

    chain = build_computation_chain(core_modules, likelihood_modules, params)

    sampler = CosmoHammerSampler(
        continue_sampling=continue_sampling,
        likelihoodComputationChain=chain,
        storageUtil= HDFStorageUtil(file_prefix),
        filePrefix=file_prefix,
        reuseBurnin=reuse_burnin,
        **mcmc_options
    )

    # The sampler writes to file, so no need to save anything ourselves.
    sampler.startSampling()

    return sampler
