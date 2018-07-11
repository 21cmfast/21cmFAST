from .core import Core21cmFastModule
import sys
from os import path, mkdir
from ._cosmo_hammer import CosmoHammerSampler, LikelihoodComputationChain
from cosmoHammer.util import Params
from py21cmmc._cosmo_hammer.storage import DataSampleFileUtil, HDFStorageUtil


def run_mcmc(redshift, parameters, datadir='.', model_name = '21cmfast',
             box_dim = {}, flag_options={}, astro_params={}, cosmo_params={},
             extra_core_modules=[], likelihood_modules = [], reuse_burnin=True,
             store = [], continue_sampling=True,
             **mcmc_options):

    flag_options['redshifts'] = redshift
    box_dim['DIREC'] = datadir

    file_prefix = path.join(datadir, model_name)

    try:
        mkdir(datadir)
    except:
        pass

    # Setup parameters.
    params = Params(*[(k, v[1:]) for k, v in parameters.items()])

    # Setup the Core Module
    core_module = Core21cmFastModule(
        params.keys,
        store = store,
        box_dim = box_dim, flag_options=flag_options,
        astro_params=astro_params, cosmo_params=cosmo_params
    )

    # Write the parameter names to a file, might be useful later.
    param_names = [v[0] for v in parameters.values()]
    with open(path.join(datadir, model_name+"_parameter_names.txt"), 'w') as f:
        f.write("\n".join(param_names)+'\n')

    # Get all the likelihood modules.
    likelihoods = []
    for lk in likelihood_modules:
        if isinstance(lk, tuple):
            if isinstance(lk[0], str):
                likelihoods += [getattr(sys.modules['py21cmmc.likelihood'],
                                        "Likelihood%s" % lk[0])(box_dim = box_dim,
                                                                astro_params=astro_params,
                                                                cosmo_params=cosmo_params,
                                                                flag_options=flag_options,
                                                                **lk[1])]
            else:
                likelihoods += [lk[0](
                    box_dim = box_dim,
                    astro_params=astro_params,
                    cosmo_params=cosmo_params,
                    flag_options=flag_options,**lk[1]
                )]

        else:
            likelihoods += [lk]

    chain = LikelihoodComputationChain(min=params[:, 1], max=params[:, 2])

    # Add default plus extra core modules.
    chain.addCoreModule(core_module)
    for cm in extra_core_modules:
        chain.addCoreModule(cm)

    for lk in likelihoods:
        chain.addLikelihoodModule(lk)

    chain.setup()


    sampler = CosmoHammerSampler(
        continue_sampling=continue_sampling,
        params=params,
        likelihoodComputationChain=chain,
        storageUtil= HDFStorageUtil(file_prefix),
        filePrefix=file_prefix,
        reuseBurnin=reuse_burnin,
        **mcmc_options
    )

    # The sampler writes to file, so no need to save anything ourselves.
    sampler.startSampling()

    return sampler
