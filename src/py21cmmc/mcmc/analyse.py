"""
Module containing functions to analyse the results of MCMC chains,a nd enable more transparent input/output of chains.
"""
from .cosmoHammer import CosmoHammerSampler
from .cosmoHammer.storage import HDFStorage

import numpy as np
import matplotlib.pyplot as plt

def get_samples(chain, indx=0):
    """
    Extract sample storage object from a chain.

    Parameters
    ----------
    chain : `~py21cmmc.mcmc.cosmoHammer.CosmoHammerSampler` or str
        Either a `LikelihoodComputationChain`, which is the output of the :func:`~mcmc.run_mcmc` function,
        or a path to an output HDF file containing the chain.

    indx : int, optional
        This is used only if `chain` is a string. It gives the index of the samples in the HDF file. Usually this is
        zero.

    Returns
    -------
    store : a `HDFStore` object.
    """
    if isinstance(chain, CosmoHammerSampler):
        return chain.storageUtil.sample_storage
    else:
        try:
            if not chain.endswith('.h5'):
                chain += '.h5'
        except AttributeError:
            raise AttributeError("chain must either be a CosmoHammerSampler instance, or str")

        return HDFStorage(chain, name="sample_%s"%indx)


def corner_plot(samples, include_lnl=True, show_guess=True, **kwargs):
    from corner import corner

    chain = samples.get_value("chain")
    lnprob = samples.get_value("log_prob")
    niter, mwalkers, nparams= chain.shape

    if show_guess:
        guess = list((samples.param_guess[0]))

    labels = list(samples.param_names)

    if include_lnl:
        plotchain = np.vstack((chain.T, np.atleast_3d(lnprob).T)).T.reshape((-1, nparams + 1))
        if show_guess:
            guess += [None]
        labels += ['lnL']
    else:
        plotchain = chain.reshape((-1, nparams))

    corner(
        plotchain,
        labels=labels,
        truths=guess if show_guess else None,
        smooth=kwargs.get("smooth", 0.75),
        smooth1d=kwargs.get("smooth1d", 1.0),
        show_titles=True,
        quantiles=kwargs.get("quantiles", [0.16, 0.5, 0.84])
    )


def trace_plot(samples, include_lnl=True, start_iter=0, thin=1, colored=False, show_guess=True):

    nwalkers, nparams = samples.shape
    if include_lnl:
        nparams += 1

    chain = samples.get_chain(thin=thin, discard=start_iter)
    lnprob = samples.get_log_prob(thin=thin, discard=start_iter)

    fig, ax = plt.subplots(nparams, 1, sharex=True, gridspec_kw={"hspace":0.05, "wspace":0.05},
                           figsize=(8, 3*nparams))

    for i in range(nwalkers):
        for j, param in enumerate(samples.param_names):
            ax[j].plot(chain[:, i, j], color='C%s' % (i % 8) if colored else 'k', alpha=0.75, lw=1)
            ax[j].set_ylabel(param)
            if show_guess and not i:
                ax[j].axhline(samples.param_guess[param][0], color='C0', lw=3)

        if include_lnl:
            ax[-1].plot(lnprob[:, i], color='C%s' % (i % 8) if colored else 'k', alpha=0.75, lw=1)
            ax[-1].set_ylabel("lnL")

    return fig, ax