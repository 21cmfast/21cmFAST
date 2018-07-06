"""
Module that contains the command line app.
"""
import click
import yaml
import pickle
from . import initial_conditions, perturb_field, CosmoParams, UserParams#run_21cmfast

import sys
from os import path
import os
# from .mcmc import run_mcmc


def _get_config(config=None):
    if config is None:
        config = path.expanduser(path.join("~", ".21CMMC", "runconfig_example.yml"))

    with open(config, "r") as f:
        cfg = yaml.load(f)

    return cfg


def _ctx_to_dct(args):
    return {args[i][2:]: args[i + 1] for i in range(0, len(args), 2)}


def _update(obj, ctx):
    # Try to use the extra arguments as an override of config.
    for k,v in ctx.items():
        try:
            val = getattr(obj, "_"+k)
            setattr(obj, "_" + k, type(val)(v))
        except AttributeError:
            try:
                val = getattr(obj, k)
                setattr(obj, k, type(val)(v))
            except AttributeError:
                pass


main = click.Group()


# @main.command()
# @click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None,
#               help="Path to the configuration file (default ~/.py21cmmc/example_config.yml)")
# @click.option('--write/--no-write', default=True,
#               help="Whether to write out intermediate files (from init and perturb_field) for later use")
# @click.option("--regen/--no-regen", default=False,
#               help="Whether to force regeneration of init/perturb files if they already exist.")
# @click.option("--outdir", type=click.Path(exists=True, dir_okay=True), default=None,
#               help="directory to write data and plots to -- must exist.")
# @click.option("--datafile", type=str, default=None, help="name of outputted datafile (default empty -- no writing)")
# @click.option("--plot", multiple=True, help="types of pdf plots to save. Valid values are [global, power, slice]")
# @click.option("--perturb/--no-perturb", default=True,
#               help="Whether to run the perturbed field calculation")
# @click.option("--ionize/--no-ionize", default=True,
#               help="Whether to run the ionization calculation")
# def single(config, write, regen, outdir, datafile, plot, perturb, ionize):
#     return run_21cmfast()


@main.command(
    context_settings = dict(  # Doing this allows arbitrary options to override config
        ignore_unknown_options=True,
        allow_extra_args=True
    )
)
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to the configuration file (default ~/.21CMMC/runconfig_single.yml)")
@click.option("--regen/--no-regen", default=False,
              help="Whether to force regeneration of init/perturb files if they already exist.")
@click.option("--direc", type=click.Path(exists=True, dir_okay=True), default=None,
              help="directory to write data and plots to -- must exist.")
@click.option("--fname", type=click.Path(dir_okay=False), default=None,
              help="filename of output.")
@click.option("--match-seed/--no-match-seed", default=False,
              help="whether to force the random seed to also match in order to be considered a match")
@click.pass_context
def init(ctx, config, regen, direc, fname, match_seed):
    """
    Run a single iteration of 21cmFAST init, saving results to file.
    The same operation can be done with ``py21cmmc single --no-perturb``.
    """
    cfg = _get_config(config)

    # Set user/cosmo params from config.
    user_params = UserParams(**cfg['user_params'])
    cosmo_params = CosmoParams(**cfg['cosmo_params'])

    # Try to use the extra arguments as an override of config.
    if ctx.args:
        ctx = _ctx_to_dct(ctx.args)
        _update(user_params, ctx)
        _update(cosmo_params, ctx)

    initial_conditions(
        user_params, cosmo_params,
        regenerate=regen, write=True, direc=direc, fname=fname, match_seed=match_seed
    )


@main.command(
    context_settings = dict(  # Doing this allows arbitrary options to override config
        ignore_unknown_options=True,
        allow_extra_args=True
    )
)
@click.argument("redshift", type=float)
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to the configuration file (default ~/.21CMMC/runconfig_single.yml)")
@click.option("--regen/--no-regen", default=False,
              help="Whether to force regeneration of init/perturb files if they already exist.")
@click.option("--direc", type=click.Path(exists=True, dir_okay=True), default=None,
              help="directory to write data and plots to -- must exist.")
@click.option("--fname", type=click.Path(dir_okay=False), default=None,
              help="filename of output.")
@click.option("--match-seed/--no-match-seed", default=False,
              help="whether to force the random seed to also match in order to be considered a match")
@click.pass_context
def perturb(ctx, redshift, config, regen, direc, fname, match_seed):
    """
    Run 21cmFAST perturb_field at the specified redshift, saving results to file.
    The same operation can be done with ``py21cmmc single --no-ionize``.
    """
    cfg = _get_config(config)

    # Set user/cosmo params from config.
    user_params = UserParams(**cfg['user_params'])
    cosmo_params = CosmoParams(**cfg['cosmo_params'])

    # Try to use the extra arguments as an override of config.
    if ctx.args:
        ctx = _ctx_to_dct(ctx.args)
        _update(user_params, ctx)
        _update(cosmo_params, ctx)

    perturb_field(
        redshift, user_params=user_params, cosmo_params=cosmo_params,
        regenerate=regen, write=True, direc=direc, fname=fname, match_seed=match_seed
    )