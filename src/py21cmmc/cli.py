"""
Module that contains the command line app.
"""
import click
from . import plotting as plts
import yaml
import pickle
from ._21cmfast import run_21cmfast

import sys
from os import path
import os
from .mcmc import run_mcmc


# def _get_config(config=None):
#     if config is None:
#         config = path.expanduser(path.join("~", ".py21cmmc", "example_config.yml"))
#
#     with open(config, "r") as f:
#         cfg = yaml.load(f)
#
#     return cfg


main = click.Group()


@main.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to the configuration file (default ~/.py21cmmc/example_config.yml)")
@click.option('--write/--no-write', default=True,
              help="Whether to write out intermediate files (from init and perturb_field) for later use")
@click.option("--regen/--no-regen", default=False,
              help="Whether to force regeneration of init/perturb files if they already exist.")
@click.option("--outdir", type=click.Path(exists=True, dir_okay=True), default=None,
              help="directory to write data and plots to -- must exist.")
@click.option("--datafile", type=str, default=None, help="name of outputted datafile (default empty -- no writing)")
@click.option("--plot", multiple=True, help="types of pdf plots to save. Valid values are [global, power, slice]")
@click.option("--perturb/--no-perturb", default=True,
              help="Whether to run the perturbed field calculation")
@click.option("--ionize/--no-ionize", default=True,
              help="Whether to run the ionization calculation")
def single(config, write, regen, outdir, datafile, plot, perturb, ionize):
    return run_21cmfast()


@main.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to the configuration file (default ~/.py21cmmc/example_config.yml)")
@click.option("--regen/--no-regen", default=False,
              help="Whether to force regeneration of init/perturb files if they already exist.")
@click.option("--outdir", type=click.Path(exists=True, dir_okay=True), default=None,
              help="directory to write data and plots to -- must exist.")
def init(config, regen, outdir):
    """
    Run a single iteration of 21cmFAST init, saving results to file.
    The same operation can be done with ``py21cmmc single --no-perturb``.
    """
    return run_21cmfast(config, write=True, regen=regen, outdir=outdir, datafile=None, plot=[],
                        run_perturb=False, run_ionize=False)


@main.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to the configuration file (default ~/.py21cmmc/example_config.yml)")
@click.option("--regen/--no-regen", default=False,
              help="Whether to force regeneration of init/perturb files if they already exist.")
@click.option("--outdir", type=click.Path(exists=True, dir_okay=True), default=None,
              help="directory to write data to.")
def perturb_field(config, regen, outdir):
    "Run a single iteration of 21cmFAST init, saving results to file."
    return run_21cmfast(config, write=True, regen=regen, outdir=outdir, datafile=None, plot=[],
                        run_perturb=True, run_ionize=False)