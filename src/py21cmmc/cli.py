"""
Module that contains the command line app.
"""
# from .mcmc import run_mcmc
import inspect
import warnings
from os import path, remove

import click
import yaml

from . import wrapper as lib  # initial_conditions, perturb_field, CosmoParams, UserParams#run_21cmfast
from ._21cmfast import cache_tools


def _get_config(config=None):
    if config is None:
        config = path.expanduser(path.join("~", ".21CMMC", "runconfig_example.yml"))

    with open(config, "r") as f:
        cfg = yaml.load(f)

    return cfg


def _ctx_to_dct(args):
    dct = {}
    j = 0
    while j < len(args):
        arg = args[j]
        if '=' in arg:
            a = arg.split("=")
            dct[a[0].replace("--", "")] = a[-1]
            j += 1
        else:
            dct[arg.replace("--", "")] = args[j + 1]
            j += 2

    return dct


def _update(obj, ctx):
    # Try to use the extra arguments as an override of config.
    kk = list(ctx.keys())
    for k in kk:
        # noinspection PyProtectedMember
        if hasattr(obj, k):
            try:
                val = getattr(obj, "_" + k)
                setattr(obj, "_" + k, type(val)(ctx[k]))
                ctx.pop(k)
            except (AttributeError, TypeError):
                try:
                    val = getattr(obj, k)
                    setattr(obj, k, type(val)(ctx[k]))
                    ctx.pop(k)
                except AttributeError:
                    pass


def _override(ctx, *param_dicts):
    # Try to use the extra arguments as an override of config.

    if ctx.args:
        ctx = _ctx_to_dct(ctx.args)
        for p in param_dicts:
            _update(p, ctx)

        # Also update globals, always.
        _update(lib.global_params, ctx)
        if ctx:
            warnings.warn("The following arguments were not able to be set: %s" % ctx)


main = click.Group()

@main.command(
    context_settings=dict(  # Doing this allows arbitrary options to override config
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
@click.option("--seed", type=int, default=None, help="specify a random seed for the initial conditions")
@click.pass_context
def init(ctx, config, regen, direc, seed):
    """
    Run a single iteration of 21cmFAST init, saving results to file.
    """
    cfg = _get_config(config)

    # Set user/cosmo params from config.
    user_params = lib.UserParams(**cfg.get('user_params', {}))
    cosmo_params = lib.CosmoParams(**cfg.get('cosmo_params', {}))

    _override(ctx, user_params, cosmo_params)

    lib.initial_conditions(
        user_params=user_params, cosmo_params=cosmo_params,
        regenerate=regen, write=True, direc=direc, random_seed=seed
    )


@main.command(
    context_settings=dict(  # Doing this allows arbitrary options to override config
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
@click.option("--seed", type=int, default=None, help="specify a random seed for the initial conditions")
@click.pass_context
def perturb(ctx, redshift, config, regen, direc, seed):
    """
    Run 21cmFAST perturb_field at the specified redshift, saving results to file.
    """
    cfg = _get_config(config)

    # Set user/cosmo params from config.
    user_params = lib.UserParams(**cfg.get('user_params',{}))
    cosmo_params = lib.CosmoParams(**cfg.get('cosmo_params', {}))

    _override(ctx, user_params, cosmo_params)

    lib.perturb_field(
        redshift=redshift, user_params=user_params, cosmo_params=cosmo_params,
        regenerate=regen, write=True, direc=direc, random_seed=seed
    )


@main.command(
    context_settings=dict(  # Doing this allows arbitrary options to override config
        ignore_unknown_options=True,
        allow_extra_args=True
    )
)
@click.argument("redshift", type=float)
@click.option("-p", "--prev_z", type=float, default=None,
              help="Previous redshift (the spin temperature data must already exist for this redshift)")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to the configuration file (default ~/.21CMMC/runconfig_single.yml)")
@click.option("--regen/--no-regen", default=False,
              help="Whether to force regeneration of init/perturb files if they already exist.")
@click.option("--direc", type=click.Path(exists=True, dir_okay=True), default=None,
              help="directory to write data and plots to -- must exist.")
@click.option("-z", "--z-step-factor", type=float,
              default=inspect.signature(lib.spin_temperature).parameters['z_step_factor'].default,
              help="logarithmic steps in redshift for evolution")
@click.option("-Z", "--z-heat-max", type=float, default=None,
              help="maximum redshift at which to search for heating sources")
@click.option("--seed", type=int, default=None, help="specify a random seed for the initial conditions")
@click.pass_context
def spin(ctx, redshift, prev_z, config, regen, direc, z_step_factor, z_heat_max, seed):
    """
    Run 21cmFAST spin_temperature at the specified redshift, saving results to file.
    """
    cfg = _get_config(config)

    # Set user/cosmo params from config.
    user_params = lib.UserParams(**cfg.get('user_params', {}))
    cosmo_params = lib.CosmoParams(**cfg.get('cosmo_params', {}))
    flag_options = lib.FlagOptions(**cfg.get('flag_options', {}))
    astro_params = lib.AstroParams(**cfg.get('astro_params',{}), INHOMO_RECO=flag_options.INHOMO_RECO)

    _override(ctx, user_params, cosmo_params, astro_params, flag_options)

    if z_step_factor is None and "z_step_factor" in cfg:
        z_step_factor = cfg['z_step_factor']
    if z_heat_max is None and "z_heat_max" in cfg:
        z_heat_max = cfg['z_heat_max']

    lib.spin_temperature(
        redshift=redshift,
        astro_params=astro_params, flag_options=flag_options,
        previous_spin_temp=prev_z,
        z_step_factor=z_step_factor, z_heat_max=z_heat_max,
        user_params=user_params, cosmo_params=cosmo_params,
        regenerate=regen, write=True, direc=direc, random_seed=seed
    )


@main.command(
    context_settings=dict(  # Doing this allows arbitrary options to override config
        ignore_unknown_options=True,
        allow_extra_args=True
    )
)
@click.argument("redshift", type=float)
@click.option("-p", "--prev_z", type=float, default=None,
              help="Previous redshift (the ionized box data must already exist for this redshift)")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to the configuration file (default ~/.21CMMC/runconfig_single.yml)")
@click.option("--regen/--no-regen", default=False,
              help="Whether to force regeneration of init/perturb files if they already exist.")
@click.option("--direc", type=click.Path(exists=True, dir_okay=True), default=None,
              help="directory to write data and plots to -- must exist.")
@click.option("-z", "--z-step-factor", type=float,
              default=inspect.signature(lib.ionize_box).parameters['z_step_factor'].default,
              help="logarithmic steps in redshift for evolution")
@click.option("-Z", "--z-heat-max", type=float, default=None,
              help="maximum redshift at which to search for heating sources")
@click.option("--seed", type=int, default=None, help="specify a random seed for the initial conditions")
@click.pass_context
def ionize(ctx, redshift, prev_z, config, regen, direc, z_step_factor, z_heat_max,seed):
    """
    Run 21cmFAST ionize_box at the specified redshift, saving results to file.
    """
    cfg = _get_config(config)

    # Set user/cosmo params from config.
    user_params = lib.UserParams(**cfg.get('user_params', {}))
    cosmo_params = lib.CosmoParams(**cfg.get('cosmo_params', {}))
    flag_options = lib.FlagOptions(**cfg.get('flag_options', {}))
    astro_params = lib.AstroParams(**cfg.get('astro_params', {}), INHOMO_RECO=flag_options.INHOMO_RECO)

    _override(ctx, user_params, cosmo_params, astro_params, flag_options)

    if z_step_factor is None and "z_step_factor" in cfg:
        z_step_factor = cfg['z_step_factor']
    if z_heat_max is None and "z_heat_max" in cfg:
        z_heat_max = cfg['z_heat_max']

    lib.ionize_box(
        redshift=redshift,
        astro_params=astro_params, flag_options=flag_options,
        previous_ionize_box=prev_z,
        z_step_factor=z_step_factor, z_heat_max=z_heat_max,
        user_params=user_params, cosmo_params=cosmo_params,
        regenerate=regen, write=True, direc=direc, random_seed=seed
    )


@main.command(
    context_settings=dict(  # Doing this allows arbitrary options to override config
        ignore_unknown_options=True,
        allow_extra_args=True
    )
)
@click.argument("redshift", type=str)
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to the configuration file (default ~/.21CMMC/runconfig_single.yml)")
@click.option("--regen/--no-regen", default=False,
              help="Whether to force regeneration of init/perturb files if they already exist.")
@click.option("--direc", type=click.Path(exists=True, dir_okay=True), default=None,
              help="directory to write data and plots to -- must exist.")
@click.option("-z", "--z-step-factor", type=float,
              default=inspect.signature(lib.run_coeval).parameters['z_step_factor'].default,
              help="logarithmic steps in redshift for evolution")
@click.option("-Z", "--z-heat-max", type=float, default=None,
              help="maximum redshift at which to search for heating sources")
@click.option("--seed", type=int, default=None, help="specify a random seed for the initial conditions")
@click.pass_context
def coeval(ctx, redshift, config, regen, direc, z_step_factor, z_heat_max, seed):
    """
    Efficiently generate coeval cubes at a given redshift.
    """

    try:
        redshift = [float(z.strip()) for z in redshift.split(",")]
    except TypeError:
        raise TypeError("redshift argument must be comma-separated list of values.")

    cfg = _get_config(config)

    # Set user/cosmo params from config.
    user_params = lib.UserParams(**cfg.get('user_params', {}))
    cosmo_params = lib.CosmoParams(**cfg.get('cosmo_params', {}))
    flag_options = lib.FlagOptions(**cfg.get('flag_options', {}))
    astro_params = lib.AstroParams(**cfg.get('astro_params', {}), INHOMO_RECO=flag_options.INHOMO_RECO)

    _override(ctx, user_params, cosmo_params, astro_params, flag_options)

    if z_step_factor is None and "z_step_factor" in cfg:
        z_step_factor = cfg['z_step_factor']
    if z_heat_max is None and "z_heat_max" in cfg:
        z_heat_max = cfg['z_heat_max']

    lib.run_coeval(
        redshift=redshift,
        astro_params=astro_params, flag_options=flag_options,
        z_step_factor=z_step_factor, z_heat_max=z_heat_max,
        user_params=user_params, cosmo_params=cosmo_params,
        regenerate=regen, write=True, direc=direc, random_seed=seed
    )


@main.command(
    context_settings=dict(  # Doing this allows arbitrary options to override config
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
@click.option("-X", "--max-z", type=float, default=None,
              help="maximum redshift of the stored lightcone data")
@click.option("-z", "--z-step-factor", type=float,
              default=inspect.signature(lib.run_lightcone).parameters['z_step_factor'].default,
              help="logarithmic steps in redshift for evolution")
@click.option("-Z", "--z-heat-max", type=float, default=None,
              help="maximum redshift at which to search for heating sources")
@click.option("--seed", type=int, default=None, help="specify a random seed for the initial conditions")
@click.pass_context
def lightcone(ctx, redshift, config, regen, direc, max_z, z_step_factor, z_heat_max, seed):
    """
    Efficiently generate coeval cubes at a given redshift.
    """
    cfg = _get_config(config)

    # Set user/cosmo params from config.
    user_params = lib.UserParams(**cfg.get('user_params', {}))
    cosmo_params = lib.CosmoParams(**cfg.get('cosmo_params', {}))
    flag_options = lib.FlagOptions(**cfg.get('flag_options', {}))
    astro_params = lib.AstroParams(**cfg.get('astro_params', {}), INHOMO_RECO=flag_options.INHOMO_RECO)

    _override(ctx, user_params, cosmo_params, astro_params, flag_options)

    if z_step_factor is None and "z_step_factor" in cfg:
        z_step_factor = cfg['z_step_factor']
    if z_heat_max is None and "z_heat_max" in cfg:
        z_heat_max = cfg['z_heat_max']

    lib.run_lightcone(
        redshift=redshift, max_redshift=max_z,
        astro_params=astro_params, flag_options=flag_options,
        z_step_factor=z_step_factor, z_heat_max=z_heat_max,
        user_params=user_params, cosmo_params=cosmo_params,
        regenerate=regen, write=True, direc=direc, random_seed=seed
    )


def _query(direc=None, kind=None, md5=None, seed=None, clear=False):
    cls = list(cache_tools.query_cache(direc=direc, kind=kind, hash=md5, seed=seed, show=False))

    if not clear:
        print("%s Data Sets Found:" % len(cls))
        print("------------------")
    else:
        print("Removing %s data sets..." % len(cls))

    for file, c in cls:
        if not clear:
            print("  @ {%s}:" % file)
            print("  %s" % str(c))

            print()

        else:
            direc = direc or path.expanduser(lib.config['boxdir'])
            remove(path.join(direc, file))


@main.command()
@click.option("-d", "--direc", type=click.Path(exists=True, dir_okay=True), default=None,
              help="directory to write data and plots to -- must exist.")
@click.option("-k", "--kind", type=str, default=None,
              help="filter by kind of data.")
@click.option("-m", "--md5", type=str, default=None,
              help="filter by md5 hash")
@click.option("-s", "--seed", type=str, default=None,
              help="filter by random seed")
@click.option("--clear/--no-clear", default=False,
              help="remove all data sets returned by this query.")
def query(direc, kind, md5, seed, clear):
    _query(direc, kind, md5, seed, clear)
