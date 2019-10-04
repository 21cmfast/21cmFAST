"""
Module that contains the command line app.
"""
import inspect
import warnings
from os import path, remove
import builtins

import click
import yaml

from . import wrapper as lib, cache_tools
from . import _cfg
from . import plotting

import matplotlib.pyplot as plt
import numpy as np
import powerbox
import logging


def _get_config(config=None):
    if config is None:
        config = path.expanduser(path.join("~", ".21cmfast", "runconfig_example.yml"))

    with open(config, "r") as f:
        cfg = yaml.load(f)

    return cfg


def _ctx_to_dct(args):
    dct = {}
    j = 0
    while j < len(args):
        arg = args[j]
        if "=" in arg:
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
        ignore_unknown_options=True, allow_extra_args=True
    )
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the configuration file (default ~/.21cmfast/runconfig_single.yml)",
)
@click.option(
    "--regen/--no-regen",
    default=False,
    help="Whether to force regeneration of init/perturb files if they already exist.",
)
@click.option(
    "--direc",
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    help="directory to write data and plots to -- must exist.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="specify a random seed for the initial conditions",
)
@click.pass_context
def init(ctx, config, regen, direc, seed):
    """
    Run a single iteration of 21cmFAST init, saving results to file.
    """
    cfg = _get_config(config)

    # Set user/cosmo params from config.
    user_params = lib.UserParams(**cfg.get("user_params", {}))
    cosmo_params = lib.CosmoParams(**cfg.get("cosmo_params", {}))

    _override(ctx, user_params, cosmo_params)

    lib.initial_conditions(
        user_params=user_params,
        cosmo_params=cosmo_params,
        regenerate=regen,
        write=True,
        direc=direc,
        random_seed=seed,
    )


@main.command(
    context_settings=dict(  # Doing this allows arbitrary options to override config
        ignore_unknown_options=True, allow_extra_args=True
    )
)
@click.argument("redshift", type=float)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the configuration file (default ~/.21cmfast/runconfig_single.yml)",
)
@click.option(
    "--regen/--no-regen",
    default=False,
    help="Whether to force regeneration of init/perturb files if they already exist.",
)
@click.option(
    "--direc",
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    help="directory to write data and plots to -- must exist.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="specify a random seed for the initial conditions",
)
@click.pass_context
def perturb(ctx, redshift, config, regen, direc, seed):
    """
    Run 21cmFAST perturb_field at the specified redshift, saving results to file.
    """
    cfg = _get_config(config)

    # Set user/cosmo params from config.
    user_params = lib.UserParams(**cfg.get("user_params", {}))
    cosmo_params = lib.CosmoParams(**cfg.get("cosmo_params", {}))

    _override(ctx, user_params, cosmo_params)

    lib.perturb_field(
        redshift=redshift,
        user_params=user_params,
        cosmo_params=cosmo_params,
        regenerate=regen,
        write=True,
        direc=direc,
        random_seed=seed,
    )


@main.command(
    context_settings=dict(  # Doing this allows arbitrary options to override config
        ignore_unknown_options=True, allow_extra_args=True
    )
)
@click.argument("redshift", type=float)
@click.option(
    "-p",
    "--prev_z",
    type=float,
    default=None,
    help="Previous redshift (the spin temperature data must already exist for this redshift)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the configuration file (default ~/.21cmfast/runconfig_single.yml)",
)
@click.option(
    "--regen/--no-regen",
    default=False,
    help="Whether to force regeneration of init/perturb files if they already exist.",
)
@click.option(
    "--direc",
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    help="directory to write data and plots to -- must exist.",
)
@click.option(
    "-z",
    "--z-step-factor",
    type=float,
    default=inspect.signature(lib.spin_temperature).parameters["z_step_factor"].default,
    help="logarithmic steps in redshift for evolution",
)
@click.option(
    "-Z",
    "--z-heat-max",
    type=float,
    default=None,
    help="maximum redshift at which to search for heating sources",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="specify a random seed for the initial conditions",
)
@click.pass_context
def spin(ctx, redshift, prev_z, config, regen, direc, z_step_factor, z_heat_max, seed):
    """
    Run 21cmFAST spin_temperature at the specified redshift, saving results to file.
    """
    cfg = _get_config(config)

    # Set user/cosmo params from config.
    user_params = lib.UserParams(**cfg.get("user_params", {}))
    cosmo_params = lib.CosmoParams(**cfg.get("cosmo_params", {}))
    flag_options = lib.FlagOptions(**cfg.get("flag_options", {}))
    astro_params = lib.AstroParams(
        **cfg.get("astro_params", {}), INHOMO_RECO=flag_options.INHOMO_RECO
    )

    _override(ctx, user_params, cosmo_params, astro_params, flag_options)

    if z_step_factor is None and "z_step_factor" in cfg:
        z_step_factor = cfg["z_step_factor"]
    if z_heat_max is None and "z_heat_max" in cfg:
        z_heat_max = cfg["z_heat_max"]

    lib.spin_temperature(
        redshift=redshift,
        astro_params=astro_params,
        flag_options=flag_options,
        previous_spin_temp=prev_z,
        z_step_factor=z_step_factor,
        z_heat_max=z_heat_max,
        user_params=user_params,
        cosmo_params=cosmo_params,
        regenerate=regen,
        write=True,
        direc=direc,
        random_seed=seed,
    )


@main.command(
    context_settings=dict(  # Doing this allows arbitrary options to override config
        ignore_unknown_options=True, allow_extra_args=True
    )
)
@click.argument("redshift", type=float)
@click.option(
    "-p",
    "--prev_z",
    type=float,
    default=None,
    help="Previous redshift (the ionized box data must already exist for this redshift)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the configuration file (default ~/.21cmfast/runconfig_single.yml)",
)
@click.option(
    "--regen/--no-regen",
    default=False,
    help="Whether to force regeneration of init/perturb files if they already exist.",
)
@click.option(
    "--direc",
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    help="directory to write data and plots to -- must exist.",
)
@click.option(
    "-z",
    "--z-step-factor",
    type=float,
    default=inspect.signature(lib.ionize_box).parameters["z_step_factor"].default,
    help="logarithmic steps in redshift for evolution",
)
@click.option(
    "-Z",
    "--z-heat-max",
    type=float,
    default=None,
    help="maximum redshift at which to search for heating sources",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="specify a random seed for the initial conditions",
)
@click.pass_context
def ionize(
    ctx, redshift, prev_z, config, regen, direc, z_step_factor, z_heat_max, seed
):
    """
    Run 21cmFAST ionize_box at the specified redshift, saving results to file.
    """
    cfg = _get_config(config)

    # Set user/cosmo params from config.
    user_params = lib.UserParams(**cfg.get("user_params", {}))
    cosmo_params = lib.CosmoParams(**cfg.get("cosmo_params", {}))
    flag_options = lib.FlagOptions(**cfg.get("flag_options", {}))
    astro_params = lib.AstroParams(
        **cfg.get("astro_params", {}), INHOMO_RECO=flag_options.INHOMO_RECO
    )

    _override(ctx, user_params, cosmo_params, astro_params, flag_options)

    if z_step_factor is None and "z_step_factor" in cfg:
        z_step_factor = cfg["z_step_factor"]
    if z_heat_max is None and "z_heat_max" in cfg:
        z_heat_max = cfg["z_heat_max"]

    lib.ionize_box(
        redshift=redshift,
        astro_params=astro_params,
        flag_options=flag_options,
        previous_ionize_box=prev_z,
        z_step_factor=z_step_factor,
        z_heat_max=z_heat_max,
        user_params=user_params,
        cosmo_params=cosmo_params,
        regenerate=regen,
        write=True,
        direc=direc,
        random_seed=seed,
    )


@main.command(
    context_settings=dict(  # Doing this allows arbitrary options to override config
        ignore_unknown_options=True, allow_extra_args=True
    )
)
@click.argument("redshift", type=str)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the configuration file (default ~/.21cmfast/runconfig_single.yml)",
)
@click.option(
    "--regen/--no-regen",
    default=False,
    help="Whether to force regeneration of init/perturb files if they already exist.",
)
@click.option(
    "--direc",
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    help="directory to write data and plots to -- must exist.",
)
@click.option(
    "-z",
    "--z-step-factor",
    type=float,
    default=inspect.signature(lib.run_coeval).parameters["z_step_factor"].default,
    help="logarithmic steps in redshift for evolution",
)
@click.option(
    "-Z",
    "--z-heat-max",
    type=float,
    default=None,
    help="maximum redshift at which to search for heating sources",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="specify a random seed for the initial conditions",
)
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
    user_params = lib.UserParams(**cfg.get("user_params", {}))
    cosmo_params = lib.CosmoParams(**cfg.get("cosmo_params", {}))
    flag_options = lib.FlagOptions(**cfg.get("flag_options", {}))
    astro_params = lib.AstroParams(
        **cfg.get("astro_params", {}), INHOMO_RECO=flag_options.INHOMO_RECO
    )

    _override(ctx, user_params, cosmo_params, astro_params, flag_options)

    if z_step_factor is None and "z_step_factor" in cfg:
        z_step_factor = cfg["z_step_factor"]
    if z_heat_max is None and "z_heat_max" in cfg:
        z_heat_max = cfg["z_heat_max"]

    lib.run_coeval(
        redshift=redshift,
        astro_params=astro_params,
        flag_options=flag_options,
        z_step_factor=z_step_factor,
        z_heat_max=z_heat_max,
        user_params=user_params,
        cosmo_params=cosmo_params,
        regenerate=regen,
        write=True,
        direc=direc,
        random_seed=seed,
    )


@main.command(
    context_settings=dict(  # Doing this allows arbitrary options to override config
        ignore_unknown_options=True, allow_extra_args=True
    )
)
@click.argument("redshift", type=float)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the configuration file (default ~/.21cmfast/runconfig_single.yml)",
)
@click.option(
    "--regen/--no-regen",
    default=False,
    help="Whether to force regeneration of init/perturb files if they already exist.",
)
@click.option(
    "--direc",
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    help="directory to write data and plots to -- must exist.",
)
@click.option(
    "-X",
    "--max-z",
    type=float,
    default=None,
    help="maximum redshift of the stored lightcone data",
)
@click.option(
    "-z",
    "--z-step-factor",
    type=float,
    default=inspect.signature(lib.run_lightcone).parameters["z_step_factor"].default,
    help="logarithmic steps in redshift for evolution",
)
@click.option(
    "-Z",
    "--z-heat-max",
    type=float,
    default=None,
    help="maximum redshift at which to search for heating sources",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="specify a random seed for the initial conditions",
)
@click.pass_context
def lightcone(
    ctx, redshift, config, regen, direc, max_z, z_step_factor, z_heat_max, seed
):
    """
    Efficiently generate coeval cubes at a given redshift.
    """
    cfg = _get_config(config)

    # Set user/cosmo params from config.
    user_params = lib.UserParams(**cfg.get("user_params", {}))
    cosmo_params = lib.CosmoParams(**cfg.get("cosmo_params", {}))
    flag_options = lib.FlagOptions(**cfg.get("flag_options", {}))
    astro_params = lib.AstroParams(
        **cfg.get("astro_params", {}), INHOMO_RECO=flag_options.INHOMO_RECO
    )

    _override(ctx, user_params, cosmo_params, astro_params, flag_options)

    if z_step_factor is None and "z_step_factor" in cfg:
        z_step_factor = cfg["z_step_factor"]
    if z_heat_max is None and "z_heat_max" in cfg:
        z_heat_max = cfg["z_heat_max"]

    lib.run_lightcone(
        redshift=redshift,
        max_redshift=max_z,
        astro_params=astro_params,
        flag_options=flag_options,
        z_step_factor=z_step_factor,
        z_heat_max=z_heat_max,
        user_params=user_params,
        cosmo_params=cosmo_params,
        regenerate=regen,
        write=True,
        direc=direc,
        random_seed=seed,
    )


def _query(direc=None, kind=None, md5=None, seed=None, clear=False):
    cls = list(
        cache_tools.query_cache(direc=direc, kind=kind, hash=md5, seed=seed, show=False)
    )

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
            direc = direc or path.expanduser(_cfg.config["boxdir"])
            remove(path.join(direc, file))


@main.command()
@click.option(
    "-d",
    "--direc",
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    help="directory to write data and plots to -- must exist.",
)
@click.option("-k", "--kind", type=str, default=None, help="filter by kind of data.")
@click.option("-m", "--md5", type=str, default=None, help="filter by md5 hash")
@click.option("-s", "--seed", type=str, default=None, help="filter by random seed")
@click.option(
    "--clear/--no-clear",
    default=False,
    help="remove all data sets returned by this query.",
)
def query(direc, kind, md5, seed, clear):
    _query(direc, kind, md5, seed, clear)


@main.command()
@click.argument("param", type=str)
@click.argument("value", type=str)
@click.option(
    "-s",
    "--struct",
    type=click.Choice(["flag_options", "cosmo_params", "user_params", "astro_params"]),
    default="flag_options",
    help="struct in which the new feature exists",
)
@click.option(
    "-t",
    "--vtype",
    type=click.Choice(["bool", "float", "int"]),
    default="bool",
    help="type of the new parameter",
)
@click.option(
    "-l/-c",
    "--lightcone/--coeval",
    default=True,
    help="whether to use a lightcone for comparison",
)
@click.option(
    "-z", "--redshift", type=float, default=6.0, help="redshift of the comparison boxes"
)
@click.option(
    "-Z",
    "--max-redshift",
    type=float,
    default=30,
    help="maximum redshift of the comparison lightcone",
)
@click.option("-r", "--random-seed", type=int, default=12345, help="random seed to use")
@click.option("-v", "--verbose", count=True)
@click.option(
    "-g/-G",
    "--regenerate/--cache",
    default=True,
    help="whether to regenerate the boxes",
)
def pr_feature(
    param,
    value,
    struct,
    vtype,
    lightcone,
    redshift,
    max_redshift,
    random_seed,
    verbose,
    regenerate,
):
    """
    Create a standard set of plots comparing a default simulation against a
    simulation with a new feature. The new feature is switched on by setting
    PARAM to VALUE.

    Plots are saved in the current directory, with the prefix "pr_feature".
    """
    lvl = [logging.WARNING, logging.INFO, logging.DEBUG][verbose]
    logger = logging.getLogger("21cmFAST")
    logger.setLevel(lvl)
    value = getattr(builtins, vtype)(value)

    structs = dict(
        user_params=dict(HII_DIM=128, BOX_LEN=250),
        #        user_params=dict(HII_DIM=35, BOX_LEN=100),
        flag_options=dict(USE_TS_FLUCT=True),
        cosmo_params=dict(),
        astro_params=dict(),
    )

    if lightcone:
        print("Running default lightcone...")
        lc_default = lib.run_lightcone(
            redshift=redshift,
            max_redshift=max_redshift,
            random_seed=random_seed,
            regenerate=regenerate,
            **structs,
        )
        structs[struct][param] = value

        print("Running lightcone with new feature...")
        lc_new = lib.run_lightcone(
            redshift=redshift,
            max_redshift=max_redshift,
            random_seed=random_seed,
            regenerate=regenerate,
            **structs,
        )

        print("Plotting lightcone slices...")
        for field in ["brightness_temp"]:
            fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)

            vmin = -150
            vmax = 30

            plotting.lightcone_sliceplot(
                lc_default, ax=ax[0], fig=fig, vmin=vmin, vmax=vmax
            )
            ax[0].set_title("Default")

            plotting.lightcone_sliceplot(
                lc_new, ax=ax[1], fig=fig, cbar=False, vmin=vmin, vmax=vmax
            )
            ax[1].set_title("New")

            plotting.lightcone_sliceplot(
                lc_default, lightcone2=lc_new, cmap="bwr", ax=ax[2], fig=fig
            )
            ax[2].set_title("Difference")

            plt.savefig("pr_feature_lighcone_2d_{}.pdf".format(field))

        def rms(x, axis=None):
            return np.sqrt(np.mean(x ** 2, axis=axis))

        print("Plotting lightcone history...")
        fig, ax = plt.subplots(4, 1, sharex=True, gridspec_kw={"hspace": 0.05})
        ax[0].plot(lc_default.node_redshifts, lc_default.global_xHI, label="Default")
        ax[0].plot(lc_new.node_redshifts, lc_new.global_xHI, label="New")
        ax[0].set_ylabel(r"$x_{\rm HI}$")
        ax[0].legend()

        ax[1].plot(
            lc_default.node_redshifts,
            lc_default.global_brightness_temp,
            label="Default",
        )
        ax[1].plot(lc_new.node_redshifts, lc_new.global_brightness_temp, label="New")
        ax[1].set_ylabel("$T_b$ [K]")
        ax[3].set_xlabel("z")

        rms_diff = rms(lc_default.brightness_temp, axis=(0, 1)) - rms(
            lc_new.brightness_temp, axis=(0, 1)
        )
        ax[2].plot(lc_default.lightcone_redshifts, rms_diff, label="RMS")
        ax[2].plot(
            lc_new.node_redshifts,
            lc_default.global_xHI - lc_new.global_xHI,
            label="$x_{HI}$",
        )
        ax[2].plot(
            lc_new.node_redshifts,
            lc_default.global_brightness_temp - lc_new.global_brightness_temp,
            label="$T_b$",
        )
        ax[2].legend()
        ax[2].set_ylabel("Differences")

        diff_rms = rms(lc_default.brightness_temp - lc_new.brightness_temp, axis=(0, 1))
        ax[3].plot(lc_default.lightcone_redshifts, diff_rms)
        ax[3].set_ylabel("RMS of Diff.")

        plt.savefig("pr_feature_history.pdf")

        print("Plotting power spectra history...")
        p_default = []
        p_new = []
        z = []
        thickness = 200  # Mpc
        ncells = int(thickness / lc_new.cell_size)
        chunk_size = lc_new.cell_size * ncells
        start = 0
        print(ncells)
        while start + ncells <= lc_new.shape[-1]:
            pd, k = powerbox.get_power(
                lc_default.brightness_temp[:, :, start : start + ncells],
                lc_default.lightcone_dimensions[:2] + (chunk_size,),
            )
            p_default.append(pd)

            pn, k = powerbox.get_power(
                lc_new.brightness_temp[:, :, start : start + ncells],
                lc_new.lightcone_dimensions[:2] + (chunk_size,),
            )
            p_new.append(pn)
            z.append(lc_new.lightcone_redshifts[start])

            start += ncells

        p_default = np.array(p_default).T
        p_new = np.array(p_new).T

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].set_yscale("log")

        inds = [
            np.where(np.abs(k - 0.1) == np.abs(k - 0.1).min())[0][0],
            np.where(np.abs(k - 0.2) == np.abs(k - 0.2).min())[0][0],
            np.where(np.abs(k - 0.5) == np.abs(k - 0.5).min())[0][0],
            np.where(np.abs(k - 1) == np.abs(k - 1).min())[0][0],
        ]

        for i, (pdef, pnew, kk) in enumerate(
            zip(p_default[inds], p_new[inds], k[inds])
        ):
            ax[0].plot(
                z, pdef, ls="--", label="k={:.2f}".format(kk), color="C{}".format(i)
            )
            ax[0].plot(z, pnew, ls="-", color="C{}".format(i))
            ax[1].plot(z, np.log10(pdef / pnew), ls="-", color="C{}".format(i))
        ax[1].set_xlabel("z")
        ax[0].set_ylabel(r"$\Delta^2 [{\rm mK}^2]$")
        ax[1].set_ylabel(r"log ratio of $\Delta^2 [{\rm mK}^2]$")
        ax[0].legend()

        plt.savefig("pr_feature_power_history.pdf")

    else:
        raise NotImplementedError()
