"""Module that contains the command line app."""

import builtins
import contextlib
import logging
import warnings
from os import path
from pathlib import Path

import attrs
import click
import matplotlib.pyplot as plt
import numpy as np
import tomllib

from . import plotting
from .drivers.coeval import run_coeval
from .drivers.lightcone import run_lightcone
from .drivers.single_field import (
    compute_initial_conditions,
    compute_ionization_field,
    compute_spin_temperature,
    perturb_field,
)
from .io.caching import OutputCache
from .lightcones import RectilinearLightconer
from .run_templates import create_params_from_template, load_template_file
from .wrapper._utils import camel_to_snake
from .wrapper.inputs import (
    InputParameters,
    get_logspaced_redshifts,
)
from .wrapper.outputs import (
    HaloBox,
    InitialConditions,
    IonizedBox,
    PerturbedField,
    TsBox,
    XraySourceBox,
)


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
    """Override attributes of an object using the click context."""
    # Try to use the extra arguments as an override of config.
    kk = list(ctx.keys())
    for k in kk:
        # noinspection PyProtectedMember
        if hasattr(obj, k):
            try:
                val = getattr(obj, f"_{k}")
                setattr(obj, f"_{k}", type(val)(ctx[k]))
                ctx.pop(k)
            except (AttributeError, TypeError):
                with contextlib.suppress(AttributeError):
                    val = getattr(obj, k)
                    setattr(obj, k, type(val)(ctx[k]))
                    ctx.pop(k)


def _get_inputs_from_ctx(ctx, seed, config="simple", **kwargs):
    """Construct InputStruct objects from the click context and config."""
    # these functions use ctx and kwargs to override params set in config,
    # where config is either a native template string or path to TOML file
    ctx = _ctx_to_dct(ctx.args)
    param_structs = create_params_from_template(config, **ctx)
    inputs = InputParameters(random_seed=seed, **{**param_structs, **kwargs})

    return inputs


main = click.Group()


@main.command(
    context_settings={  # Doing this allows arbitrary options to override config
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
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
    """Run a single iteration of 21cmFAST init, saving results to file.

    Parameters
    ----------
    ctx :
        A parameter from the parent CLI function to be able to override config.
    config : str
        Path to the configuration file.
    regen : bool
        Whether to regenerate all data, even if found in cache.
    direc : str
        Where to search for cached items.
    seed : int
        Random seed used to generate data.
    """
    # Set user/cosmo params from config.
    inputs = _get_inputs_from_ctx(ctx, seed, config=config)

    compute_initial_conditions(
        inputs=inputs,
        regenerate=regen,
        write=True,
        cache=OutputCache(direc),
    )


@main.command(
    context_settings={  # Doing this allows arbitrary options to override config
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
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
    """Run 21cmFAST perturb_field at the specified redshift, saving results to file.

    Parameters
    ----------
    ctx :
        A parameter from the parent CLI function to be able to override config.
    redshift : float
        Redshift at which to generate perturbed field.
    config : str
        Path to the configuration file.
    regen : bool
        Whether to regenerate all data, even if found in cache.
    direc : str
        Where to search for cached items.
    seed : int
        Random seed used to generate data.
    """
    inputs = _get_inputs_from_ctx(ctx, seed, config=config)
    ics = InitialConditions(inputs=inputs)

    perturb_field(
        redshift=redshift,
        inputs=inputs,
        initial_conditions=ics.read(direc),
        regenerate=regen,
        write=True,
        direc=direc,
    )


@main.command(
    context_settings={  # Doing this allows arbitrary options to override config
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
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
    "--seed",
    type=int,
    default=None,
    help="specify a random seed for the initial conditions",
)
@click.pass_context
def spin(ctx, redshift, prev_z, config, regen, direc, seed):
    """Run spin_temperature at the specified redshift, saving results to file.

    Parameters
    ----------
    ctx :
        A parameter from the parent CLI function to be able to override config.
    redshift : float
        The redshift to generate the field at.
    prev_z : float
        The redshift of a previous box from which to evolve to the current one.
    config : str
        Path to the configuration file.
    regen : bool
        Whether to regenerate all data, even if found in cache.
    direc : str
        Where to search for cached items.
    seed : int
        Random seed used to generate data.
    """
    raise NotImplementedError(
        "Spin Temerature requires inputs over specific redshifts. Use the python wrapper or coeval/lightcone"
    )
    inputs = _get_inputs_from_ctx(ctx, seed, config=config)

    ics = InitialConditions(inputs=inputs)
    pt = PerturbedField(inputs=inputs)
    xrs = XraySourceBox(inputs=inputs)
    prev_ts = TsBox(inputs=inputs.clone(redshift=prev_z))

    compute_spin_temperature(
        redshift=redshift,
        inputs=inputs,
        initial_conditions=ics.read(direc),
        perturb_field=pt.read(direc),
        xray_source_box=xrs.read(direc),
        previous_spin_temp=prev_ts.read(direc),
        regenerate=regen,
        write=True,
        direc=direc,
    )


@main.command(
    context_settings={  # Doing this allows arbitrary options to override config
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
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
    "--seed",
    type=int,
    default=None,
    help="specify a random seed for the initial conditions",
)
@click.pass_context
def ionize(ctx, redshift, prev_z, config, regen, direc, seed):
    """Run 21cmFAST ionize_box at the specified redshift, saving results to file.

    Parameters
    ----------
    ctx :
        A parameter from the parent CLI function to be able to override config.
    redshift : float
        The redshift to generate the field at.
    prev_z : float
        The redshift of a previous box from which to evolve to the current one.
    config : str
        Path to the configuration file.
    regen : bool
        Whether to regenerate all data, even if found in cache.
    direc : str
        Where to search for cached items.
    seed : int
        Random seed used to generate data.
    """
    inputs = _get_inputs_from_ctx(ctx, seed, config=config)
    if (
        inputs.astro_options.USE_TS_FLUCT
        or inputs.astro_options.INHOMO_RECO
        or inputs.matter_options.HALO_STOCHASTICITY
    ):
        raise NotImplementedError(
            "Ionized box with any of (USE_TS_FLUCT,INHOMO_RECO,HALO_STOCHASTICITY)"
            "requires inputs over specific redshifts."
            "Use the python wrapper or coeval/lightcone"
        )

    ics = InitialConditions(inputs=inputs)
    pt = PerturbedField(inputs=inputs)
    hb = HaloBox(inputs=inputs)
    prev_ion = IonizedBox(inputs=inputs.clone(redshift=prev_z))
    prev_pt = PerturbedField(inputs=inputs.clone(redshift=prev_z))
    ts = TsBox(inputs=inputs)

    compute_ionization_field(
        inputs=inputs,
        initial_conditions=ics.read(direc),
        perturbed_field=pt.read(direc),
        halobox=hb.read(direc),
        spin_temp=ts.read(direc),
        previous_perturbed_field=prev_pt.read(direc),
        previous_ionize_box=prev_ion.read(direc),
        regenerate=regen,
        write=True,
        direc=direc,
    )


@main.command(
    context_settings={  # Doing this allows arbitrary options to override config
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.argument("redshift", type=str)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the configuration file (default ~/.21cmfast/runconfig_single.yml)",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=True, file_okay=True),
    default=None,
    help="Path to output full Coeval simulation to (directory OK).",
)
@click.option(
    "--regen/--no-regen",
    default=False,
    help="Whether to force regeneration of init/perturb files if they already exist.",
)
@click.option(
    "--cache-dir",
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    help="cache directory",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="specify a random seed for the initial conditions",
)
@click.pass_context
def coeval(ctx, redshift, config, out, regen, cache_dir, seed):
    """Efficiently generate coeval cubes at a given redshift.

    Parameters
    ----------
    ctx :
        A parameter from the parent CLI function to be able to override config.
    redshift : float
        The redshift to generate the field at.
    config : str
        Path to the configuration file.
    regen : bool
        Whether to regenerate all data, even if found in cache.
    direc : str
        Where to search for cached items.
    seed : int
        Random seed used to generate data.
    """
    if out is not None:
        out = Path(out).absolute()
        if len(out.suffix) not in (2, 3) and not out.exists():
            out.mkdir()
        elif not out.parent.exists():
            out.parent.mkdir()

    try:
        redshift = [float(z.strip()) for z in redshift.split(",")]
    except TypeError as e:
        raise TypeError(
            "redshift argument must be comma-separated list of values."
        ) from e

    inputs = _get_inputs_from_ctx(ctx, seed, config=config)

    coeval = run_coeval(
        out_redshifts=redshift,
        inputs=inputs,
        regenerate=regen,
        write=True,
        cache=OutputCache(cache_dir) if cache_dir else None,
    )

    if out:
        for _i, (z, c) in enumerate(zip(redshift, coeval, strict=True)):
            if out.is_dir():
                fname = out / c.get_unique_filename()
            elif len(redshift) == 1:
                fname = out
            else:
                out = out.parent / f"{out.name}_z{z}{out.suffix}"
            c.save(fname)

        print(f"Saved Coeval box to {fname}.")


@main.command(
    context_settings={  # Doing this allows arbitrary options to override config
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.argument("redshift", type=float)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the configuration file (default ~/.21cmfast/runconfig_single.yml)",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=True, file_okay=True),
    default=None,
    help="Path to output full Lightcone to (directory OK).",
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
    "--z-step",
    type=float,
    default=1.02,
    help="maximum redshift of the stored lightcone data",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="specify a random seed for the initial conditions",
)
@click.option(
    "--lq",
    type=str,
    multiple=True,
    default=("brightness_temp",),
    help="quantities to make lightcones out of",
)
@click.pass_context
def lightcone(ctx, redshift, config, out, regen, direc, max_z, z_step, seed, lq):
    """Efficiently generate coeval cubes at a given redshift.

    Parameters
    ----------
    ctx :
        A parameter from the parent CLI function to be able to override config.
    redshift : float
        The redshift to generate the field at.
    config : str
        Path to the configuration file.
    regen : bool
        Whether to regenerate all data, even if found in cache.
    direc : str
        Where to search for cached items.
    max_z : float
        Maximum redshift to include in the produced lightcone.
    seed : int
        Random seed used to generate data.
    """
    if out is not None:
        out = Path(out).absolute()
        if len(out.suffix) not in (2, 3) and not out.exists():
            out.mkdir()
        elif not out.parent.exists():
            out.parent.mkdir()

    inputs = _get_inputs_from_ctx(
        ctx,
        seed,
        config=config,
        node_redshifts=get_logspaced_redshifts(
            min_redshift=redshift,
            max_redshift=max_z,
            z_step_factor=z_step,
        ),
    )

    # For now, always use the old default lightconing algorithm
    lcn = RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=redshift,
        max_redshift=max_z,
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
        quantities=lq,
    )

    lc = run_lightcone(
        lightconer=lcn,
        inputs=inputs,
        regenerate=regen,
        write=True,
        cache=OutputCache(direc),
    )

    if out:
        fname = out / lc.get_unique_filename() if out.is_dir() else out
        lc.save(fname)

        print(f"Saved Lightcone to {fname}.")


@main.command()
@click.argument("param", type=str)
@click.argument("value", type=str)
@click.option(
    "-s",
    "--struct",
    type=click.Choice(
        ["astro_options", "cosmo_params", "simulation_options", "astro_params"]
    ),
    default="astro_options",
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
    Create standard plots comparing a default simulation against a simulation with a new feature.

    The new feature is switched on by setting PARAM to VALUE.
    Plots are saved in the current directory, with the prefix "pr_feature".

    Parameters
    ----------
    param : str
        Name of the parameter to modify to "switch on" the feature.
    value : float
        Value to which to set it.
    struct : str
        The input parameter struct to which `param` belongs.
    vtype : str
        Type of the new parameter.
    lightcone : bool
        Whether the comparison should be done on a lightcone.
    redshift : float
        Redshift of comparison.
    max_redshift : float
        If using a lightcone, the maximum redshift in the lightcone to compare.
    random_seed : int
        Random seed at which to compare.
    verbose : int
        How verbose the output should be.
    regenerate : bool
        Whether to regenerate all data, even if it is in cache.
    """
    import powerbox

    lvl = [logging.WARNING, logging.INFO, logging.DEBUG][verbose]
    logger = logging.getLogger(__name__)
    logger.setLevel(lvl)
    value = getattr(builtins, vtype)(value)

    structs = {
        "simulation_options": {"HII_DIM": 128, "BOX_LEN": 250},
        "astro_options": {"USE_TS_FLUCT": True},
        "cosmo_params": {},
        "astro_params": {},
    }

    if lightcone:
        print("Running default lightcone...")
        lc_default = run_lightcone(
            redshift=redshift,
            max_redshift=max_redshift,
            regenerate=regenerate,
            **structs,
        )
        structs[struct][param] = value

        print("Running lightcone with new feature...")
        lc_new = run_lightcone(
            redshift=redshift,
            max_redshift=max_redshift,
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

            plt.savefig(f"pr_feature_lighcone_2d_{field}.pdf")

        def rms(x, axis=None):
            return np.sqrt(np.mean(x**2, axis=axis))

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
            zip(p_default[inds], p_new[inds], k[inds], strict=True)
        ):
            ax[0].plot(z, pdef, ls="--", label=f"k={kk:.2f}", color=f"C{i}")
            ax[0].plot(z, pnew, ls="-", color=f"C{i}")
            ax[1].plot(z, np.log10(pdef / pnew), ls="-", color=f"C{i}")
        ax[1].set_xlabel("z")
        ax[0].set_ylabel(r"$\Delta^2 [{\rm mK}^2]$")
        ax[1].set_ylabel(r"log ratio of $\Delta^2 [{\rm mK}^2]$")
        ax[0].legend()

        plt.savefig("pr_feature_power_history.pdf")

    else:
        raise NotImplementedError()
