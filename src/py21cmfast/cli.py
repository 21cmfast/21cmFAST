"""Module that contains the command line app."""

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Annotated, Literal

import attrs
import matplotlib.pyplot as plt
import numpy as np
from cyclopts import App, ArgumentCollection, Group, Parameter
from cyclopts import types as cyctp
from cyclopts import validators as vld
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

from . import __version__, plotting
from .drivers.coeval import generate_coeval
from .drivers.lightcone import run_lightcone
from .drivers.single_field import compute_initial_conditions
from .io.caching import OutputCache, RunCache
from .lightconers import RectilinearLightconer
from .run_templates import (
    list_templates,
    write_template,
)
from .wrapper.inputs import (
    AstroOptions,
    AstroParams,
    CosmoParams,
    InputParameters,
    MatterOptions,
    SimulationOptions,
)

cns = Console()

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("21cmFAST")

AVAILABLE_TEMPLATES: list[str] = [str(tmpl["name"]) for tmpl in list_templates()]

app = App()
app.command(
    cfg := App(name="template", help="Manage 21cmFAST configuration files/templates.")
)
app.command(run := App(name="run", help="Run 21cmFAST simulations."))
app.command(dev := App(name="dev", help="Run development tasks."))


def print_banner():
    """Print a cool banner for 21cmFAST using rich."""
    panel = Panel(
        Text.from_markup("[orange]:duck: 21cmFAST :duck:", justify="center"),
        subtitle=f"v{__version__}",
        style="cyan",
    )
    cns.print(panel)


def _at_least_one_validator(argument_collection: ArgumentCollection):
    if not any(argument.has_tokens for argument in argument_collection):
        raise ValueError(
            f"Must specify one of: {{{(', ').join(arg.name for arg in argument_collection)}}}"
        )


_paramgroup = Group(
    "Simulation Parameters (set one)",
    validator=(vld.MutuallyExclusive(), _at_least_one_validator),
)


@Parameter(name="*")
@dataclass(frozen=True, kw_only=True)
class ParameterSelection:
    """Common options for choosing simulation parameters.

    You can either specify a TOML file, or a template name.
    """

    param_file: Annotated[cyctp.ExistingTomlPath, Parameter(group=_paramgroup)] = None
    "Path to a TOML configuration file (can be generated with `21cmfast template create`)."

    template: Annotated[Literal[*AVAILABLE_TEMPLATES], Parameter(group=_paramgroup)] = (
        None
    )
    "The name of a valid builtin template (see available with `21cmfast template avail`)."


@Parameter(name="*")
@dataclass(kw_only=True)
class RunParams:
    """Common parameters for run functions."""

    param_selection: ParameterSelection = ParameterSelection()

    seed: int = 42
    "Random seed used to generate data."

    regenerate: Annotated[bool, Parameter(name=["--regenerate", "--regen"])] = False
    "Whether to regenerate all data, even if found in cache."

    cachedir: cyctp.ExistingDirectory = Path()
    "Where to write and search for cached items."

    verbosity: Annotated[
        Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        Parameter(alias=("-v", "--v")),
    ] = "WARNING"


def _param_cls_factory(cls):
    out = attrs.make_class(
        cls.__name__,
        {
            fld.name: attrs.field(default=None, type=fld.type)
            for fld in attrs.fields(cls)
        },
        kw_only=True,
        frozen=True,
    )
    out.__doc__ = cls.__doc__.replace(":class:", "")
    return out


_AstroOptions = _param_cls_factory(AstroOptions)
_AstroParams = _param_cls_factory(AstroParams)
_CosmoParams = _param_cls_factory(CosmoParams)
_MatterOptions = _param_cls_factory(MatterOptions)
_SimulationOptions = _param_cls_factory(SimulationOptions)


@Parameter(name="*")
@dataclass
class Parameters:
    """A trimmed-down version of InputParameters with all defaults of None."""

    simulation_options: Annotated[
        _SimulationOptions, Parameter(name="*", group="SimulationOptions")
    ] = _SimulationOptions()
    astro_options: Annotated[
        _AstroOptions, Parameter(name="*", group="AstroOptions")
    ] = _AstroOptions()
    astro_params: Annotated[_AstroParams, Parameter(name="*", group="AstroParams")] = (
        _AstroParams()
    )
    cosmo_params: Annotated[_CosmoParams, Parameter(name="*", group="CosmoParams")] = (
        _CosmoParams()
    )
    matter_options: Annotated[
        _MatterOptions, Parameter(name="*", group="MatterOptions")
    ] = _MatterOptions()


def _get_inputs(
    options: RunParams | ParameterSelection, params: Parameters
) -> InputParameters:
    # Set user/cosmo params from config.
    inputs = InputParameters.from_template(
        options.template or options.param_file,
        random_seed=getattr(options, "seed", 42),
    )

    # kwargs from params:
    kwargs = {}
    for field in fields(Parameters):
        this = getattr(params, field.name)
        kwargs |= {
            name: val for name, val in attrs.asdict(this).items() if val is not None
        }

    return inputs.evolve_input_structs(**kwargs)


@cfg.command(name="avail")
def show_configs():
    templates = list_templates()
    for template in templates:
        name = template["name"]

        cns.print(f"[bold purple]{name}[/bold purple]", end=" ")
        aliases = template["aliases"]
        aliases.remove(name)  # Remove the name from aliases if present

        if aliases:
            cns.print(f"[purple] | {' | '.join(aliases)}")
        else:
            cns.print()

        cns.print(f"\t{template['description']}")


@cfg.command(name="create")
def template_create(
    out: Annotated[cyctp.TomlPath, Parameter(validator=(vld.Path(exists=False)))],
    param_selection: ParameterSelection = ParameterSelection(),
    user_params: Parameters = Parameters(),
):
    inputs = _get_inputs(param_selection, user_params)
    if not out.parent.exists():
        out.parent.mkdir(exist_ok=True, parents=True)
    write_template(inputs, out)
    cns.print(f":duck:[green] Wrote new template file at [purple]{out}")


def _run_setup(
    options: RunParams, params: Parameters, zmin: float | None = None
) -> InputParameters:
    print_banner()

    logger.setLevel(options.verbosity)

    inputs = _get_inputs(options, params)

    if zmin is not None:
        inputs = inputs.with_logspaced_redshifts(zmin=zmin)

    config_file = options.cachedir / "config.toml"
    write_template(inputs, config_file)
    if (
        options.param_file is not None
        and options.param_file.resolve() != config_file.resolve()
    ):
        cns.print(f":duck: [green]Wrote full configuration to [purple]{config_file}")
    return inputs


@run.command
def params(params: Parameters = Parameters()):
    """Show the current simulation parameters."""
    cns.print("Usage: 21cmfast run params --help")


@run.command()
def ics(
    options: RunParams,
    params: Annotated[Parameters, Parameter(show=False, name="*")] = Parameters(),
):
    """Run a single iteration of 21cmFAST init, saving results to file.

    To see the full list of simulation parameter options, run
    `21cmfast run params --help`
    """
    inputs = _run_setup(options, params)
    cache = OutputCache(options.cachedir)
    rc = RunCache.from_inputs(inputs=inputs, cache=cache)

    if rc.InitialConditions.exists():
        if options.regenerate:
            cns.print(
                "[yellow]:warning: Initial conditions already exist, but regeneration is requested. Overriding."
            )
        else:
            cns.print(
                f"[green]:duck: Initial conditions already exist at [purple]'{rc.InitialConditions}'[/purple], skipping computation."
            )
            return

    compute_initial_conditions(
        inputs=inputs,
        regenerate=options.regenerate,
        write=True,
        cache=OutputCache(options.cachedir),
    )

    cns.print(f"[green]:duck: Saved initial conditions to {rc.InitialConditions}.")


@run.command()
def coeval(
    redshifts: list[float],
    options: RunParams,
    params: Annotated[Parameters, Parameter(show=False)],
    out: Annotated[cyctp.ExistingDirectory, Parameter(name=("--out", "-o"))] = Path(),
    save_all_redshifts: Annotated[
        bool, Parameter(name=("--save-all-redshifts", "-a", "--all"))
    ] = False,
    min_evolved_redshift: Annotated[
        float, Parameter(name=("--zmin-evolution", "--zmin"))
    ] = 5.5,
):
    """Generate coeval cubes at given redshifts.

    Parameters
    ----------
    redshifts
        The redshifts at which to generate the coeval boxes.
    out
        The path at which to save the coeval boxes. The coeval data at each
        redshift will be saved to the filename "coeval_z{redshift:.2f}.h5".
    save_all_redshifts
        Whether to save all redshifts in `node_redshifts` (i.e. all those
        in the evolution of the simulation), or only those in the redshifts given.
    min_evolved_redshift
        The minimum redshift down to which to evolve the simulation. For some simulation
        configurations, this is not used at all, while for others it will subtly change
        the evolution.
    """
    inputs = _run_setup(options, params, zmin=min_evolved_redshift)

    for coeval, in_outputs in generate_coeval(
        out_redshifts=redshifts,
        inputs=inputs,
        regenerate=options.regenerate,
        write=True,
        cache=OutputCache(options.cachedir),
        progressbar=True,
    ):
        if not in_outputs and not save_all_redshifts:
            continue

        outfile = out / f"coeval_z{coeval.redshift:.2f}.h5"

        coeval.save(outfile)
        cns.print(
            f"[green]:duck:[/green] Saved z={coeval.redshift:.2f} coeval box to {outfile}."
        )


@run.command()
def lightcone(
    options: RunParams,
    params: Annotated[Parameters, Parameter(show=False)],
    redshift_range: tuple[float, float] = (6.0, 30.0),
    out: Annotated[
        Path, Parameter(validator=(vld.Path(exists=False, ext=("h5",)),))
    ] = Path("lightcone.h5"),
    lightcone_quantities: Annotated[tuple[str], Parameter(name=("--lq",))] = (
        "brightness_temp",
    ),
    global_quantities: Annotated[tuple[str], Parameter(name=("--gq",))] = (
        "global_xHI",
        "global_brightness_temp",
    ),
    min_evolved_redshift: Annotated[
        float, Parameter(name=("--zmin-evolution", "--zmin"))
    ] = 5.5,
):
    """Generate a lightcone between given redshifts.

    Parameters
    ----------
    redshift_range
        The redshifts between which to generate the lightcone.
    out
        The filename to which to save the lightcone data.
    lightcone_quantities
        Computed fields to generate lightcones for.
    global_quantities
        Fields for which to compute the globally-averaged signal.
    min_evolved_redshift
        The minimum redshift down to which to evolve the simulation. For some simulation
        configurations, this is not used at all, while for others it will subtly change
        the evolution.
    """
    if not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)

    inputs = _run_setup(options, params, min_evolved_redshift)

    # For now, always use the old default lightconing algorithm
    lcn = RectilinearLightconer.between_redshifts(
        min_redshift=redshift_range[0],
        max_redshift=redshift_range[1],
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
        quantities=lightcone_quantities,
        global_quantities=global_quantities,
    )

    lc = run_lightcone(
        lightconer=lcn,
        inputs=inputs,
        regenerate=options.regenerate,
        write=True,
        cache=OutputCache(options.cachedir),
        progressbar=True,
    )

    lc.save(out)

    cns.print(f"[green]:duck: Saved Lightcone to {out}.")


@dev.command(name="feature")
def pr_feature(
    params: Parameters,
    options: RunParams,
    redshift_range: tuple[float, float] = (6.0, 30.0),
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

    inputs_default = _run_setup(options, Parameters())
    inputs_new = _run_setup(options, params)

    # For now, always use the old default lightconing algorithm
    lcn = RectilinearLightconer.between_redshifts(
        min_redshift=redshift_range[0],
        max_redshift=redshift_range[1],
        resolution=inputs_default.simulation_options.cell_size,
        cosmo=inputs_default.cosmo_params.cosmo,
        quantities=("brightness_temp",),
        global_quantities=("global_xHI", "global_brightness_temp"),
    )

    cns.print("Running default lightcone...")

    lc_default = run_lightcone(
        lightconer=lcn,
        inputs=inputs_default,
        regenerate=options.regenerate,
        write=True,
        cache=OutputCache(options.cachedir),
        progressbar=True,
    )

    cns.print("Running lightcone with new feature...")
    lc_new = run_lightcone(
        lightconer=lcn,
        inputs=inputs_new,
        regenerate=options.regenerate,
        write=True,
        cache=OutputCache(options.cachedir),
        progressbar=True,
    )

    cns.print("Plotting lightcone slices...")
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

    cns.print("Plotting lightcone history...")
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

    cns.print("Plotting power spectra history...")
    p_default = []
    p_new = []
    z = []
    thickness = 200  # Mpc
    ncells = int(thickness / lc_new.cell_size)
    chunk_size = lc_new.cell_size * ncells
    start = 0
    cns.print(ncells)
    while start + ncells <= lc_new.shape[-1]:
        pd, k = powerbox.get_power(
            lc_default.brightness_temp[:, :, start : start + ncells],
            (*lc_default.lightcone_dimensions[:2], chunk_size),
        )
        p_default.append(pd)

        pn, k = powerbox.get_power(
            lc_new.brightness_temp[:, :, start : start + ncells],
            (*lc_new.lightcone_dimensions[:2], chunk_size),
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


if __name__ == "__main__":
    app()
