"""Module that contains the command line app."""

import logging
import uuid
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Annotated, Literal

import attrs
import matplotlib.pyplot as plt
import numpy as np
from cyclopts import App, Group, Parameter
from cyclopts import types as cyctp
from cyclopts import validators as vld
from rich import box
from rich.columns import Columns
from rich.console import Console, group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from . import __version__, plotting
from ._templates import TOMLMode, list_templates, write_template
from .drivers import coeval as cvlmodule
from .drivers.coeval import generate_coeval
from .drivers.lightcone import run_lightcone
from .drivers.single_field import compute_initial_conditions
from .input_serialization import convert_inputs_to_dict
from .io.caching import CacheConfig, OutputCache, RunCache
from .lightconers import RectilinearLightconer
from .wrapper.inputs import (
    AstroOptions,
    AstroParams,
    CosmoParams,
    InputParameters,
    InputStruct,
    MatterOptions,
    SimulationOptions,
)

cns = Console()

FORMAT = "%(message)s"
logging.basicConfig(
    level="WARNING",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=cns)],
)
logger = logging.getLogger("py21cmfast")

AVAILABLE_TEMPLATES: list[str] = [
    str(alias) for tmpl in list_templates() for alias in tmpl["aliases"]
] + ["defaults"]

app = App()
app.command(
    cfg := App(name="template", help="Manage 21cmFAST configuration files/templates.")
)
app.command(run := App(name="run", help="Run 21cmFAST simulations."))
app.command(dev := App(name="dev", help="Run development tasks."))
app.command(pred := App(name="predict", help="Predict properties of simulations"))


def print_banner():
    """Print a cool banner for 21cmFAST using rich."""
    panel = Panel(
        Text.from_markup("[orange]:duck: 21cmFAST :duck:", justify="center"),
        subtitle=f"v{__version__}",
        style="cyan",
        padding=1,
        highlight=True,
        box=box.DOUBLE_EDGE,
    )
    cns.print(panel)
    cns.print()


_paramgroup = Group(
    "Simulation Parameters (set one)",
    validator=(vld.MutuallyExclusive(),),
)


@Parameter(name="*")
@dataclass(frozen=True, kw_only=True)
class ParameterSelection:
    """Common options for choosing simulation parameters.

    You can either specify a TOML file, or a template name.
    """

    param_file: Annotated[
        list[cyctp.ExistingTomlPath], Parameter(group=_paramgroup)
    ] = None
    "Path to a TOML configuration file (can be generated with `21cmfast template create`)."

    template: Annotated[
        list[Literal[*AVAILABLE_TEMPLATES]],
        Parameter(group=_paramgroup, alias=("--base-template"), consume_multiple=True),
    ] = field(default_factory=lambda: ["defaults"])

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
    (
        "Where to write and search for cached items and output fullspec configuration "
        "files. Note that caches will be in hash-style folders inside this folder."
    )
    cache_strategy: Literal["on", "off", "noloop", "dmfield", "last_step_only"] = (
        "dmfield"
    )
    """A strategy for which fields to cache (only used for coeval and lightcones).
Options are: (on) cache everything, (off) cache nothing (noloop) cache only boxes
outside the astrophysics evolution loop (dmfield) alias for noloop (last_step_only)
cache only boxes that are required more than one step away
    """

    outcfg: Annotated[
        Path,
        Parameter(validator=(vld.Path(file_okay=False, dir_okay=False, ext=("toml",)))),
    ] = None
    "A filepath where the full configuration TOML of the run can be written. Random hash if not specified."

    verbosity: Annotated[
        Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        Parameter(alias=("-v", "--v")),
    ] = "WARNING"
    "How much information to print out while running the simulation."

    progress: bool = True
    "Whether to display a progress bar as the simulation runs."


def _param_cls_factory(cls: type[InputStruct]) -> type:
    out = attrs.make_class(
        cls.__name__,
        {
            fld.alias: attrs.field(default=None, type=fld.type)
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
) -> tuple[InputParameters, bool]:
    pselect = options.param_selection if isinstance(options, RunParams) else options

    seed = getattr(options, "seed", 42)

    # Turn our dummy Parameters into dictionaries, ignoring any unset values (None)
    kwargs = {}
    for _field in fields(Parameters):
        this = getattr(params, _field.name)
        kwargs |= {
            name: val for name, val in attrs.asdict(this).items() if val is not None
        }

    inputs = InputParameters.from_template(
        pselect.param_file or pselect.template, random_seed=seed, **kwargs
    )

    modified = False
    if kwargs:
        # Other functions need to know if we modified the template/file params at all,
        # so we make a version without the changes to compare.
        without_kw = InputParameters.from_template(
            pselect.param_file or pselect.template, random_seed=seed
        )
        modified = without_kw != inputs

    return inputs, modified


@cfg.command(name="avail")
def cfg_avail():
    """Print all available builtin templates."""
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


def pretty_print_inputs(
    inputs: InputParameters, name: str, description: str, mode: TOMLMode = "full"
):
    inputs_dct = convert_inputs_to_dict(inputs, mode, only_cstruct_params=True)

    @group()
    def get_panel_elements():
        yield description + "\n"

        for structname, params in inputs_dct.items():
            if params:
                yield Rule(f"[bold purple]{structname}")

                keys = [f"[bold]{k}[/bold]:" for k in params]
                vals = [
                    f"{f'[green italic]{v}' if isinstance(v, str) else v}"
                    for v in params.values()
                ]
                yield Columns(
                    [val for pair in zip(keys, vals, strict=False) for val in pair],
                    expand=False,
                )
                yield ""

    cns.print(Panel(get_panel_elements(), title=name, highlight=True))


@cfg.command(name="show")
def cfg_show(
    names: Annotated[
        list[Literal[*AVAILABLE_TEMPLATES]], Parameter(accepts_keys=False)
    ],
    mode: TOMLMode = "full",
):
    """Show and describe an in-built template."""
    template_desc = [
        next(
            f"[blue bold]{t['name']}[/]: {t['description']}"
            for t in list_templates()
            if name.upper() in (tt.upper() for tt in t["aliases"])
        )
        for name in names
    ]

    inputs = InputParameters.from_template(names, random_seed=0)
    pretty_print_inputs(
        inputs=inputs,
        name=" + ".join(names),
        description="\n".join(template_desc),
        mode=mode,
    )


@cfg.command(name="create")
def template_create(
    out: Annotated[
        cyctp.TomlPath,
        Parameter(validator=(vld.Path(file_okay=False, dir_okay=False, ext=("toml",)))),
    ],
    param_selection: ParameterSelection = ParameterSelection(),
    user_params: Parameters = Parameters(),
    mode: TOMLMode = "full",
):
    """Create a new full simulation parameter template.

    The created template file (TOML file) contains *all* of the available parameters.
    To create it, use a base template (either via --param-file or --template) and
    optionally override any particular simulation parameters. To see the available
    simulation parameters and how to specify them, use `21cmfast run params --help`.
    """
    inputs, _ = _get_inputs(param_selection, user_params)
    if not out.parent.exists():
        out.parent.mkdir(exist_ok=True, parents=True)

    pretty_print_inputs(
        inputs=inputs,
        name="Custom Template",
        description="The minimal set of non-default parameters of your custom model",
        mode="minimal",
    )
    write_template(inputs, out, mode=mode)
    cns.print(f":duck:[spring_green3] Wrote new template file at [purple]{out}")


def _run_setup(
    options: RunParams,
    params: Parameters,
    zmin: float | None = None,
    force_nodez: bool = False,
) -> InputParameters:
    print_banner()

    cns.print(Rule("Setting Up The Simulation", characters="="))

    def custom_rich_warning(message: str, *args, **kwargs):
        cns.print(f"[orange1]:warning: {message}")

    cvlmodule._console = cns
    warnings.showwarning = custom_rich_warning
    logger.setLevel(options.verbosity)

    inputs, modified = _get_inputs(options, params)

    if zmin is not None and (inputs.evolution_required or force_nodez):
        inputs = inputs.with_logspaced_redshifts(zmin=zmin)

    if (
        modified
        or options.param_selection.param_file is None
        or len(options.param_selection.param_file) > 1
        or (len(options.param_selection.template) > 1)
        or options.outcfg is not None
    ):
        if options.outcfg is not None:
            config_file = options.outcfg
        elif not modified and options.param_selection.param_file is None:
            name = "_and_".join(options.param_selection.template)
            config_file = options.cachedir / f"{name}.toml"
        else:
            config_file = options.cachedir / f"config-{uuid.uuid4().hex[:6]}.toml"

        write_template(inputs, config_file)
        cns.print(
            f":duck: [spring_green3]Wrote full configuration to [purple]{config_file}"
        )
        name = config_file.name
    else:
        name = " + ".join(cfg.name for cfg in options.param_selection.param_file)

    pretty_print_inputs(
        inputs=inputs,
        name=name,
        description="The minimal set of non-default parameters of your model",
        mode="minimal",
    )

    cns.print()
    cns.print(Rule("Starting Simulation", characters="="))
    return inputs


@run.command
def params(params: Parameters = Parameters()):
    """Show the current simulation parameters."""
    cns.print("Usage: 21cmfast run params --help")


@run.command()
def ics(
    options: RunParams = RunParams(),
    params: Annotated[Parameters, Parameter(show=False, name="*")] = Parameters(),
    min_evolved_redshift: Annotated[
        float, Parameter(name=("--zmin-evolution", "--zmin"))
    ] = 5.5,
):
    """Run a single iteration of 21cmFAST init, saving results to file.

    To specify simulation parameters, use a base template (either via --param-file or
    --template) and optionally override any particular simulation parameters. To see the
    available simulation parameters and how to specify them, use
    `21cmfast run params --help`.

    Parameters
    ----------
    min_evolved_redshift
        The minimum redshift down to which to evolve the simulation. For some simulation
        configurations, this is not used at all, while for others it will subtly change
        the evolution.
    """
    inputs = _run_setup(options, params, zmin=min_evolved_redshift)
    cache = OutputCache(options.cachedir)
    rc = RunCache.from_inputs(inputs=inputs, cache=cache)

    if rc.InitialConditions.exists():
        if options.regenerate:
            cns.print(
                "[yellow]:warning: Initial conditions already exist, but regeneration is requested. Overriding."
            )
        else:
            cns.print(
                f"[spring_green3]:duck: Initial conditions already exist at [purple]'{rc.InitialConditions}'[/purple], skipping computation."
            )
            return

    compute_initial_conditions(
        inputs=inputs,
        regenerate=options.regenerate,
        write=True,
        cache=OutputCache(options.cachedir),
    )

    cns.print(
        f"[spring_green3]:duck: Saved initial conditions to {rc.InitialConditions}"
    )


@run.command()
def coeval(
    redshifts: Annotated[list[float], Parameter(alias=("-z",), required=True)],
    options: RunParams = RunParams(),
    params: Annotated[Parameters, Parameter(show=False, name="*")] = Parameters(),
    out: Annotated[cyctp.ExistingDirectory, Parameter(name=("--out", "-o"))] = Path(),
    save_all_redshifts: Annotated[
        bool, Parameter(name=("--save-all-redshifts", "-a", "--all"))
    ] = False,
    min_evolved_redshift: Annotated[
        float, Parameter(name=("--zmin-evolution", "--zmin"))
    ] = 5.5,
):
    """Generate coeval cubes at given redshifts.

    To specify simulation parameters, use a base template (either via --param-file or
    --template) and optionally override any particular simulation parameters. To see the
    available simulation parameters and how to specify them, use
    `21cmfast run params --help`.

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
        cache=OutputCache(options.cachedir),
        write=getattr(
            CacheConfig, options.cache_strategy.replace("dmfield", "noloop")
        )(),
        progressbar=options.progress,
    ):
        if not in_outputs and not save_all_redshifts:
            continue

        outfile = out / f"coeval_z{coeval.redshift:.2f}.h5"
        coeval.save(outfile)
        cns.print(
            f"[spring_green3]:duck: Saved z={coeval.redshift:.2f} coeval box to [purple]{outfile}."
        )


@run.command()
def lightcone(
    options: RunParams = RunParams(),
    params: Annotated[Parameters, Parameter(show=False, name="*")] = Parameters(),
    redshift_range: tuple[float, float] = (6.0, 30.0),
    out: Annotated[
        Path,
        Parameter(validator=(vld.Path(dir_okay=False, file_okay=False, ext=("h5",)),)),
    ] = Path("lightcone.h5"),
    lightcone_quantities: Annotated[tuple[str], Parameter(alias=("--lq",))] = (
        "brightness_temp",
    ),
    min_evolved_redshift: Annotated[
        float, Parameter(name=("--zmin-evolution", "--zmin"))
    ] = 5.5,
):
    """Generate a lightcone between given redshifts.

    To specify simulation parameters, use a base template (either via --param-file or
    --template) and optionally override any particular simulation parameters. To see the
    available simulation parameters and how to specify them, use
    `21cmfast run params --help`.

    Parameters
    ----------
    redshift_range
        The redshifts between which to generate the lightcone.
    out
        The filename to which to save the lightcone data.
    lightcone_quantities
        Computed fields to generate lightcones for.
    min_evolved_redshift
        The minimum redshift down to which to evolve the simulation. For some simulation
        configurations, this is not used at all, while for others it will subtly change
        the evolution.
    """
    if not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)

    inputs = _run_setup(options, params, min_evolved_redshift, force_nodez=True)

    # For now, always use the old default lightconing algorithm
    lcn = RectilinearLightconer.between_redshifts(
        min_redshift=redshift_range[0],
        max_redshift=redshift_range[1],
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
        quantities=lightcone_quantities,
    )

    lc = run_lightcone(
        lightconer=lcn,
        inputs=inputs,
        regenerate=options.regenerate,
        write=getattr(
            CacheConfig, options.cache_strategy.replace("dmfield", "noloop")
        )(),
        cache=OutputCache(options.cachedir),
        progressbar=options.progress,
    )

    lc.save(out)

    cns.print(f"[spring_green3]:duck: Saved Lightcone to {out}.")


@dev.command(name="feature")
def pr_feature(
    params: Parameters,
    options: RunParams,
    redshift_range: tuple[float, float] = (6.0, 30.0),
    outdir: cyctp.ExistingDirectory = Path(),
):
    """
    Create standard plots comparing a default simulation against a simulation with a new feature.

    The base of the comparison is set by either --param-file or --template.
    The "new feature" which is to be compared is set by overriding specific parameters
    at the command line (e.g --use-ts-fluct).

    Plots are saved in the current directory, with the prefix "pr_feature".

    Parameters
    ----------
    redshift_range
        The redshifts between which to compute the lightcones for comparison.
    """
    import powerbox

    inputs_default, _ = _get_inputs(options, Parameters())
    inputs_new, _ = _get_inputs(options, params)

    inputs_default = inputs_default.with_logspaced_redshifts(
        zmin=redshift_range[0] - 0.1
    )
    inputs_new = inputs_new.with_logspaced_redshifts(zmin=redshift_range[0] - 0.1)

    # For now, always use the old default lightconing algorithm
    lcn = RectilinearLightconer.between_redshifts(
        min_redshift=redshift_range[0],
        max_redshift=redshift_range[1],
        resolution=inputs_default.simulation_options.cell_size,
        cosmo=inputs_default.cosmo_params.cosmo,
        quantities=("brightness_temp",),
    )

    cns.print("Running default lightcone...")

    lc_default = run_lightcone(
        lightconer=lcn,
        inputs=inputs_default,
        regenerate=options.regenerate,
        write=getattr(
            CacheConfig, options.cache_strategy.replace("dmfield", "noloop")
        )(),
        cache=OutputCache(options.cachedir),
        progressbar=True,
    )

    cns.print("Running lightcone with new feature...")
    lc_new = run_lightcone(
        lightconer=lcn,
        inputs=inputs_new,
        regenerate=options.regenerate,
        write=getattr(
            CacheConfig, options.cache_strategy.replace("dmfield", "noloop")
        )(),
        cache=OutputCache(options.cachedir),
        progressbar=True,
    )

    cns.print("Plotting lightcone slices...")
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)

    vmin = -150
    vmax = 30

    plotting.lightcone_sliceplot(lc_default, ax=ax[0], fig=fig, vmin=vmin, vmax=vmax)
    ax[0].set_title("Default")

    plotting.lightcone_sliceplot(
        lc_new, ax=ax[1], fig=fig, cbar=False, vmin=vmin, vmax=vmax
    )
    ax[1].set_title("New")

    plotting.lightcone_sliceplot(
        lc_default, lightcone2=lc_new, cmap="bwr", ax=ax[2], fig=fig
    )
    ax[2].set_title("Difference")

    plt.savefig(f"{outdir}/pr_feature_lightcone_2d_brightness_temp.pdf")

    def rms(x, axis=None):
        return np.sqrt(np.mean(x**2, axis=axis))

    cns.print("Plotting lightcone history...")
    fig, ax = plt.subplots(4, 1, sharex=True, gridspec_kw={"hspace": 0.05})
    ax[0].plot(
        lc_default.inputs.node_redshifts,
        lc_default.global_quantities["neutral_fraction"],
        label="Default",
    )
    ax[0].plot(
        lc_new.inputs.node_redshifts,
        lc_new.global_quantities["neutral_fraction"],
        label="New",
    )
    ax[0].set_ylabel(r"$x_{\rm HI}$")
    ax[0].legend()

    ax[1].plot(
        lc_default.inputs.node_redshifts,
        lc_default.global_quantities["brightness_temp"],
        label="Default",
    )
    ax[1].plot(
        lc_new.inputs.node_redshifts,
        lc_new.global_quantities["brightness_temp"],
        label="New",
    )
    ax[1].set_ylabel("$T_b$ [K]")
    ax[3].set_xlabel("z")

    rms_diff = rms(lc_default.lightcones["brightness_temp"], axis=(0, 1)) - rms(
        lc_new.lightcones["brightness_temp"], axis=(0, 1)
    )
    ax[2].plot(lc_default.lightcone_redshifts, rms_diff, label="RMS")
    ax[2].plot(
        lc_new.inputs.node_redshifts,
        lc_default.global_quantities["neutral_fraction"]
        - lc_new.global_quantities["neutral_fraction"],
        label="$x_{HI}$",
    )
    ax[2].plot(
        lc_new.inputs.node_redshifts,
        lc_default.global_quantities["brightness_temp"]
        - lc_new.global_quantities["brightness_temp"],
        label="$T_b$",
    )
    ax[2].legend()
    ax[2].set_ylabel("Differences")

    diff_rms = rms(
        lc_default.lightcones["brightness_temp"] - lc_new.lightcones["brightness_temp"],
        axis=(0, 1),
    )
    ax[3].plot(lc_default.lightcone_redshifts, diff_rms)
    ax[3].set_ylabel("RMS of Diff.")

    plt.savefig(f"{outdir}/pr_feature_history.pdf")

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
            lc_default.lightcones["brightness_temp"][:, :, start : start + ncells],
            (*lc_default.lightcone_dimensions[:2], chunk_size),
            bins_upto_boxlen=True,
        )
        p_default.append(pd)

        pn, k = powerbox.get_power(
            lc_new.lightcones["brightness_temp"][:, :, start : start + ncells],
            (*lc_new.lightcone_dimensions[:2], chunk_size),
            bins_upto_boxlen=True,
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

    plt.savefig(f"{outdir}/pr_feature_power_history.pdf")


@pred.command(name="struct-size")
def predict_struct_size(
    param_selection: ParameterSelection = ParameterSelection(),
    user_params: Parameters = Parameters(),
    unit: Literal["b", "kb", "mb", "gb"] | None = None,
    cache_config: Literal["on", "off", "noloop", "last_step_only"] = "on",
):
    """Compute the required storage per output kind for given inputs."""
    from .management import get_expected_sizes

    inputs, _ = _get_inputs(param_selection, user_params)
    sizes = get_expected_sizes(
        inputs, cache_config=getattr(CacheConfig, cache_config)()
    )

    units = ["b", "kb", "mb", "gb", "tb"]
    if unit is None:
        bigness = int((np.log(list(sizes.values())) / np.log(1024)).max())
    else:
        bigness = units.index(unit)

    table = Table(title="Output Struct Sizes")
    table.add_column("Struct Name")
    table.add_column("Size")
    table.add_column("Unit")

    for name, size in sizes.items():
        table.add_row(name, f"{size / 1024**bigness:.2f}", units[bigness].upper())
    table.add_section()
    table.add_row(
        "Total", f"{sum(sizes.values()) / 1024**bigness:.2f}", units[bigness].upper()
    )
    cns.print(table)


@pred.command(name="storage-size")
def predict_storage_size(
    param_selection: ParameterSelection = ParameterSelection(),
    user_params: Parameters = Parameters(),
    min_evolved_redshift: Annotated[
        float, Parameter(name=("--zmin-evolution", "--zmin"))
    ] = 5.5,
    unit: Literal["b", "kb", "mb", "gb"] | None = None,
    cache_config: Literal["on", "off", "noloop", "last_step_only"] = "on",
):
    """Compute the required storage for an entire run."""
    from .management import get_total_storage_size

    inputs, _ = _get_inputs(param_selection, user_params)
    inputs = inputs.with_logspaced_redshifts(zmin=min_evolved_redshift)

    sizes = get_total_storage_size(
        inputs, cache_config=getattr(CacheConfig, cache_config)()
    )

    units = ["b", "kb", "mb", "gb", "tb"]
    if unit is None:
        bigness = int(np.log(max(size for _, size in sizes.values())) / np.log(1024))
    else:
        bigness = units.index(unit)

    table = Table(title="Storage Sizes")
    table.add_column("Struct Name")
    table.add_column(f"Size [{units[bigness].upper()}]")
    table.add_column("Quantity")

    total_size = 0
    total_quant = 0
    for name, (quant, size) in sizes.items():
        total_size += size
        total_quant += quant
        table.add_row(name, f"{size / 1024**bigness:.2f}", f"{quant}")
    table.add_section()
    table.add_row("Total", f"{total_size / 1024**bigness:.2f}", f"{total_quant}")
    cns.print(table)


if __name__ == "__main__":
    app()
