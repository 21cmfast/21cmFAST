"""
Produce integration test data.

THis data is tested by the `test_integration_features.py`
tests. One thing to note here is that all redshifts are reasonably high.

This is necessary, because low redshifts mean that neutral fractions are small,
and then numerical noise gets relatively more important, and can make the comparison
fail at the tens-of-percent level.
"""

import logging
import tempfile
import warnings
from pathlib import Path
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tyro
from powerbox import get_power
from rich.console import Console
from rich.rule import Rule

from py21cmfast import (
    Coeval,
    InputParameters,
    LightCone,
    OutputCache,
    SimulationOptions,
    compute_initial_conditions,
    config,
    get_logspaced_redshifts,
    perturb_field,
    plotting,
    run_coeval,
    run_lightcone,
)
from py21cmfast.lightconers import RectilinearLightconer

cns = Console()
logger = logging.getLogger("py21cmfast")
logging.basicConfig()

SEED = 12345
DATA_PATH = Path(__file__).parent / "test_data"

DEFAULT_INPUTS_TESTRUNS = {
    # SimulationOptions
    "HII_DIM": 50,
    "DIM": 150,
    "BOX_LEN": 100,
    "SAMPLER_MIN_MASS": 1e9,
    "ZPRIME_STEP_FACTOR": 1.04,
    # MatterOptions
    "USE_HALO_FIELD": False,
    "HALO_STOCHASTICITY": False,
    # AstroOptions
    "USE_EXP_FILTER": False,
    "CELL_RECOMB": False,
    "USE_TS_FLUCT": False,
    "USE_UPPER_STELLAR_TURNOVER": False,
    "N_THREADS": 2,
}

LIGHTCONE_FIELDS = [
    "density",
    "velocity_z",
    "spin_temperature",
    "xray_ionised_fraction",
    "J_21_LW",
    "kinetic_temp_neutral",
    "ionisation_rate_G12",
    "cumulative_recombinations",
    "neutral_fraction",
    "z_reion",
    "brightness_temp",
]

COEVAL_FIELDS = LIGHTCONE_FIELDS.copy()
COEVAL_FIELDS.insert(COEVAL_FIELDS.index("spin_temperature"), "lowres_vx_2LPT")
COEVAL_FIELDS.insert(COEVAL_FIELDS.index("spin_temperature"), "lowres_vx")
COEVAL_FIELDS.insert(COEVAL_FIELDS.index("spin_temperature"), "lowres_density")

OPTIONS_TESTRUNS = {
    "simple": [18, {}],
    "no-mdz": [
        18,
        {
            "USE_MASS_DEPENDENT_ZETA": False,
        },
    ],
    "mini": [
        18,
        {
            "USE_MINI_HALOS": True,
            "INHOMO_RECO": True,
            "R_BUBBLE_MAX": 50.0,
            "USE_TS_FLUCT": True,
            "M_TURN": 5.0,
            "Z_HEAT_MAX": 25,
            "ZPRIME_STEP_FACTOR": 1.1,
            "N_THREADS": 4,
            "USE_RELATIVE_VELOCITIES": True,
            "POWER_SPECTRUM": "CLASS",
        },
    ],
    "mini_gamma_approx": [
        18,
        {
            "USE_MINI_HALOS": True,
            "INHOMO_RECO": True,
            "R_BUBBLE_MAX": 50.0,
            "USE_TS_FLUCT": True,
            "M_TURN": 5.0,
            "Z_HEAT_MAX": 25,
            "ZPRIME_STEP_FACTOR": 1.1,
            "N_THREADS": 4,
            "INTEGRATION_METHOD_MINI": "GAMMA-APPROX",
            "INTEGRATION_METHOD_ATOMIC": "GAMMA-APPROX",
            "POWER_SPECTRUM": "CLASS",
        },
    ],
    "ts": [
        18,
        {"USE_TS_FLUCT": True},
    ],
    "ts_nomdz": [
        18,
        {"USE_TS_FLUCT": True, "USE_MASS_DEPENDENT_ZETA": False},
    ],
    "inhomo": [
        18,
        {
            "INHOMO_RECO": True,
            "R_BUBBLE_MAX": 50.0,
        },
    ],
    "inhomo_ts": [
        18,
        {
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
            "R_BUBBLE_MAX": 50.0,
        },
    ],
    "sampler": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
        },
    ],
    "fixed_halogrids": [
        18,
        {
            "USE_HALO_FIELD": True,
            "FIXED_HALO_GRIDS": True,
        },
    ],
    "sampler_mini": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_MINI_HALOS": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "R_BUBBLE_MAX": 50.0,
            "M_TURN": 5.0,
        },
    ],
    "sampler_ts": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_TS_FLUCT": True,
        },
    ],
    "sampler_ir": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "INHOMO_RECO": True,
            "R_BUBBLE_MAX": 50.0,
        },
    ],
    "sampler_ts_ir": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "R_BUBBLE_MAX": 50.0,
        },
    ],
    "sampler_ts_ir_onethread": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "R_BUBBLE_MAX": 50.0,
            "N_THREADS": 1,
        },
    ],
    "sampler_noncubic": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "NON_CUBIC_FACTOR": 1.2,
        },
    ],
    "dexm": [
        18,
        {
            "USE_HALO_FIELD": True,
        },
    ],
    "photoncons-z": [
        18,
        {
            "PHOTON_CONS_TYPE": "z-photoncons",
        },
    ],
    "minimize_mem": [
        18,
        {
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "R_BUBBLE_MAX": 50.0,
            "MINIMIZE_MEMORY": True,
        },
    ],
    "rsd": [18, {"SUBCELL_RSD": True}],
    "fftw_wisdom": [18, {"USE_FFTW_WISDOM": True}],
}

if len(set(OPTIONS_TESTRUNS.keys())) != len(list(OPTIONS_TESTRUNS.keys())):
    raise ValueError("There is a non-unique option name!")

OPTIONS_PT = {
    "simple": [10, {}],
    "no2lpt": [10, {"PERTURB_ALGORITHM": "ZELDOVICH"}],
    "linear": [10, {"PERTURB_ALGORITHM": "LINEAR"}],
    "highres": [10, {"PERTURB_ON_HIGH_RES": True}],
}

if len(set(OPTIONS_PT.keys())) != len(list(OPTIONS_PT.keys())):
    raise ValueError("There is a non-unique option_pt name!")


def get_node_z(redshift, lc=False, **kwargs):
    """Get the node redshifts we want to use for test runs.

    Values for the spacing and maximum go kwargs --> test defaults --> struct defaults
    """
    node_redshifts = None
    max_redshift = redshift + 2
    if kwargs.get("USE_TS_FLUCT", False) or kwargs.get("INHOMO_RECO", False):
        max_redshift = kwargs.get(
            "Z_HEAT_MAX",
            DEFAULT_INPUTS_TESTRUNS.get(
                "Z_HEAT_MAX", SimulationOptions.new().Z_HEAT_MAX
            ),
        )

    if lc or kwargs.get("USE_TS_FLUCT", False) or kwargs.get("INHOMO_RECO", False):
        node_redshifts = get_logspaced_redshifts(
            min_redshift=redshift,
            max_redshift=max_redshift,
            z_step_factor=kwargs.get(
                "ZPRIME_STEP_FACTOR",
                DEFAULT_INPUTS_TESTRUNS.get(
                    "ZPRIME_STEP_FACTOR", SimulationOptions.new().ZPRIME_STEP_FACTOR
                ),
            ),
        )
    return node_redshifts


def get_all_options_struct(redshift, lc=False, **kwargs):
    node_redshifts = get_node_z(redshift, lc=lc, **kwargs)

    inputs = InputParameters(
        node_redshifts=node_redshifts,
        random_seed=SEED,
    ).evolve_input_structs(
        **{
            **DEFAULT_INPUTS_TESTRUNS,
            **kwargs,
        }
    )

    options = {"inputs": inputs}
    if not lc:
        options["out_redshifts"] = redshift
    return options


def produce_coeval_power_spectra(redshift: float, cache: OutputCache, **kwargs):
    options = get_all_options_struct(redshift, lc=False, **kwargs)
    cns.print("\tRunning Coeval")
    [coeval] = run_coeval(
        write=True,  # write so that perturbed fields and halos can be cached.
        regenerate=True,
        cache=cache,
        **options,
    )
    p = {}

    for field in COEVAL_FIELDS:
        if hasattr(coeval, field):
            p[field], k = get_power(
                getattr(coeval, field),
                boxlength=coeval.simulation_options.BOX_LEN,
                bins_upto_boxlen=True,
            )

    return k, p, coeval


def get_lc_fields(inputs):
    quantities = LIGHTCONE_FIELDS[:]
    if not inputs.astro_options.USE_TS_FLUCT:
        [
            quantities.remove(k)
            for k in {
                "spin_temperature",
                "xray_ionised_fraction",
                "kinetic_temp_neutral",
            }
        ]
    if not inputs.astro_options.USE_MINI_HALOS:
        quantities.remove("J_21_LW")
    if not inputs.astro_options.INHOMO_RECO:
        quantities.remove("cumulative_recombinations")

    return quantities


def produce_lc_power_spectra(redshift: float, cache: OutputCache, **kwargs):
    options = get_all_options_struct(redshift, lc=True, **kwargs)

    cns.print("\tRunning Lightcone")
    node_z = options["inputs"].node_redshifts

    quantities = get_lc_fields(options["inputs"])
    lcn = RectilinearLightconer.between_redshifts(
        min_redshift=node_z[-1] + 0.2,
        max_redshift=node_z[0] - 0.2,
        quantities=quantities,
        resolution=options["inputs"].simulation_options.cell_size,
    )

    _, _, _, lightcone = run_lightcone(
        lightconer=lcn,
        write=True,  # write so that perturbed fields and halos can be cached.
        regenerate=True,
        cache=cache,
        **options,
    )

    p = {}
    for field in LIGHTCONE_FIELDS:
        if field in lightcone.lightcones:
            p[field], k = get_power(
                lightcone.lightcones[field],
                boxlength=lightcone.lightcone_dimensions,
                bins_upto_boxlen=True,
            )

    return k, p, lightcone


def produce_perturb_field_data(redshift, **kwargs):
    options = get_all_options_struct(redshift, lc=False, **kwargs)
    del options["out_redshifts"]

    velocity_normalisation = 1e16

    init_box = compute_initial_conditions(**options)
    pt_box = perturb_field(redshift=redshift, initial_conditions=init_box)

    p_dens, k_dens = get_power(
        pt_box.get("density"),
        boxlength=options["inputs"].simulation_options.BOX_LEN,
        bins_upto_boxlen=True,
    )
    p_vel, k_vel = get_power(
        pt_box.get("velocity_z") * velocity_normalisation,
        boxlength=options["inputs"].simulation_options.BOX_LEN,
        bins_upto_boxlen=True,
    )

    def hist(kind, xmin, xmax, nbins):
        data = pt_box.get(kind)
        if kind == "velocity_z":
            data = velocity_normalisation * data

        bins, edges = np.histogram(
            data,
            bins=np.linspace(xmin, xmax, nbins),
            range=[xmin, xmax],
            density=True,
        )

        left, right = edges[:-1], edges[1:]

        X = np.array([left, right]).T.flatten()
        Y = np.array([bins, bins]).T.flatten()
        return X, Y

    X_dens, Y_dens = hist("density", -0.8, 2.0, 50)
    X_vel, Y_vel = hist("velocity_z", -2, 2, 50)

    return k_dens, p_dens, k_vel, p_vel, X_dens, Y_dens, X_vel, Y_vel, init_box


def get_filename(kind, name):
    # get sorted keys
    fname = f"{kind}_{name}.h5"
    return DATA_PATH / fname


def produce_power_spectra_for_tests(
    name, redshift, force, direc, **kwargs
) -> tuple[Path, Coeval | None, LightCone | None]:
    fname = get_filename("power_spectra", name)

    # Need to manually remove it, otherwise h5py tries to add to it
    if fname.exists():
        if force:
            fname.unlink()
        else:
            cns.print(
                f"\t[orange]:warning: Skipping {fname} because it already exists."
            )
            return fname, None, None

    # For tests, we *don't* want to use cached boxes, but we also want to use the
    # cache between the power spectra and lightcone. So we create a temporary
    # directory in which to cache results.
    cns.print(f"\tOptions: {kwargs}")
    k, p, coeval = produce_coeval_power_spectra(
        redshift, cache=OutputCache(direc), **kwargs
    )
    k_l, p_l, lc = produce_lc_power_spectra(
        redshift, cache=OutputCache(direc), **kwargs
    )

    with h5py.File(fname, "w") as fl:
        for key, v in kwargs.items():
            fl.attrs[key] = v

        fl.attrs["HII_DIM"] = coeval.simulation_options.HII_DIM
        fl.attrs["DIM"] = coeval.simulation_options.DIM
        fl.attrs["BOX_LEN"] = coeval.simulation_options.BOX_LEN

        coeval_grp = fl.create_group("coeval")
        coeval_grp["k"] = k
        for key, val in p.items():
            coeval_grp[f"power_{key}"] = val

        lc_grp = fl.create_group("lightcone")
        lc_grp["k"] = k_l
        for key, val in p_l.items():
            lc_grp[f"power_{key}"] = val

        lc_grp["global_neutral_fraction"] = lc.global_quantities["neutral_fraction"]
        lc_grp["global_brightness_temp"] = lc.global_quantities["brightness_temp"]

    cns.print(f"\tProduced {fname}")
    return fname, coeval, lc


def produce_data_for_perturb_field_tests(name, redshift, force, **kwargs):
    (
        k_dens,
        p_dens,
        k_vel,
        p_vel,
        X_dens,
        Y_dens,
        X_vel,
        Y_vel,
        init_box,
    ) = produce_perturb_field_data(redshift, **kwargs)

    fname = get_filename("perturb_field_data", name)

    # Need to manually remove it, otherwise h5py tries to add to it
    if fname.exists():
        if force:
            fname.unlink()
        else:
            return fname

    with h5py.File(fname, "w") as fl:
        for k, v in kwargs.items():
            fl.attrs[k] = v

        fl.attrs["HII_DIM"] = init_box.simulation_options.HII_DIM
        fl.attrs["DIM"] = init_box.simulation_options.DIM
        fl.attrs["BOX_LEN"] = init_box.simulation_options.BOX_LEN

        fl["power_dens"] = p_dens
        fl["k_dens"] = k_dens

        fl["power_vel"] = p_vel
        fl["k_vel"] = k_vel

        fl["pdf_dens"] = Y_dens
        fl["x_dens"] = X_dens

        fl["pdf_vel"] = Y_vel
        fl["x_vel"] = X_vel

    cns.print(f"\tProduced {fname}")
    return fname


def print_failure_stats(test, truth, inputs, abs_tol, rel_tol, name):
    sel_failed = np.fabs(truth - test) > (abs_tol + np.fabs(truth) * rel_tol)

    if not np.any(sel_failed):
        return False

    failed_idx = np.where(sel_failed)
    warnings.warn(
        f"{name}: atol {abs_tol} rtol {rel_tol} failed {sel_failed.sum()} of {sel_failed.size} {sel_failed.sum() / sel_failed.size * 100:.4f}%",
        stacklevel=2,
    )
    warnings.warn(
        f"subcube of failures [min] [max] {[f.min() for f in failed_idx]} {[f.max() for f in failed_idx]}",
        stacklevel=2,
    )
    warnings.warn(
        f"failure range truth ({truth[sel_failed].min():.3e},{truth[sel_failed].max():.3e}) test ({test[sel_failed].min():.3e},{test[sel_failed].max():.3e})",
        stacklevel=2,
    )
    warnings.warn(
        f"max abs diff of failures {np.fabs(truth - test)[sel_failed].max():.4e} relative {(np.fabs(truth - test) / truth)[sel_failed].max():.4e}",
        stacklevel=2,
    )

    failed_inp = [
        inp[sel_failed if inp.shape == test.shape else failed_idx[i]]
        for i, inp in enumerate(inputs)
    ]
    for i, _inp in enumerate(inputs):
        warnings.warn(
            f"failure range of inputs axis {i} {failed_inp[i].min():.2e} {failed_inp[i].max():.2e}",
            stacklevel=2,
        )

    warnings.warn("----- First 10 -----", stacklevel=2)
    for j in range(min(10, sel_failed.sum())):
        input_arr = [f"{failed_inp[i][j]:.2e}" for i, finp in enumerate(failed_inp)]
        warnings.warn(
            f"CRD {input_arr}"
            + f"  {truth[sel_failed].flatten()[j]:.4e} {test[sel_failed].flatten()[j]:.4e}",
            stacklevel=2,
        )

    return True


CASE_CHOICES = Literal[tuple(OPTIONS_TESTRUNS.keys())]


def go(
    log_level: Literal["WARNING", "DEBUG", "INFO", "ERROR", "CRITICAL"] = "WARNING",
    force: bool = False,
    remove: bool = True,
    pt_only: bool = False,
    no_pt: bool = False,
    names: tuple[CASE_CHOICES, ...] = tuple(OPTIONS_TESTRUNS.keys()),
):
    cns.print(Rule("[bold][purple]Reproducing Integration Test Data!"))

    logger.setLevel(log_level.upper())

    if names != list(OPTIONS_TESTRUNS.keys()):
        remove = False
        force = True

    if pt_only or no_pt:
        remove = False

    # For tests, we *don't* want to use cached boxes, but we also want to use the
    # cache between the power spectra and lightcone. So we create a temporary
    # directory in which to cache results.
    direc = tempfile.mkdtemp()
    fnames = []

    if not pt_only:
        # While we're making the lightcones / coevals, take the opportunity to make some
        # nice plots of the fields.
        bt_coeval_fig, bt_coeval_ax = plt.subplots(
            int(np.ceil(np.sqrt(len(names)))),
            int(np.ceil(np.sqrt(len(names)))),
            figsize=(12, 12),
            sharex=True,
            sharey=True,
            gridspec_kw={"hspace": 0.1, "wspace": 0.1},
            layout="constrained",
        )

        bt_lc_fig, bt_lc_ax = plt.subplots(
            len(names),
            1,
            figsize=(12, 3 * len(names)),
            sharex=True,
            sharey=True,
            gridspec_kw={"hspace": 0.1, "wspace": 0.1},
            layout="constrained",
        )

        for i, name in enumerate(names):
            cns.print(f"Running case '{name}'... [{i + 1}/{len(names)}]")
            redshift = OPTIONS_TESTRUNS[name][0]
            kwargs = OPTIONS_TESTRUNS[name][1]

            fname, coeval, lc = produce_power_spectra_for_tests(
                name, redshift, force, direc, **kwargs
            )
            fnames.append(fname)

            # Make a coeval plot for _this_ case with all the fields...
            if coeval is not None:
                fig, ax = plt.subplots(
                    1,
                    len(COEVAL_FIELDS),
                    figsize=(3 * len(COEVAL_FIELDS), 4),
                    gridspec_kw={"hspace": 0, "wspace": 0},
                    sharex=True,
                    sharey=True,
                    layout="constrained",
                )
                for j, field in enumerate(COEVAL_FIELDS):
                    if hasattr(coeval, field):
                        plotting.coeval_sliceplot(coeval, kind=field, fig=fig, ax=ax[j])
                        ax[j].set_title(field)
                    else:
                        ax[j].off()

                fig.savefig(f"integration-test-plots/coeval-sliceplots-{name}.pdf")
                plt.close(fig)

                # Now plot the brightness temperature coeval with other cases.
                plotting.coeval_sliceplot(
                    coeval,
                    kind="brightness_temp",
                    ax=bt_coeval_ax.flatten()[i],
                    fig=bt_coeval_fig,
                )
                bt_coeval_ax.flatten()[i].text(
                    0.1,
                    0.9,
                    name,
                    transform=bt_coeval_ax.flatten()[i].transAxes,
                    color="white",
                )

            # Make a lightcone plot for _this_ case with all the fields...
            if lc is not None:
                fig, ax = plt.subplots(
                    len(LIGHTCONE_FIELDS),
                    1,
                    figsize=(8, 3 * len(LIGHTCONE_FIELDS)),
                    sharex=True,
                    sharey=True,
                    gridspec_kw={"wspace": 0, "hspace": 0},
                    layout="constrained",
                )
                for j, field in enumerate(LIGHTCONE_FIELDS):
                    if field in lc.lightcones:
                        plotting.lightcone_sliceplot(lc, kind=field, fig=fig, ax=ax[j])
                        ax[j].set_title(field)
                    else:
                        ax[j].off()

                fig.savefig(f"integration-test-plots/lightcone-sliceplots-{name}.pdf")
                plt.close(fig)

                plotting.lightcone_sliceplot(
                    lc, kind="brightness_temp", ax=bt_lc_ax.flatten()[i], fig=bt_lc_fig
                )
                bt_lc_ax.flatten()[i].text(
                    0.1,
                    0.9,
                    name,
                    transform=bt_lc_ax.flatten()[i].transAxes,
                    color="white",
                )

        bt_coeval_fig.savefig(
            "integration-test-plots/coeval-brightness-temp-allcases.pdf"
        )
        bt_lc_fig.savefig("integration-test-plots/lc-brightness-temp-allcases.pdf")

    cns.print("[green]:tick: Finished producing Coeval and Lightcone power spectra.")

    if not no_pt:
        cns.print("Running perturb_field test data...")
        fnames.extend(
            produce_data_for_perturb_field_tests(name, redshift, force, **kwargs)
            for name, (redshift, kwargs) in OPTIONS_PT.items()
        )
    # Remove extra files
    if not ((names != list(OPTIONS_TESTRUNS.keys())) or pt_only or no_pt):
        all_files = DATA_PATH.glob("*")
        for fl in all_files:
            if fl not in fnames:
                if remove:
                    cns.print(f"[orange]Removing old file: {fl}")
                    fl.unlink()
                else:
                    cns.print(
                        f":warning: File is now redundant and can be removed: {fl}"
                    )


if __name__ == "__main__":
    tyro.cli(go)
