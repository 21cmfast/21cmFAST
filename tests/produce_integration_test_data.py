"""
Produce integration test data.

THis data is tested by the `test_integration_features.py`
tests. One thing to note here is that all redshifts are reasonably high.

This is necessary, because low redshifts mean that neutral fractions are small,
and then numerical noise gets relatively more important, and can make the comparison
fail at the tens-of-percent level.
"""

import glob
import logging
import os
import sys
import tempfile
from pathlib import Path

import attrs
import click
import h5py
import numpy as np
import questionary as qs
from powerbox import get_power

from py21cmfast import (
    CacheConfig,
    InputParameters,
    SimulationOptions,
    compute_initial_conditions,
    config,
    determine_halo_list,
    get_logspaced_redshifts,
    perturb_field,
    perturb_halo_list,
    run_coeval,
    run_lightcone,
)
from py21cmfast.lightcones import RectilinearLightconer

logger = logging.getLogger("py21cmfast")
logging.basicConfig()

SEED = 12345
DATA_PATH = Path(__file__).parent / "test_data"

# These defaults are overwritten by the OPTIONS kwargs
DEFAULT_SIMULATION_OPTIONS = {
    "HII_DIM": 50,
    "DIM": 150,
    "BOX_LEN": 100,
    "SAMPLER_MIN_MASS": 1e9,
    "ZPRIME_STEP_FACTOR": 1.04,
    "M_MIN_in_mass": False,
}

DEFAULT_MATTER_OPTIONS = {
    "USE_HALO_FIELD": False,
    "NO_RNG": True,
    "HALO_STOCHASTICITY": False,
}

DEFAULT_ASTRO_OPTIONS = {
    "USE_EXP_FILTER": False,
    "CELL_RECOMB": False,
    "USE_TS_FLUCT": False,
    "INHOMO_RECO": False,
    "USE_UPPER_STELLAR_TURNOVER": False,
    "USE_MASS_DEPENDENT_ZETA": False,
}

DEFAULT_ASTRO_PARAMS = {
    "L_X": 40.0,
    "L_X_MINI": 40.0,
    "F_STAR7_MINI": -2.0,
}

LIGHTCONE_FIELDS = [
    "density",
    "velocity_z",
    "Ts_box",
    "ionisation_rate_G12",
    "cumulative_recombinations",
    "xray_ionised_fraction",
    "kinetic_temp_neutral",
    "J_21_LW",
    "neutral_fraction",
    "z_reion",
    "brightness_temp",
]

COEVAL_FIELDS = LIGHTCONE_FIELDS.copy()
COEVAL_FIELDS.insert(COEVAL_FIELDS.index("Ts_box"), "lowres_density")
COEVAL_FIELDS.insert(COEVAL_FIELDS.index("Ts_box"), "lowres_vx_2LPT")
COEVAL_FIELDS.insert(COEVAL_FIELDS.index("Ts_box"), "lowres_vx")

OPTIONS = {
    "simple": [12, {}],
    "perturb_high_res": [12, {"PERTURB_ON_HIGH_RES": True}],
    "change_step_factor": [11, {"ZPRIME_STEP_FACTOR": 1.02}],
    "change_z_heat_max": [30, {"Z_HEAT_MAX": 40}],
    "larger_step_factor": [
        13,
        {"ZPRIME_STEP_FACTOR": 1.05, "Z_HEAT_MAX": 25, "HMF": "PS"},
    ],
    "interp_perturb_field": [16, {"interp_perturb_field": True}],
    "mdzeta": [14, {"USE_MASS_DEPENDENT_ZETA": True, "M_MIN_in_mass": True}],
    "rsd": [9, {"SUBCELL_RSD": True}],
    "inhomo": [10, {"INHOMO_RECO": True, "R_BUBBLE_MAX": 50.0}],
    "tsfluct": [16, {"HMF": "WATSON-Z", "USE_TS_FLUCT": True}],
    "mmin_in_mass": [20, {"Z_HEAT_MAX": 45, "M_MIN_in_Mass": True, "HMF": "WATSON"}],
    "fftw_wisdom": [35, {"USE_FFTW_WISDOM": True}],
    "mini_halos": [
        18,
        {
            "Z_HEAT_MAX": 25,
            "USE_MINI_HALOS": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_mass": True,
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
            "ZPRIME_STEP_FACTOR": 1.1,
            "N_THREADS": 4,
            "USE_FFTW_WISDOM": True,
            "NUM_FILTER_STEPS_FOR_Ts": 8,
            "M_TURN": 5.0,
            "R_BUBBLE_MAX": 50.0,
        },
    ],
    "nthreads": [8, {"N_THREADS": 2}],
    "photoncons": [10, {"PHOTON_CONS_TYPE": "z-photoncons"}],
    "mdz_and_photoncons": [
        8.5,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_mass": True,
            "PHOTON_CONS_TYPE": "z-photoncons",
            "Z_HEAT_MAX": 25,
            "ZPRIME_STEP_FACTOR": 1.1,
        },
    ],
    "mdz_and_ts_fluct": [
        9,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_mass": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "PHOTON_CONS_TYPE": "z-photoncons",
            "Z_HEAT_MAX": 25,
            "ZPRIME_STEP_FACTOR": 1.1,
            "R_BUBBLE_MAX": 50.0,
        },
    ],
    "minimize_mem": [
        9,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_mass": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "PHOTON_CONS_TYPE": "z-photoncons",
            "Z_HEAT_MAX": 25,
            "ZPRIME_STEP_FACTOR": 1.1,
            "MINIMIZE_MEMORY": True,
            "R_BUBBLE_MAX": 50.0,
        },
    ],
    "mdz_and_tsfluct_nthreads": [
        8.5,
        {
            "N_THREADS": 2,
            "USE_FFTW_WISDOM": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_mass": True,
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
            "PHOTON_CONS_TYPE": "z-photoncons",
            "Z_HEAT_MAX": 25,
            "ZPRIME_STEP_FACTOR": 1.1,
            "R_BUBBLE_MAX": 50.0,
        },
    ],
    "halo_field": [9, {"USE_HALO_FIELD": True}],
    "halo_field_mdz": [
        8.5,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_mass": True,
            "USE_HALO_FIELD": True,
            "USE_TS_FLUCT": True,
            "Z_HEAT_MAX": 25,
            "ZPRIME_STEP_FACTOR": 1.1,
        },
    ],
    "halo_field_mdz_highres": [
        8.5,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_mass": True,
            "USE_HALO_FIELD": True,
            "PERTURB_ON_HIGH_RES": True,
            "N_THREADS": 4,
            "Z_HEAT_MAX": 25,
            "ZPRIME_STEP_FACTOR": 1.1,
        },
    ],
    "mdz_tsfluct_nthreads": [
        12.0,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_mass": True,
            "USE_TS_FLUCT": True,
            "PERTURB_ON_HIGH_RES": False,
            "N_THREADS": 4,
            "Z_HEAT_MAX": 25,
            "ZPRIME_STEP_FACTOR": 1.2,
            "NUM_FILTER_STEPS_FOR_Ts": 4,
            "USE_INTERPOLATION_TABLES": "no-interpolation",
        },
    ],
    "ts_fluct_no_tables": [
        12.0,
        {
            "USE_TS_FLUCT": True,
            "N_THREADS": 4,
            "Z_HEAT_MAX": 25,
            "ZPRIME_STEP_FACTOR": 1.2,
            "NUM_FILTER_STEPS_FOR_Ts": 4,
            "USE_INTERPOLATION_TABLES": "no-interpolation",
        },
    ],
    "minihalos_no_tables": [
        12.0,
        {
            "USE_MINI_HALOS": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_mass": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "N_THREADS": 4,
            "Z_HEAT_MAX": 25,
            "ZPRIME_STEP_FACTOR": 1.1,
            "NUM_FILTER_STEPS_FOR_Ts": 4,
            "USE_INTERPOLATION_TABLES": "no-interpolation",
            "M_TURN": 5.0,
        },
    ],
    "fast_fcoll_hiz": [
        18,
        {
            "N_THREADS": 4,
            "INTEGRATION_METHOD_MINI": "GAMMA-APPROX",
            "USE_INTERPOLATION_TABLES": "hmf-interpolation",
        },
    ],
    "fast_fcoll_lowz": [
        8,
        {
            "N_THREADS": 4,
            "INTEGRATION_METHOD_MINI": "GAMMA-APPROX",
            "USE_INTERPOLATION_TABLES": "hmf-interpolation",
        },
    ],
    "relvel": [
        18,
        {
            "Z_HEAT_MAX": 25,
            "USE_MINI_HALOS": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_mass": True,
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
            "ZPRIME_STEP_FACTOR": 1.1,
            "N_THREADS": 4,
            "NUM_FILTER_STEPS_FOR_Ts": 8,
            "USE_INTERPOLATION_TABLES": "hmf-interpolation",
            "INTEGRATION_METHOD_MINI": "GAMMA-APPROX",
            "USE_RELATIVE_VELOCITIES": True,
            "M_TURN": 5.0,
        },
    ],
    "lyman_alpha_heating": [
        8,
        {"N_THREADS": 4, "USE_CMB_HEATING": False},
    ],
    "cmb_heating": [
        8,
        {"N_THREADS": 4, "USE_LYA_HEATING": False},
    ],
    "halo_sampling": [
        12,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_mass": True,
        },
    ],
}

if len(set(OPTIONS.keys())) != len(list(OPTIONS.keys())):
    raise ValueError("There is a non-unique option name!")

OPTIONS_PT = {
    "simple": [10, {}],
    "no2lpt": [10, {"PERTURB_ALGORITHM": "ZELDOVICH"}],
    "linear": [10, {"EVOLVE_DENSITY_LINEARLY": 1}],
    "highres": [10, {"PERTURB_ON_HIGH_RES": True}],
}

if len(set(OPTIONS_PT.keys())) != len(list(OPTIONS_PT.keys())):
    raise ValueError("There is a non-unique option_pt name!")

OPTIONS_HALO = {
    "halo_field": [9, {"USE_HALO_FIELD": True, "USE_MASS_DEPENDENT_ZETA": True}]
}

if len(set(OPTIONS_HALO.keys())) != len(list(OPTIONS_HALO.keys())):
    raise ValueError("There is a non-unique option_halo name!")


def get_node_z(redshift, lc=False, **kwargs):
    """Get the node redshifts we want to use for test runs.

    Values for the spacing and maximum go kwargs --> test defaults --> struct defaults
    """
    node_redshifts = None
    if lc or kwargs.get("USE_TS_FLUCT", False) or kwargs.get("INHOMO_RECO", False):
        node_redshifts = get_logspaced_redshifts(
            min_redshift=redshift,
            max_redshift=kwargs.get(
                "Z_HEAT_MAX",
                DEFAULT_SIMULATION_OPTIONS.get(
                    "Z_HEAT_MAX", SimulationOptions.new().Z_HEAT_MAX
                ),
            ),
            z_step_factor=kwargs.get(
                "ZPRIME_STEP_FACTOR",
                DEFAULT_SIMULATION_OPTIONS.get(
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
            **DEFAULT_MATTER_OPTIONS,
            **DEFAULT_SIMULATION_OPTIONS,
            **DEFAULT_ASTRO_OPTIONS,
            **DEFAULT_ASTRO_PARAMS,
            **kwargs,
        }
    )

    options = {"inputs": inputs}
    if not lc:
        options["out_redshifts"] = redshift
    return options


def produce_coeval_power_spectra(redshift, **kwargs):
    options = get_all_options_struct(redshift, lc=False, **kwargs)
    print("----- OPTIONS USED -----")
    print(options)
    print("------------------------")

    [coeval] = run_coeval(write=False, **options)
    p = {}

    for field in COEVAL_FIELDS:
        if hasattr(coeval, field):
            p[field], k = get_power(
                getattr(coeval, field), boxlength=coeval.simulation_options.BOX_LEN
            )

    return k, p, coeval


def produce_lc_power_spectra(redshift, **kwargs):
    options = get_all_options_struct(redshift, lc=False, **kwargs)
    print("----- OPTIONS USED -----")
    print(options)
    print("------------------------")

    # NOTE: this is here only so that we get the same answer as previous versions,
    #       which have a bug where the max_redshift gets set higher than it needs to be.
    astro_options = options["inputs"].astro_options
    if astro_options.INHOMO_RECO or astro_options.USE_TS_FLUCT:
        max_redshift = options["inputs"].simulation_options.Z_HEAT_MAX
    else:
        max_redshift = options["out_redshifts"] + 2

    # convert options to lightcone version
    options["inputs"] = options["inputs"].clone(
        node_redshifts=get_logspaced_redshifts(
            min_redshift=options.pop("out_redshifts"),
            max_redshift=max_redshift,
            z_step_factor=options["inputs"].simulation_options.ZPRIME_STEP_FACTOR,
        )
    )

    quantities = LIGHTCONE_FIELDS[:]
    if not astro_options.USE_TS_FLUCT:
        [
            quantities.remove(k)
            for k in {"Ts_box", "xray_ionised_fraction", "kinetic_temp_neutral"}
        ]
    if not astro_options.USE_MINI_HALOS:
        quantities.remove("J_21_LW")

    lcn = RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=redshift,
        max_redshift=max_redshift,
        quantities=quantities,
        resolution=options["inputs"].simulation_options.cell_size,
    )

    _, _, _, lightcone = run_lightcone(
        lightconer=lcn,
        write=False,
        **options,
    )

    p = {}
    for field in LIGHTCONE_FIELDS:
        if field in lightcone.lightcones:
            p[field], k = get_power(
                lightcone.lightcones[field],
                boxlength=lightcone.lightcone_dimensions,
            )

    return k, p, lightcone


def produce_perturb_field_data(redshift, **kwargs):
    options = get_all_options_struct(redshift, **kwargs)
    del options["out_redshifts"]

    velocity_normalisation = 1e16

    with config.use(regenerate=True, write=False):
        init_box = compute_initial_conditions(**options)
        pt_box = perturb_field(redshift=redshift, initial_conditions=init_box)

    p_dens, k_dens = get_power(
        pt_box.get("density"),
        boxlength=options["inputs"].simulation_options.BOX_LEN,
    )
    p_vel, k_vel = get_power(
        pt_box.get("velocity_z") * velocity_normalisation,
        boxlength=options["inputs"].simulation_options.BOX_LEN,
    )

    def hist(kind, xmin, xmax, nbins):
        data = pt_box.get(kind)
        if kind == "velocity":
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


def produce_halo_field_data(redshift, **kwargs):
    options = get_all_options_struct(redshift, **kwargs)

    with config.use(regenerate=True, write=False):
        init_box = compute_initial_conditions(**options)
        halos = determine_halo_list(initial_conditions=init_box, **options)
        pt_halos = perturb_halo_list(
            initial_conditions=init_box, halo_field=halos, **options
        )

    return pt_halos


def get_filename(kind, name, **kwargs):
    # get sorted keys
    fname = f"{kind}_{name}.h5"
    return DATA_PATH / fname


def get_old_filename(redshift, kind, **kwargs):
    # get sorted keys
    kwargs = {k: kwargs[k] for k in sorted(kwargs)}
    string = "_".join(f"{k}={v}" for k, v in kwargs.items())
    fname = f"{kind}_z{redshift:.2f}_{string}.h5"

    return DATA_PATH / fname


def produce_power_spectra_for_tests(name, redshift, force, direc, **kwargs):
    fname = get_filename("power_spectra", name)

    # Need to manually remove it, otherwise h5py tries to add to it
    if fname.exists():
        if force:
            fname.unlink()
        else:
            print(f"Skipping {fname} because it already exists.")
            return fname

    # For tests, we *don't* want to use cached boxes, but we also want to use the
    # cache between the power spectra and lightcone. So we create a temporary
    # directory in which to cache results.
    with config.use(direc=direc):
        k, p, coeval = produce_coeval_power_spectra(redshift, **kwargs)
        k_l, p_l, lc = produce_lc_power_spectra(redshift, **kwargs)

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

        lc_grp["global_xH"] = lc.global_xH
        lc_grp["global_brightness_temp"] = lc.global_brightness_temp

    print(f"Produced {fname} with {kwargs}")
    return fname


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

    print(f"Produced {fname} with {kwargs}")
    return fname


def produce_data_for_halo_field_tests(name, redshift, force, **kwargs):
    pt_halos = produce_halo_field_data(redshift, **kwargs)

    fname = get_filename("halo_field_data", name)

    # Need to manually remove it, otherwise h5py tries to add to it
    if fname.exists():
        if force:
            fname.unlink()
        else:
            return fname

    with h5py.File(fname, "w") as fl:
        for k, v in kwargs.items():
            fl.attrs[k] = v

        fl["n_pt_halos"] = pt_halos.n_halos
        fl["pt_halo_masses"] = pt_halos.halo_masses

    print(f"Produced {fname} with {kwargs}")
    return fname


main = click.Group()


@main.command()
@click.option("--log-level", default="WARNING")
@click.option("--force/--no-force", default=False)
@click.option("--remove/--no-remove", default=True)
@click.option("--pt-only/--not-pt-only", default=False)
@click.option("--no-pt/--pt", default=False)
@click.option("--no-halo/--do-halo", default=False)
@click.option(
    "--names",
    multiple=True,
    type=click.Choice(list(OPTIONS.keys())),
    default=list(OPTIONS.keys()),
)
def go(
    log_level: str,
    force: bool,
    remove: bool,
    pt_only: bool,
    no_pt: bool,
    no_halo,
    names,
):
    logger.setLevel(log_level.upper())

    if names != list(OPTIONS.keys()):
        remove = False
        force = True

    if pt_only or no_pt or no_halo:
        remove = False

    # For tests, we *don't* want to use cached boxes, but we also want to use the
    # cache between the power spectra and lightcone. So we create a temporary
    # directory in which to cache results.
    direc = tempfile.mkdtemp()
    fnames = []

    if not pt_only:
        for name in names:
            redshift = OPTIONS[name][0]
            kwargs = OPTIONS[name][1]

            fnames.append(
                produce_power_spectra_for_tests(name, redshift, force, direc, **kwargs)
            )

    if not no_pt:
        for name, (redshift, kwargs) in OPTIONS_PT.items():
            fnames.append(
                produce_data_for_perturb_field_tests(name, redshift, force, **kwargs)
            )

    if not no_halo:
        for name, (redshift, kwargs) in OPTIONS_HALO.items():
            fnames.append(
                produce_data_for_halo_field_tests(name, redshift, force, **kwargs)
            )

    # Remove extra files that
    if not (names or pt_only or no_pt or no_halo):
        all_files = DATA_PATH.glob("*")
        for fl in all_files:
            if fl not in fnames:
                if remove:
                    print(f"Removing old file: {fl}")
                    fl.unlink()
                else:
                    print(f"File is now redundant and can be removed: {fl}")


@main.command()
def convert():
    """Convert old-style data file names to new ones."""
    all_files = DATA_PATH.glob("*")

    old_names = {
        get_old_filename(v[0], "power_spectra", **v[1]): k for k, v in OPTIONS.items()
    }
    old_names_pt = {
        get_old_filename(v[0], "perturb_field_data", **v[1]): k
        for k, v in OPTIONS_PT.items()
    }
    old_names_hf = {
        get_old_filename(v[0], "halo_field_data", **v[1]): k
        for k, v in OPTIONS_HALO.items()
    }

    for fl in all_files:
        if fl.name.startswith("power_spectra"):
            if fl.stem.split("power_spectra_")[-1] in OPTIONS:
                continue
            elif fl in old_names:
                new_file = get_filename("power_spectra", old_names[fl])
                fl.rename(new_file)
                continue
        elif fl.name.startswith("perturb_field_data"):
            if fl.stem.split("perturb_field_data_")[-1] in OPTIONS_PT:
                continue
            elif fl in old_names_pt:
                new_file = get_filename("perturb_field_data", old_names_pt[fl])
                fl.rename(new_file)
                continue
        elif fl.name.startswith("halo_field_data"):
            if fl.stem.split("halo_field_data_")[-1] in OPTIONS_HALO:
                continue
            elif fl in old_names_hf:
                new_file = get_filename("halo_field_data", old_names_hf[fl])
                fl.rename(new_file)
                continue

        if qs.confirm(f"Remove {fl}?").ask():
            fl.unlink()


@main.command()
def clean():
    """Convert old-style data file names to new ones."""
    all_files = DATA_PATH.glob("*")

    for fl in all_files:
        if (
            (
                fl.stem.startswith("power_spectra")
                and fl.stem.split("power_spectra_")[-1] in OPTIONS
            )
            or (
                fl.stem.startswith("perturb_field_data")
                and fl.stem.split("perturb_field_data_")[-1] in OPTIONS_PT
            )
            or (
                fl.stem.startswith("halo_field_data")
                and fl.stem.split("halo_field_data_")[-1] in OPTIONS_HALO
            )
        ):
            continue

        if qs.confirm(f"Remove {fl}?").ask():
            fl.unlink()


if __name__ == "__main__":
    main()
