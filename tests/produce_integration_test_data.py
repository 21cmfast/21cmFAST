"""
Produce integration test data, which is tested by the `test_integration_features.py`
tests. One thing to note here is that all redshifts are reasonably high.

This is necessary, because low redshifts mean that neutral fractions are small,
and then numerical noise gets relatively more important, and can make the comparison
fail at the tens-of-percent level.
"""

import click
import glob
import h5py
import logging
import numpy as np
import os
import questionary as qs
import sys
import tempfile
from pathlib import Path
from powerbox import get_power

from py21cmfast import (
    AstroParams,
    CosmoParams,
    FlagOptions,
    InitialConditions,
    UserParams,
    config,
    determine_halo_list,
    global_params,
    initial_conditions,
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
DEFAULT_USER_PARAMS = {
    "HII_DIM": 50,
    "DIM": 150,
    "BOX_LEN": 100,
    "NO_RNG": True,
    "USE_INTERPOLATION_TABLES": True,
}
DEFAULT_ZPRIME_STEP_FACTOR = 1.04

LIGHTCONE_FIELDS = [
    "density",
    "velocity",
    "Ts_box",
    "Gamma12_box",
    "dNrec_box",
    "x_e_box",
    "Tk_box",
    "J_21_LW_box",
    "xH_box",
    "z_re_box",
    "brightness_temp",
]

COEVAL_FIELDS = LIGHTCONE_FIELDS.copy()
COEVAL_FIELDS.insert(COEVAL_FIELDS.index("Ts_box"), "lowres_density")
COEVAL_FIELDS.insert(COEVAL_FIELDS.index("Ts_box"), "lowres_vx_2LPT")
COEVAL_FIELDS.insert(COEVAL_FIELDS.index("Ts_box"), "lowres_vx")

OPTIONS = {
    "simple": [12, {}],
    "perturb_high_res": [12, {"PERTURB_ON_HIGH_RES": True}],
    "change_step_factor": [11, {"zprime_step_factor": 1.02}],
    "change_z_heat_max": [30, {"z_heat_max": 40}],
    "larger_step_factor": [
        13,
        {"zprime_step_factor": 1.05, "z_heat_max": 25, "HMF": 0},
    ],
    "interp_perturb_field": [16, {"interp_perturb_field": True}],
    "mdzeta": [14, {"USE_MASS_DEPENDENT_ZETA": True}],
    "rsd": [9, {"SUBCELL_RSD": True}],
    "inhomo": [10, {"INHOMO_RECO": True}],
    "tsfluct": [16, {"HMF": 3, "USE_TS_FLUCT": True}],
    "mmin_in_mass": [20, {"z_heat_max": 45, "M_MIN_in_Mass": True, "HMF": 2}],
    "fftw_wisdom": [35, {"USE_FFTW_WISDOM": True}],
    "mini_halos": [
        18,
        {
            "z_heat_max": 25,
            "USE_MINI_HALOS": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
            "zprime_step_factor": 1.1,
            "N_THREADS": 4,
            "USE_FFTW_WISDOM": True,
            "NUM_FILTER_STEPS_FOR_Ts": 8,
        },
    ],
    "nthreads": [8, {"N_THREADS": 2}],
    "photoncons": [10, {"PHOTON_CONS": True}],
    "mdz_and_photoncons": [
        8.5,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "PHOTON_CONS": True,
            "z_heat_max": 25,
            "zprime_step_factor": 1.1,
        },
    ],
    "mdz_and_ts_fluct": [
        9,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "PHOTON_CONS": True,
            "z_heat_max": 25,
            "zprime_step_factor": 1.1,
        },
    ],
    "minimize_mem": [
        9,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "PHOTON_CONS": True,
            "z_heat_max": 25,
            "zprime_step_factor": 1.1,
            "MINIMIZE_MEMORY": True,
        },
    ],
    "mdz_and_tsfluct_nthreads": [
        8.5,
        {
            "N_THREADS": 2,
            "USE_FFTW_WISDOM": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
            "PHOTON_CONS": True,
            "z_heat_max": 25,
            "zprime_step_factor": 1.1,
        },
    ],
    "halo_field": [9, {"USE_HALO_FIELD": True}],
    "halo_field_mdz": [
        8.5,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "USE_HALO_FIELD": True,
            "USE_TS_FLUCT": True,
            "z_heat_max": 25,
            "zprime_step_factor": 1.1,
        },
    ],
    "halo_field_mdz_highres": [
        8.5,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "USE_HALO_FIELD": True,
            "USE_TS_FLUCT": False,
            "PERTURB_ON_HIGH_RES": True,
            "N_THREADS": 4,
            "z_heat_max": 25,
            "zprime_step_factor": 1.1,
        },
    ],
    "mdz_tsfluct_nthreads": [
        12.0,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "USE_TS_FLUCT": True,
            "PERTURB_ON_HIGH_RES": False,
            "N_THREADS": 4,
            "z_heat_max": 25,
            "zprime_step_factor": 1.2,
            "NUM_FILTER_STEPS_FOR_Ts": 4,
            "USE_INTERPOLATION_TABLES": False,
        },
    ],
    "ts_fluct_no_tables": [
        12.0,
        {
            "USE_TS_FLUCT": True,
            "N_THREADS": 4,
            "z_heat_max": 25,
            "zprime_step_factor": 1.2,
            "NUM_FILTER_STEPS_FOR_Ts": 4,
            "USE_INTERPOLATION_TABLES": False,
        },
    ],
    "minihalos_no_tables": [
        12.0,
        {
            "USE_MINI_HALOS": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "USE_TS_FLUCT": True,
            "N_THREADS": 4,
            "z_heat_max": 25,
            "zprime_step_factor": 1.1,
            "NUM_FILTER_STEPS_FOR_Ts": 4,
            "USE_INTERPOLATION_TABLES": False,
        },
    ],
    "fast_fcoll_hiz": [
        18,
        {"N_THREADS": 4, "FAST_FCOLL_TABLES": True, "USE_INTERPOLATION_TABLES": True},
    ],
    "fast_fcoll_lowz": [
        8,
        {"N_THREADS": 4, "FAST_FCOLL_TABLES": True, "USE_INTERPOLATION_TABLES": True},
    ],
    "relvel": [
        18,
        {
            "z_heat_max": 25,
            "USE_MINI_HALOS": True,
            "zprime_step_factor": 1.1,
            "N_THREADS": 4,
            "NUM_FILTER_STEPS_FOR_Ts": 8,
            "USE_INTERPOLATION_TABLES": True,
            "FAST_FCOLL_TABLES": True,
            "USE_RELATIVE_VELOCITIES": True,
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
}

if len(set(OPTIONS.keys())) != len(list(OPTIONS.keys())):
    raise ValueError("There is a non-unique option name!")

OPTIONS_PT = {
    "simple": [10, {}],
    "no2lpt": [10, {"USE_2LPT": False}],
    "linear": [10, {"EVOLVE_DENSITY_LINEARLY": 1}],
    "highres": [10, {"PERTURB_ON_HIGH_RES": True}],
}

if len(set(OPTIONS_PT.keys())) != len(list(OPTIONS_PT.keys())):
    raise ValueError("There is a non-unique option_pt name!")

OPTIONS_HALO = {"halo_field": [9, {"USE_HALO_FIELD": True}]}

if len(set(OPTIONS_HALO.keys())) != len(list(OPTIONS_HALO.keys())):
    raise ValueError("There is a non-unique option_halo name!")


def get_defaults(kwargs, cls):
    return {k: kwargs.get(k, v) for k, v in cls._defaults_.items()}


def get_all_defaults(kwargs):
    flag_options = get_defaults(kwargs, FlagOptions)
    astro_params = get_defaults(kwargs, AstroParams)
    cosmo_params = get_defaults(kwargs, CosmoParams)
    user_params = get_defaults(kwargs, UserParams)
    return user_params, cosmo_params, astro_params, flag_options


def get_all_options(redshift, **kwargs):
    user_params, cosmo_params, astro_params, flag_options = get_all_defaults(kwargs)
    user_params.update(DEFAULT_USER_PARAMS)
    out = {
        "redshift": redshift,
        "user_params": user_params,
        "cosmo_params": cosmo_params,
        "astro_params": astro_params,
        "flag_options": flag_options,
        "use_interp_perturb_field": kwargs.get("use_interp_perturb_field", False),
        "random_seed": SEED,
    }

    for key in kwargs:
        if key.upper() in (k.upper() for k in global_params.keys()):
            out[key] = kwargs[key]
    return out


def get_all_options_ics(**kwargs):
    user_params, cosmo_params, astro_params, flag_options = get_all_defaults(kwargs)
    user_params.update(DEFAULT_USER_PARAMS)
    out = {
        "user_params": user_params,
        "cosmo_params": cosmo_params,
        "random_seed": SEED,
    }

    for key in kwargs:
        if key.upper() in (k.upper() for k in global_params.keys()):
            out[key] = kwargs[key]
    return out


def get_all_options_halo(redshift, **kwargs):
    user_params, cosmo_params, astro_params, flag_options = get_all_defaults(kwargs)
    user_params.update(DEFAULT_USER_PARAMS)
    out = {
        "redshift": redshift,
        "user_params": user_params,
        "cosmo_params": cosmo_params,
        "astro_params": astro_params,
        "flag_options": flag_options,
        "random_seed": SEED,
    }

    for key in kwargs:
        if key.upper() in (k.upper() for k in global_params.keys()):
            out[key] = kwargs[key]
    return out


def produce_coeval_power_spectra(redshift, **kwargs):
    options = get_all_options(redshift, **kwargs)

    with config.use(ignore_R_BUBBLE_MAX_error=True):
        coeval = run_coeval(write=write_ics_only_hook, **options)
    p = {}

    for field in COEVAL_FIELDS:
        if hasattr(coeval, field):
            p[field], k = get_power(
                getattr(coeval, field), boxlength=coeval.user_params.BOX_LEN
            )

    return k, p, coeval


def produce_lc_power_spectra(redshift, **kwargs):
    options = get_all_options(redshift, **kwargs)

    # NOTE: this is here only so that we get the same answer as previous versions,
    #       which have a bug where the max_redshift gets set higher than it needs to be.
    flag_options = FlagOptions(options.pop("flag_options"))
    if flag_options.INHOMO_RECO or flag_options.USE_TS_FLUCT:
        max_redshift = options.get("z_heat_max", global_params.Z_HEAT_MAX)
        del options["redshift"]
    else:
        max_redshift = options.pop("redshift") + 2

    lcn = RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=redshift,
        max_redshift=max_redshift,
        quantities=[
            k
            for k in LIGHTCONE_FIELDS
            if (
                flag_options.USE_TS_FLUCT
                or k not in ("Ts_box", "x_e_box", "Tk_box", "J_21_LW_box")
            )
        ],
        resolution=UserParams(options["user_params"]).cell_size,
    )

    with config.use(ignore_R_BUBBLE_MAX_error=True):
        lightcone = run_lightcone(
            lightconer=lcn,
            write=write_ics_only_hook,
            flag_options=flag_options,
            **options,
        )

    p = {}
    for field in LIGHTCONE_FIELDS:
        if hasattr(lightcone, field):
            p[field], k = get_power(
                getattr(lightcone, field), boxlength=lightcone.lightcone_dimensions
            )

    return k, p, lightcone


def produce_perturb_field_data(redshift, **kwargs):
    options = get_all_options(redshift, **kwargs)
    options_ics = get_all_options_ics(**kwargs)

    out = {
        key: kwargs[key]
        for key in kwargs
        if key.upper() in (k.upper() for k in global_params.keys())
    }

    velocity_normalisation = 1e16

    with config.use(regenerate=True, write=False):
        init_box = initial_conditions(**options_ics)
        pt_box = perturb_field(redshift=redshift, init_boxes=init_box, **out)

    p_dens, k_dens = get_power(
        pt_box.density,
        boxlength=options["user_params"]["BOX_LEN"],
    )
    p_vel, k_vel = get_power(
        pt_box.velocity * velocity_normalisation,
        boxlength=options["user_params"]["BOX_LEN"],
    )

    def hist(kind, xmin, xmax, nbins):
        data = getattr(pt_box, kind)
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
    X_vel, Y_vel = hist("velocity", -2, 2, 50)

    return k_dens, p_dens, k_vel, p_vel, X_dens, Y_dens, X_vel, Y_vel, init_box


def produce_halo_field_data(redshift, **kwargs):
    options_halo = get_all_options_halo(redshift, **kwargs)

    with config.use(regenerate=True, write=False):
        pt_halos = perturb_halo_list(**options_halo)

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


def write_ics_only_hook(obj, **params):
    if isinstance(obj, InitialConditions):
        obj.write(**params)


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

        fl.attrs["HII_DIM"] = coeval.user_params.HII_DIM
        fl.attrs["DIM"] = coeval.user_params.DIM
        fl.attrs["BOX_LEN"] = coeval.user_params.BOX_LEN

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
    if os.path.exists(fname):
        if force:
            os.remove(fname)
        else:
            return fname

    with h5py.File(fname, "w") as fl:
        for k, v in kwargs.items():
            fl.attrs[k] = v

        fl.attrs["HII_DIM"] = init_box.user_params.HII_DIM
        fl.attrs["DIM"] = init_box.user_params.DIM
        fl.attrs["BOX_LEN"] = init_box.user_params.BOX_LEN

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
    if os.path.exists(fname):
        if force:
            os.remove(fname)
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
    global_params.ZPRIME_STEP_FACTOR = DEFAULT_ZPRIME_STEP_FACTOR

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
                    os.remove(fl)
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
            fl.stem.startswith("power_spectra")
            and fl.stem.split("power_spectra_")[-1] in OPTIONS
        ):
            continue
        elif (
            fl.stem.startswith("perturb_field_data")
            and fl.stem.split("perturb_field_data_")[-1] in OPTIONS_PT
        ):
            continue
        elif (
            fl.stem.startswith("halo_field_data")
            and fl.stem.split("halo_field_data_")[-1] in OPTIONS_HALO
        ):
            continue

        if qs.confirm(f"Remove {fl}?").ask():
            fl.unlink()


if __name__ == "__main__":
    main()
