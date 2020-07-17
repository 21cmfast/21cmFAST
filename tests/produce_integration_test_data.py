"""
Produce integration test data, which is tested by the `test_integration_features.py`
tests. One thing to note here is that all redshifts are reasonably high.

This is necessary, because low redshifts mean that neutral fractions are small,
and then numerical noise gets relatively more important, and can make the comparison
fail at the tens-of-percent level.
"""
import glob
import os
import sys
import tempfile

import h5py
import numpy as np
from powerbox import get_power

from py21cmfast import AstroParams
from py21cmfast import CosmoParams
from py21cmfast import FlagOptions
from py21cmfast import UserParams
from py21cmfast import config
from py21cmfast import determine_halo_list
from py21cmfast import global_params
from py21cmfast import initial_conditions
from py21cmfast import perturb_field
from py21cmfast import perturb_halo_list
from py21cmfast import run_coeval
from py21cmfast import run_lightcone

SEED = 12345
DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")
DEFAULT_USER_PARAMS = {
    "HII_DIM": 50,
    "DIM": 150,
    "BOX_LEN": 100,
    "NO_RNG": True,
    "USE_INTERPOLATION_TABLES": True,
}
DEFAULT_ZPRIME_STEP_FACTOR = 1.04


OPTIONS = (
    [12, {}],
    [12, {"PERTURB_ON_HIGH_RES": True}],
    [11, {"zprime_step_factor": 1.02}],
    [30, {"z_heat_max": 40}],
    [13, {"zprime_step_factor": 1.05, "z_heat_max": 25, "HMF": 0}],
    [16, {"interp_perturb_field": True}],
    [14, {"USE_MASS_DEPENDENT_ZETA": True}],
    [9, {"SUBCELL_RSD": True}],
    [10, {"INHOMO_RECO": True}],
    [16, {"HMF": 3, "USE_TS_FLUCT": True}],
    [20, {"z_heat_max": 45, "M_MIN_in_Mass": True, "HMF": 2}],
    [35, {"USE_FFTW_WISDOM": True}],
    [
        18,
        {
            "z_heat_max": 25,
            "USE_MINI_HALOS": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
            "zprime_step_factor": 1.2,
            "N_THREADS": 4,
            "USE_FFTW_WISDOM": True,
            "NUM_FILTER_STEPS_FOR_Ts": 8,
        },
    ],
    [8, {"N_THREADS": 2}],
    [10, {"PHOTON_CONS": True}],
    [
        8.5,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "PHOTON_CONS": True,
            "z_heat_max": 25,
            "zprime_step_factor": 1.1,
        },
    ],
    [
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
    [
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
    [9, {"USE_HALO_FIELD": True}],
    [
        8.5,
        {
            "USE_MASS_DEPENDENT_ZETA": True,
            "USE_HALO_FIELD": True,
            "USE_TS_FLUCT": True,
            "z_heat_max": 25,
            "zprime_step_factor": 1.1,
        },
    ],
    [
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
    [
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
    [
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
    [
        12.0,
        {
            "USE_MINI_HALOS": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "USE_TS_FLUCT": True,
            "N_THREADS": 4,
            "z_heat_max": 25,
            "zprime_step_factor": 1.2,
            "NUM_FILTER_STEPS_FOR_Ts": 4,
            "USE_INTERPOLATION_TABLES": False,
        },
    ],
)

OPTIONS_PT = (
    [10, {}],
    [10, {"SECOND_ORDER_LPT_CORRECTIONS": 0}],
    [10, {"EVOLVE_DENSITY_LINEARLY": 1}],
    [10, {"PERTURB_ON_HIGH_RES": True}],
)

OPTIONS_HALO = ([9, {"USE_HALO_FIELD": True}],)


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

    coeval = run_coeval(**options)
    p, k = get_power(coeval.brightness_temp, boxlength=coeval.user_params.BOX_LEN,)

    return k, p, coeval


def produce_lc_power_spectra(redshift, **kwargs):
    options = get_all_options(redshift, **kwargs)
    lightcone = run_lightcone(max_redshift=options["redshift"] + 2, **options)

    p, k = get_power(
        lightcone.brightness_temp, boxlength=lightcone.lightcone_dimensions
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
        pt_box.density, boxlength=options["user_params"]["BOX_LEN"],
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
            data, bins=np.linspace(xmin, xmax, nbins), range=[xmin, xmax], normed=True,
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


def get_filename(redshift, **kwargs):
    # get sorted keys
    kwargs = {k: kwargs[k] for k in sorted(kwargs)}
    string = "_".join(f"{k}={v}" for k, v in kwargs.items())
    fname = f"power_spectra_z{redshift:.2f}_{string}.h5"

    return os.path.join(DATA_PATH, fname)


def get_filename_pt(redshift, **kwargs):
    # get sorted keys
    kwargs = {k: kwargs[k] for k in sorted(kwargs)}
    string = "_".join(f"{k}={v}" for k, v in kwargs.items())
    fname = f"perturb_field_data_z{redshift:.2f}_{string}.h5"

    return os.path.join(DATA_PATH, fname)


def get_filename_halo(redshift, **kwargs):
    # get sorted keys
    kwargs = {k: kwargs[k] for k in sorted(kwargs)}
    string = "_".join(f"{k}={v}" for k, v in kwargs.items())
    fname = f"halo_field_data_z{redshift:.2f}_{string}.h5"

    return os.path.join(DATA_PATH, fname)


def produce_power_spectra_for_tests(redshift, force, direc, **kwargs):
    fname = get_filename(redshift, **kwargs)

    # Need to manually remove it, otherwise h5py tries to add to it
    if os.path.exists(fname):
        if force:
            os.remove(fname)
        else:
            return fname

    # For tests, we *don't* want to use cached boxes, but we also want to use the
    # cache between the power spectra and lightcone. So we create a temporary
    # directory in which to cache results.
    with config.use(direc=direc):
        k, p, coeval = produce_coeval_power_spectra(redshift, **kwargs)
        k_l, p_l, lc = produce_lc_power_spectra(redshift, **kwargs)

    with h5py.File(fname, "w") as fl:
        for k, v in kwargs.items():
            fl.attrs[k] = v

        fl.attrs["HII_DIM"] = coeval.user_params.HII_DIM
        fl.attrs["DIM"] = coeval.user_params.DIM
        fl.attrs["BOX_LEN"] = coeval.user_params.BOX_LEN

        fl["power_coeval"] = p
        fl["k_coeval"] = k

        fl["power_lc"] = p_l
        fl["k_lc"] = k_l

        fl["xHI"] = lc.global_xH
        fl["Tb"] = lc.global_brightness_temp

    print(f"Produced {fname} with {kwargs}")
    return fname


def produce_data_for_perturb_field_tests(redshift, force, **kwargs):
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

    fname = get_filename_pt(redshift, **kwargs)

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


def produce_data_for_halo_field_tests(redshift, force, **kwargs):

    pt_halos = produce_halo_field_data(redshift, **kwargs)

    fname = get_filename_halo(redshift, **kwargs)

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


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("21cmFAST")

    lvl = "WARNING"
    for arg in sys.argv:
        if arg.startswith("--log"):
            lvl = arg.split("--log")[-1]
    lvl = getattr(logging, lvl)
    logger.setLevel(lvl)

    global_params.ZPRIME_STEP_FACTOR = DEFAULT_ZPRIME_STEP_FACTOR

    force = "--force" in sys.argv
    remove = "--no-clean" not in sys.argv
    pt_only = "--pt-only" in sys.argv
    no_pt = "--no-pt" in sys.argv
    no_halo = "--no-halo" in sys.argv

    nums = range(len(OPTIONS))
    for arg in sys.argv:
        if arg.startswith("--nums="):
            nums = [int(x) for x in arg.split("=")[-1].split(",")]
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
        for redshift, kwargs in [OPTIONS[n] for n in nums]:
            fnames.append(
                produce_power_spectra_for_tests(redshift, force, direc, **kwargs)
            )

    if not no_pt:
        for redshift, kwargs in OPTIONS_PT:
            fnames.append(
                produce_data_for_perturb_field_tests(redshift, force, **kwargs)
            )

    if not no_halo:
        for redshift, kwargs in OPTIONS_HALO:
            fnames.append(produce_data_for_halo_field_tests(redshift, force, **kwargs))

    # Remove extra files that
    if not (nums or pt_only or no_pt or no_halo):
        all_files = glob.glob(os.path.join(DATA_PATH, "*"))
        for fl in all_files:
            if fl not in fnames:
                if remove:
                    print(f"Removing old file: {fl}")
                    os.remove(fl)
                else:
                    print(f"File is now redundant and can be removed: {fl}")
