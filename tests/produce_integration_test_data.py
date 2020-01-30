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

import h5py
from powerbox import get_power

from py21cmfast import AstroParams
from py21cmfast import CosmoParams
from py21cmfast import FlagOptions
from py21cmfast import UserParams
from py21cmfast import global_params
from py21cmfast import run_coeval
from py21cmfast import run_lightcone

SEED = 12345
DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")
DEFAULT_USER_PARAMS = {"HII_DIM": 50, "DIM": 150, "BOX_LEN": 100}
DEFAULT_ZPRIME_STEP_FACTOR = 1.04

OPTIONS = (
    [12, {}],
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
)


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
        "regenerate": True,
        "write": False,
        "use_interp_perturb_field": kwargs.get("use_interp_perturb_field", False),
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


def get_filename(redshift, **kwargs):
    # get sorted keys
    kwargs = {k: kwargs[k] for k in sorted(kwargs)}
    string = "_".join(f"{k}={v}" for k, v in kwargs.items())
    fname = f"power_spectra_z{redshift:.2f}_{string}.h5"

    return os.path.join(DATA_PATH, fname)


def produce_power_spectra_for_tests(redshift, **kwargs):
    k, p, coeval = produce_coeval_power_spectra(redshift, **kwargs)
    k_l, p_l, lc = produce_lc_power_spectra(redshift, **kwargs)

    fname = get_filename(redshift, **kwargs)

    # Need to manually remove it, otherwise h5py tries to add to it
    if os.path.exists(fname):
        os.remove(fname)

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

        fl["xHI"] = lc.global_xHI
        fl["Tb"] = lc.global_brightness_temp

    print(f"Produced {fname} with {kwargs}")


if __name__ == "__main__":
    global_params.ZPRIME_STEP_FACTOR = DEFAULT_ZPRIME_STEP_FACTOR

    for redshift, kwargs in OPTIONS:
        produce_power_spectra_for_tests(redshift, **kwargs)
