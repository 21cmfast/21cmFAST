from py21cmfast import (
    run_coeval,
    run_lightcone,
    FlagOptions,
    AstroParams,
    CosmoParams,
    UserParams,
)
from powerbox import get_power
import h5py
import os
import hashlib

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")


def _get_defaults(kwargs, cls):
    return {k: kwargs.get(k.lower(), v) for k, v in cls._defaults_.items()}


def _get_all_defaults(kwargs):
    flag_options = _get_defaults(kwargs, FlagOptions)
    astro_params = _get_defaults(kwargs, AstroParams)
    cosmo_params = _get_defaults(kwargs, CosmoParams)
    user_params = _get_defaults(kwargs, UserParams)
    return user_params, cosmo_params, astro_params, flag_options


def produce_coeval_power_spectra(**kwargs):
    user_params, cosmo_params, astro_params, flag_options = _get_all_defaults(kwargs)
    # Now ensure some properties that make the box small
    user_params["HII_DIM"] = 50
    user_params["DIM"] = 150
    user_params["BOX_LEN"] = 100

    init, perturb, ionize, brightness_temp = run_coeval(
        redshift=kwargs.get("redshift", 7),
        user_params=user_params,
        cosmo_params=cosmo_params,
        astro_params=astro_params,
        flag_options=flag_options,
        z_step_factor=kwargs.get("z_step_factor", None),
        z_heat_max=kwargs.get("z_heat_max", None),
        use_interp_perturb_field=kwargs.get("use_interp_perturb_field", False),
        random_seed=12345,
    )

    p, k = get_power(brightness_temp.brightness_temp, boxlength=user_params["BOX_LEN"])

    # TODO: might be better to ensure that only kwargs that specify non-defaults are
    #      kept in the filename
    fname = "power_spectra_coeval_{}.h5".format(
        hashlib.md5(str(kwargs).encode()).hexdigest()
    )

    with h5py.File(os.path.join(DATA_PATH, fname)) as fl:
        for k, v in kwargs.items():
            fl.attrs[k] = v

        fl["power"] = p
        fl["k"] = k
