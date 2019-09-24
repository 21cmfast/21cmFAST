import hashlib
import os

import h5py
from powerbox import get_power

from py21cmfast import (
    run_coeval,
    run_lightcone,
    FlagOptions,
    AstroParams,
    CosmoParams,
    UserParams,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")


def _get_defaults(kwargs, cls):
    return {k: kwargs.get(k, v) for k, v in cls._defaults_.items()}


def _get_all_defaults(kwargs):
    flag_options = _get_defaults(kwargs, FlagOptions)
    astro_params = _get_defaults(kwargs, AstroParams)
    cosmo_params = _get_defaults(kwargs, CosmoParams)
    user_params = _get_defaults(kwargs, UserParams)
    return user_params, cosmo_params, astro_params, flag_options


def produce_power_spectra(**kwargs):
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
        regenerate=True,
        z_step_factor=kwargs.get("z_step_factor", None),
        z_heat_max=kwargs.get("z_heat_max", None),
        use_interp_perturb_field=kwargs.get("use_interp_perturb_field", False),
        random_seed=12345,
    )

    lightcone = run_lightcone(
        redshift=redshift,
        max_redshift=redshift + 2,
        user_params=user_params,
        cosmo_params=cosmo_params,
        astro_params=astro_params,
        flag_options=flag_options,
        regenerate=True,
        write=False,
        z_step_factor=kwargs.get("z_step_factor", 1.02),
        z_heat_max=kwargs.get("z_heat_max", None),
        use_interp_perturb_field=kwargs.get("use_interp_perturb_field", False),
        random_seed=12345,
    )

    p, k = get_power(brightness_temp.brightness_temp, boxlength=user_params["BOX_LEN"])
    p_l, k_l = get_power(
        lightcone.brightness_temp, boxlength=lightcone.lightcone_dimensions
    )

    # TODO: might be better to ensure that only kwargs that specify non-defaults are
    #      kept in the filename
    fname = (
        f"power_spectra_z{redshift:.2f}_Z{kwargs['z_heat_max']}_"
        f"{hashlib.md5(str(kwargs).encode()).hexdigest()}.h5"
    )

    if os.path.exists(os.path.join(DATA_PATH, fname)):
        os.remove(os.path.join(DATA_PATH, fname))

    with h5py.File(os.path.join(DATA_PATH, fname)) as fl:
        for k, v in kwargs.items():
            fl.attrs[k] = v

        fl.attrs["HII_DIM"] = user_params["HII_DIM"]
        fl.attrs["DIM"] = user_params["DIM"]
        fl.attrs["BOX_LEN"] = user_params["BOX_LEN"]

        fl["power_coeval"] = p
        fl["k_coeval"] = k

        fl["power_lc"] = p_l
        fl["k_lc"] = k_l

        fl["xHI"] = lightcone.global_xHI
        fl["Tb"] = lightcone.global_brightness_temp

    print(f"Produced {fname} with {kwargs}")


if __name__ == "__main__":
    _redshifts = [6, 7, 9, 11, 15, 20, 30]
    _z_step_factor = [1.02, 1.05]
    _z_heat_max = [35, 40, 25]
    _hmf = [0, 1, 2, 3]
    _interp_pf, _mdz, _rsd, _inh_reco, _ts, _mmin_mass, _wisdom = [[False, True]] * 7

    for redshift, zsp, zhm, hmf, ipf, mdz, rsd, ihr, ts, mmin, wisdom in (
        [6, 1.02, 35, 1, False, False, False, False, False, False, False],
        [11, 1.05, 35, 1, False, False, False, False, False, False, False],
        [30, 1.02, 40, 1, False, False, False, False, False, False, False],
        [6, 1.05, 25, 0, False, False, False, False, False, False, False],
        [8, 1.02, 35, 1, True, False, False, False, False, False, False],
        [7, 1.02, 35, 1, False, True, False, False, False, False, False],
        [9, 1.02, 35, 1, False, False, True, False, False, False, False],
        [10, 1.03, 35, 2, False, False, False, True, False, False, False],
        [15, 1.02, 35, 3, False, False, False, False, True, False, False],
        [20, 1.02, 45, 4, False, False, False, False, False, True, False],
        [35, 1.02, 35, 1, False, False, False, False, False, False, True],
    ):
        produce_power_spectra(
            redshift=redshift,
            z_step_factor=zsp,
            z_heat_max=zhm,
            HMF=hmf,
            USE_FFTW_WISDOM=wisdom,
            USE_MASS_DEPENDENT_ZETA=mdz,
            SUBCELL_RSD=rsd,
            INHOMO_RECO=ihr,
            USE_TS_FLUCT=ts,
            M_MIN_in_Mass=mmin,
            use_interp_perturb_field=ipf,
        )
