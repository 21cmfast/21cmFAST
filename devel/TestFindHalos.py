"""Quick test of finding halos."""

from py21cmfast import (
    AstroParams,
    CosmoParams,
    FlagOptions,
    UserParams,
    determine_halo_list,
    initial_conditions,
    perturb_field,
)
from py21cmfast._utils import StructInstanceWrapper
from py21cmfast.c_21cmfast import ffi, lib

global_params = StructInstanceWrapper(lib.global_params, ffi)

user_params = UserParams(
    DIM=150,
    HII_DIM=50,
    BOX_LEN=150.0,
    USE_FFTW_WISDOM=False,
    HMF=1,
    N_THREADS=1,
    NO_RNG=True,
    PERTURB_ON_HIGH_RES=True,
)

flag_options = FlagOptions(
    USE_MASS_DEPENDENT_ZETA=True,
    USE_TS_FLUCT=False,
    INHOMO_RECO=False,
    SUBCELL_RSD=False,
    M_MIN_in_Mass=False,
    PHOTON_CONS=False,
)

if __name__ == "__main__":
    random_seed = 42

    cosmo_params = CosmoParams(
        OMb=0.0486, OMm=0.3075, POWER_INDEX=0.97, SIGMA_8=0.82, hlittle=0.6774
    )

    astro_params = AstroParams(
        ALPHA_ESC=-0.5,
        ALPHA_STAR=0.5,
        F_ESC10=-1.30102999566,
        F_STAR10=-1.0,
        L_X=40.5,
        M_TURN=8.7,
        NU_X_THRESH=500.0,
        X_RAY_SPEC_INDEX=1.0,
        t_STAR=0.5,
        R_BUBBLE_MAX=15.0,
    )

    init_box = initial_conditions(
        user_params=user_params,
        cosmo_params=cosmo_params,
        random_seed=random_seed,
        regenerate=True,
        write=False,
    )

    redshift = 9.0

    pt_box = perturb_field(
        redshift=redshift,
        init_boxes=init_box,
        user_params=user_params,
        cosmo_params=cosmo_params,
    )

    halos = determine_halo_list(
        redshift=redshift,
        init_boxes=init_box,
        user_params=user_params,
        cosmo_params=cosmo_params,
        astro_params=astro_params,
        flag_options=flag_options,
        regenerate=True,
        write=False,
        OPTIMIZE=False,
    )

    print(halos.halo_masses)
