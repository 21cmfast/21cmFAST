"""Low-level python wrappers of C functions."""

import logging
import numpy as np
from functools import cache
from scipy.interpolate import interp1d
from typing import Literal, Sequence

import py21cmfast.c_21cmfast as lib
from ..drivers.param_config import InputParameters
from ._utils import _process_exitcode
from .globals import global_params
from .inputs import AstroParams, CosmoParams, FlagOptions, UserParams
from .outputs import InitialConditions, PerturbHaloField

logger = logging.getLogger(__name__)


def get_expected_nhalo(
    redshift: float,
    user_params: UserParams,
    cosmo_params: CosmoParams,
) -> int:
    """Get the expected number of halos in a given box.

    Parameters
    ----------
    redshift : float
        The redshift at which to calculate the halo list.
    user_params : :class:`~UserParams`
        User params defining the box size and resolution.
    cosmo_params : :class:`~CosmoParams`
        Cosmological parameters.
    """
    return lib.expected_nhalo(redshift, user_params.cstruct, cosmo_params.cstruct)


def get_halo_list_buffer_size(
    redshift: float,
    user_params: UserParams,
    cosmo_params: CosmoParams,
    min_size: int = 1000000,
) -> int:
    """Compute the required size of the memory buffer to hold a halo list.

    Parameters
    ----------
    redshift : float
        The redshift at which to calculate the halo list.
    user_params : :class:`~UserParams`
        User params defining the box size and resolution.
    cosmo_params : :class:`~CosmoParams`
        Cosmological parameters.
    min_size : int, optional
        A minimum size to be used as the buffer.
    """
    # find the buffer size from expected halos in the box
    hbuffer_size = get_expected_nhalo(redshift, user_params, cosmo_params)
    hbuffer_size = int((hbuffer_size + 1) * user_params.MAXHALO_FACTOR)

    # set a minimum in case of fluctuation at high z
    return int(max(hbuffer_size, min_size))


def compute_tau(
    *,
    redshifts: Sequence[float],
    global_xHI: Sequence[float],
    user_params: UserParams | dict | None = None,
    cosmo_params: CosmoParams | dict | None = None,
) -> float:
    """Compute the optical depth to reionization under the given model.

    Parameters
    ----------
    redshifts : array-like
        Redshifts defining an evolution of the neutral fraction.
    global_xHI : array-like
        The mean neutral fraction at `redshifts`.
    user_params : :class:`~inputs.UserParams`
        Parameters defining the simulation run.
    cosmo_params : :class:`~inputs.CosmoParams`
        Cosmological parameters.

    Returns
    -------
    tau : float
        The optional depth to reionization

    Raises
    ------
    ValueError :
        If `redshifts` and `global_xHI` have inconsistent length or if redshifts are not
        in ascending order.
    """
    inputs = InputParameters(user_params, cosmo_params)

    if len(redshifts) != len(global_xHI):
        raise ValueError("redshifts and global_xHI must have same length")

    if not np.all(np.diff(redshifts) > 0):
        raise ValueError("redshifts and global_xHI must be in ascending order")

    # Convert the data to the right type
    redshifts = np.array(redshifts, dtype="float32")
    global_xHI = np.array(global_xHI, dtype="float32")

    # WIP: CFFI Refactor
    # z = ffi.cast("float *", ffi.from_buffer(redshifts))
    # xHI = ffi.cast("float *", ffi.from_buffer(global_xHI))
    z = redshifts
    xHI = global_xHI

    # Run the C code
    return lib.ComputeTau(
        inputs.user_params.cstruct, inputs.cosmo_params.cstruct, len(
            redshifts), z, xHI
    )


def compute_luminosity_function(
    *,
    redshifts: Sequence[float],
    user_params: UserParams | dict | None = None,
    cosmo_params: CosmoParams | dict | None = None,
    astro_params: AstroParams | dict | None = None,
    flag_options: FlagOptions | dict | None = None,
    nbins: int = 100,
    mturnovers: np.ndarray | None = None,
    mturnovers_mini: np.ndarray | None = None,
    component: Literal["both", "acg", "mcg"] = "both",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a the luminosity function over a given number of bins and redshifts.

    Parameters
    ----------
    redshifts : array-like
        The redshifts at which to compute the luminosity function.
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    astro_params : :class:`~AstroParams`, optional
        The astrophysical parameters defining the course of reionization.
    flag_options : :class:`~FlagOptions`, optional
        Some options passed to the reionization routine.
    nbins : int, optional
        The number of luminosity bins to produce for the luminosity function.
    mturnovers : array-like, optional
        The turnover mass at each redshift for massive halos (ACGs).
        Only required when USE_MINI_HALOS is True.
    mturnovers_mini : array-like, optional
        The turnover mass at each redshift for minihalos (MCGs).
        Only required when USE_MINI_HALOS is True.
    component : str, {'both', 'acg', 'mcg}
        The component of the LF to be calculated.

    Returns
    -------
    Muvfunc : np.ndarray
        Magnitude array (i.e. brightness). Shape [nredshifts, nbins]
    Mhfunc : np.ndarray
        Halo mass array. Shape [nredshifts, nbins]
    lfunc : np.ndarray
        Number density of haloes corresponding to each bin defined by `Muvfunc`.
        Shape [nredshifts, nbins].
    """
    user_params = UserParams.new(user_params)
    cosmo_params = CosmoParams.new(cosmo_params)
    flag_options = FlagOptions.new(flag_options)
    astro_params = AstroParams.new(astro_params, flag_options=flag_options)

    redshifts = np.array(redshifts, dtype="float32")
    if flag_options.USE_MINI_HALOS:
        if component in ["both", "acg"]:
            if mturnovers is None:
                raise ValueError(
                    "calculating ACG LFs with mini-halo feature requires users to "
                    "specify mturnovers!"
                )

            mturnovers = np.array(mturnovers, dtype=np.float32)
            if len(mturnovers) != len(redshifts):
                raise ValueError(
                    f"mturnovers ({len(mturnovers)}) does not match the length of "
                    f"redshifts ({len(redshifts)})"
                )
        if component in ["both", "mcg"]:
            if mturnovers_mini is None:
                raise ValueError(
                    "calculating MCG LFs with mini-halo feature requires users to "
                    "specify mturnovers_MINI!"
                )

            mturnovers_mini = np.array(mturnovers_mini, dtype="float32")
            if len(mturnovers_mini) != len(redshifts):
                raise ValueError(
                    f"mturnovers_MINI ({len(mturnovers)}) does not match the length of "
                    f"redshifts ({len(redshifts)})"
                )

    else:
        mturnovers = (
            np.zeros(len(redshifts), dtype=np.float32) +
            10**astro_params.M_TURN
        )
        component = "acg"

    lfunc = np.zeros(len(redshifts) * nbins)
    Muvfunc = np.zeros(len(redshifts) * nbins)
    Mhfunc = np.zeros(len(redshifts) * nbins)

    lfunc.shape = (len(redshifts), nbins)
    Muvfunc.shape = (len(redshifts), nbins)
    Mhfunc.shape = (len(redshifts), nbins)

    # WIP: CFFI Refactor
    # c_Muvfunc = ffi.cast("double *", ffi.from_buffer(Muvfunc))
    # c_Mhfunc = ffi.cast("double *", ffi.from_buffer(Mhfunc))
    # c_lfunc = ffi.cast("double *", ffi.from_buffer(lfunc))
    c_Muvfunc = Muvfunc
    c_Mhfunc = Mhfunc
    c_lfunc = lfunc

    lfunc_MINI = np.zeros(len(redshifts) * nbins)
    Muvfunc_MINI = np.zeros(len(redshifts) * nbins)
    Mhfunc_MINI = np.zeros(len(redshifts) * nbins)

    lfunc_MINI.shape = (len(redshifts), nbins)
    Muvfunc_MINI.shape = (len(redshifts), nbins)
    Mhfunc_MINI.shape = (len(redshifts), nbins)

    # WIP: CFFI Refactor
    # c_Muvfunc_MINI = ffi.cast("double *", ffi.from_buffer(Muvfunc_MINI))
    # c_Mhfunc_MINI = ffi.cast("double *", ffi.from_buffer(Mhfunc_MINI))
    # c_lfunc_MINI = ffi.cast("double *", ffi.from_buffer(lfunc_MINI))
    c_Muvfunc_MINI = Muvfunc_MINI
    c_Mhfunc_MINI = Mhfunc_MINI
    c_lfunc_MINI = lfunc_MINI

    if component in ("both", "acg"):
        # Run the C code
        errcode = lib.ComputeLF(
            nbins,
            user_params.cstruct,
            cosmo_params.cstruct,
            astro_params.cstruct,
            flag_options.cstruct,
            1,
            len(redshifts),
            # WIP: CFFI Refactor
            # ffi.cast("float *", ffi.from_buffer(redshifts)),
            # ffi.cast("float *", ffi.from_buffer(mturnovers)),
            redshifts,
            mturnovers,
            c_Muvfunc,
            c_Mhfunc,
            c_lfunc,
        )

        _process_exitcode(
            errcode,
            lib.ComputeLF,
            (
                nbins,
                user_params.cstruct,
                cosmo_params.cstruct,
                astro_params.cstruct,
                flag_options.cstruct,
                1,
                len(redshifts),
            ),
        )

    if component in ("both", "mcg"):
        # Run the C code
        errcode = lib.ComputeLF(
            nbins,
            user_params.cstruct,
            cosmo_params.cstruct,
            astro_params.cstruct,
            flag_options.cstruct,
            2,
            len(redshifts),
            # WIP: CFFI Refactor
            # ffi.cast("float *", ffi.from_buffer(redshifts)),
            # ffi.cast("float *", ffi.from_buffer(mturnovers_mini)),
            redshifts,
            mturnovers_mini,
            c_Muvfunc_MINI,
            c_Mhfunc_MINI,
            c_lfunc_MINI,
        )

        _process_exitcode(
            errcode,
            lib.ComputeLF,
            (
                nbins,
                user_params.cstruct,
                cosmo_params.cstruct,
                astro_params.cstruct,
                flag_options.cstruct,
                2,
                len(redshifts),
            ),
        )

    if component == "both":
        # redo the Muv range using the faintest (most likely MINI) and the brightest (most likely massive)
        lfunc_all = np.zeros(len(redshifts) * nbins)
        Muvfunc_all = np.zeros(len(redshifts) * nbins)
        Mhfunc_all = np.zeros(len(redshifts) * nbins * 2)

        lfunc_all.shape = (len(redshifts), nbins)
        Muvfunc_all.shape = (len(redshifts), nbins)
        Mhfunc_all.shape = (len(redshifts), nbins, 2)

        for iz in range(len(redshifts)):
            Muvfunc_all[iz] = np.linspace(
                np.min([Muvfunc.min(), Muvfunc_MINI.min()]),
                np.max([Muvfunc.max(), Muvfunc_MINI.max()]),
                nbins,
            )
            lfunc_all[iz] = np.log10(
                10
                ** (
                    interp1d(Muvfunc[iz], lfunc[iz], fill_value="extrapolate")(
                        Muvfunc_all[iz]
                    )
                )
                + 10
                ** (
                    interp1d(
                        Muvfunc_MINI[iz], lfunc_MINI[iz], fill_value="extrapolate"
                    )(Muvfunc_all[iz])
                )
            )
            Mhfunc_all[iz] = np.array(
                [
                    interp1d(Muvfunc[iz], Mhfunc[iz], fill_value="extrapolate")(
                        Muvfunc_all[iz]
                    ),
                    interp1d(
                        Muvfunc_MINI[iz], Mhfunc_MINI[iz], fill_value="extrapolate"
                    )(Muvfunc_all[iz]),
                ],
            ).T
        lfunc_all[lfunc_all <= -30] = np.nan
        return Muvfunc_all, Mhfunc_all, lfunc_all
    elif component == "acg":
        lfunc[lfunc <= -30] = np.nan
        return Muvfunc, Mhfunc, lfunc
    elif component == "mcg":
        lfunc_MINI[lfunc_MINI <= -30] = np.nan
        return Muvfunc_MINI, Mhfunc_MINI, lfunc_MINI
    else:
        raise ValueError(
            f"Unknown component '{component}'. Must be 'both', 'acg' or 'mcg'"
        )


@cache
def construct_fftw_wisdoms(
    *,
    user_params: UserParams | dict | None = None,
    cosmo_params: CosmoParams | dict | None = None,
) -> int:
    """Construct all necessary FFTW wisdoms.

    Parameters
    ----------
    user_params : :class:`~inputs.UserParams`
        Parameters defining the simulation run.

    """
    user_params = UserParams.new(user_params)
    cosmo_params = CosmoParams.new(cosmo_params)

    # Run the C code
    if user_params.USE_FFTW_WISDOM:
        return lib.CreateFFTWWisdoms(user_params.cstruct, cosmo_params.cstruct)
    else:
        return 0


# Below are evaulations of certain integrals and interpolation tables used at lower levels in the code.
#  They are mostly used for testing but may be useful in some post-processing applications


def get_delta_crit(user_params, cosmo_params, mass, redshift):
    """Gets the critical collapse density given a mass, redshift and parameters."""
    sigma, _ = evaluate_sigma(user_params, cosmo_params, mass)
    # evaluate_sigma already broadcasts the paramters so we don't need to repeat
    growth = lib.dicke(redshift)
    return get_delta_crit_nu(user_params, sigma, growth)


def get_delta_crit_nu(user_params, sigma, growth):
    """Uses the nu paramters (sigma and growth factor) to get critical density."""
    # None of the parameter structs are used in this function so we don't need a broadcast
    return np.vectorize(lib.get_delta_crit)(user_params.cdict["HMF"], sigma, growth)


def evaluate_sigma(
    user_params: UserParams,
    cosmo_params: CosmoParams,
    masses: Sequence[float],
):
    """
    Evaluates the variance of a mass scale.

    Uses the 21cmfast backend
    """
    lib.Broadcast_struct_global_noastro(
        user_params.cstruct, cosmo_params.cstruct)

    lib.init_ps()
    if user_params.USE_INTERPOLATION_TABLES:
        lib.initialiseSigmaMInterpTable(
            global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
        )

    sigma = np.vectorize(lib.EvaluateSigma)(np.log(masses))
    dsigmasq = np.vectorize(lib.EvaluatedSigmasqdm)(np.log(masses))

    return sigma, dsigmasq


def evaluate_massfunc_cond(
    *,
    user_params: UserParams,
    cosmo_params: CosmoParams,
    astro_params: AstroParams,
    flag_options: FlagOptions,
    M_min: float,
    M_max: float,
    redshift: float,
    cond_param: float,
    cond_array: Sequence[float],
    from_catalog: bool,
):
    """
    Evaluates the conditional mass function integral.

    includes halo number and mass, using the 21cmfast backend
    """
    lib.Broadcast_struct_global_all(
        user_params.cstruct,
        cosmo_params.cstruct,
        astro_params.cstruct,
        flag_options.cstruct,
    )

    lib.init_ps()
    if user_params.USE_INTERPOLATION_TABLES:
        lib.initialiseSigmaMInterpTable(
            global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
        )

    growth_out = lib.dicke(redshift)
    if from_catalog:
        # cond_param is descendant redshift, cond_array are halo masses
        growth_in = lib.dicke(cond_param)
        sigma_cond = np.vectorize(lib.EvaluateSigma)(cond_array)
        delta = (
            np.vectorize(lib.get_delta_crit)(
                user_params.cdict["HMF"], sigma_cond, growth_in
            )
            * growth_out
            / growth_in
        )
        cond_mass = np.exp(cond_array)
    else:
        # cond_param is cell mass, cond_array are deltas
        sigma_cond = lib.EvaluateSigma(np.log(cond_param))
        delta = cond_array
        cond_mass = cond_param

    if user_params.USE_INTERPOLATION_TABLES:
        lib.initialise_dNdM_tables(
            cond_array.min(),
            cond_array.max(),
            np.log(M_min),
            np.log(M_max),
            growth_out,
            growth_in if from_catalog else np.log(cond_mass),
            from_catalog,
        )

    nhalo = (
        np.vectorize(lib.EvaluateNhalo)(
            cond_array,
            growth_out,
            np.log(M_min),
            np.log(M_max),
            cond_mass,
            sigma_cond,
            delta,
        )
        * cond_mass
    )

    mcoll = (
        np.vectorize(lib.EvaluateMcoll)(
            cond_array,
            growth_out,
            np.log(M_min),
            np.log(M_max),
            cond_mass,
            sigma_cond,
            delta,
        )
        * cond_mass
    )

    return nhalo, mcoll


def get_cmf_integral(
    user_params: UserParams,
    cosmo_params: CosmoParams,
    astro_params: AstroParams,
    flag_options: FlagOptions,
    M_min: Sequence[float] | float,
    M_max: Sequence[float] | float,
    M_cond: Sequence[float] | float,
    redshift: float,
    delta: Sequence[float] | float | None = None,
    z_desc: float | None = None,
):
    """
    Evaluates the simple halo mass function intgral.

    Can be computed over a range of conditions and mass bounds.
    """
    lib.Broadcast_struct_global_all(
        user_params.cstruct,
        cosmo_params.cstruct,
        astro_params.cstruct,
        flag_options.cstruct,
    )
    lib.init_ps()
    lib.initialiseSigmaMInterpTable(
        global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
    )

    sigma = np.vectorize(lib.EvaluateSigma)(np.log(M_cond))

    growth_out = lib.dicke(redshift)
    if z_desc is not None and delta is None:
        growth_in = lib.dicke(z_desc)
        delta = (
            (
                np.vectorize(lib.get_delta_crit)(
                    user_params.cdict["HMF"], sigma, growth_in
                )
            )
            * growth_out
            / growth_in
        )

    cmf_integral = np.vectorize(lib.Nhalo_Conditional)(
        growth_out,
        np.log(M_min),
        np.log(M_max),
        M_cond,
        sigma,
        delta,
        0,  # GSL-QAG
    )

    # final shape (delta, sigma, M_min)
    return cmf_integral


def evaluate_inv_massfunc_cond(
    *,
    user_params: UserParams,
    cosmo_params: CosmoParams,
    astro_params: AstroParams,
    flag_options: FlagOptions,
    M_min: float,
    redshift: float,
    cond_param: float,
    cond_array: Sequence[float],
    probabilities: Sequence[float],
    from_catalog: bool,
):
    """
    Evaluates the inverse cumulative halo mass function.

    Used to verify sampling tables.
    """
    lib.Broadcast_struct_global_all(
        user_params.cstruct,
        cosmo_params.cstruct,
        astro_params.cstruct,
        flag_options.cstruct,
    )

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(
        global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
    )

    growth_out = lib.dicke(redshift)

    if from_catalog:
        # cond_param is descendant redshift, cond_array are halo masses
        growth_in = lib.dicke(cond_param)
        cond_mass = np.exp(cond_array)
    else:
        # cond_param is cell mass, cond_array are deltas
        cond_mass = cond_param

    lib.initialise_dNdM_inverse_table(
        cond_array.min(),
        cond_array.max() * 1.01,
        np.log(M_min),
        growth_out,
        growth_in if from_catalog else np.log(cond_mass),
        from_catalog,
    )
    masses = (
        np.vectorize(lib.EvaluateNhaloInv)(
            cond_array,
            probabilities,
        )
        * cond_mass
    )
    return masses


def evaluate_FgtrM_cond(
    *,
    user_params: UserParams,
    cosmo_params: CosmoParams,
    astro_params: AstroParams,
    flag_options: FlagOptions,
    M_min: float,
    M_max: float,
    redshift: float,
    cond_mass: float,
    densities: Sequence[float],
):
    """Evaluates F(>M) using EPS (erfc) from the 21cmfast backend."""
    lib.Broadcast_struct_global_all(
        user_params.cstruct,
        cosmo_params.cstruct,
        astro_params.cstruct,
        flag_options.cstruct,
    )
    lib.init_ps()

    if user_params.USE_INTERPOLATION_TABLES:
        lib.initialiseSigmaMInterpTable(M_min, M_max)

    growth_out = lib.dicke(redshift)
    sigma_min = lib.sigma_z0(M_min)
    sigma_cond = lib.sigma_z0(cond_mass)

    lib.initialise_FgtrM_delta_table(
        densities.min() - 0.01,
        densities.max() + 0.01,
        redshift,
        growth_out,
        sigma_min,
        sigma_cond,
    )

    fcoll = np.vectorize(lib.EvaluateFcoll_delta)(
        densities, growth_out, sigma_min, sigma_cond
    )
    dfcolldz = np.vectorize(lib.EvaluatedFcolldz)(
        densities, redshift, sigma_min, sigma_cond
    )

    return fcoll, dfcolldz


def evaluate_SFRD_z(
    *,
    user_params: UserParams,
    cosmo_params: CosmoParams,
    astro_params: AstroParams,
    flag_options: FlagOptions,
    M_min: float,
    M_max: float,
    redshifts: Sequence[float],
    log10mturnovers: Sequence[float],
    return_integral: bool = False,
):
    """Evaluates the global star formation rate density expected at a range of redshifts."""
    lib.Broadcast_struct_global_all(
        user_params.cstruct,
        cosmo_params.cstruct,
        astro_params.cstruct,
        flag_options.cstruct,
    )
    lib.init_ps()

    if user_params.USE_INTERPOLATION_TABLES:
        lib.initialiseSigmaMInterpTable(M_min, M_max)
    if (
        user_params.INTEGRATION_METHOD_ATOMIC == "GAUSS-LEGENDRE"
        or user_params.INTEGRATION_METHOD_MINI == "GAUSS-LEGENDRE"
    ):
        lib.initialise_GL(np.log(M_min), np.log(M_max))

    ap_c = astro_params.cdict

    mlim_fstar_acg = (
        1e10 * ap_c["F_STAR10"] ** (-1.0 / ap_c["ALPHA_STAR"])
        if ap_c["ALPHA_STAR"]
        else 0.0
    )
    mlim_fstar_mcg = (
        1e7 * ap_c["F_STAR7_MINI"] ** (-1.0 / ap_c["ALPHA_STAR_MINI"])
        if ap_c["ALPHA_STAR_MINI"]
        else 0.0
    )

    sfrd_mini = 0.0
    # Unfortunately we have to do this until we sort out the USE_INTERPOLATION_TABLES flag
    # Since these integrals take forever if the flag is false
    mcrit_atom = (
        np.vectorize(lib.atomic_cooling_threshold)(redshifts)
        if flag_options.USE_MINI_HALOS
        else ap_c["M_TURN"]
    )
    if return_integral:
        sfrd = np.vectorize(lib.Nion_General)(
            redshifts,
            np.log(M_min),
            np.log(M_max),
            mcrit_atom,
            ap_c["ALPHA_STAR"],
            0.0,
            ap_c["F_STAR10"],
            1.0,
            mlim_fstar_acg,
            0.0,
        )
        if flag_options.USE_MINI_HALOS:
            sfrd_mini = np.vectorize(lib.Nion_General_MINI)(
                redshifts[:, None],
                np.log(M_min),
                np.log(M_max),
                10 ** log10mturnovers[None, :],
                mcrit_atom[:, None],
                ap_c["ALPHA_STAR_MINI"],
                0.0,
                ap_c["F_STAR7_MINI"],
                1.0,
                mlim_fstar_mcg,
                0.0,
            )
        return sfrd, sfrd_mini

    if user_params.USE_INTERPOLATION_TABLES:
        lib.initialise_SFRD_spline(
            400,
            redshifts.min() - 1.01,
            redshifts.max() + 1.01,
            ap_c["ALPHA_STAR"],
            ap_c["ALPHA_STAR_MINI"],
            ap_c["F_STAR10"],
            ap_c["F_STAR7_MINI"],
            ap_c["M_TURN"],
            flag_options.USE_MINI_HALOS,
        )

    sfrd = np.vectorize(lib.EvaluateSFRD)(redshifts, mlim_fstar_acg)
    if flag_options.USE_MINI_HALOS:
        sfrd_mini = np.vectorize(lib.EvaluateSFRD_MINI)(
            redshifts[:, None],
            log10mturnovers[None, :],
            mlim_fstar_mcg,
        )
    return sfrd, sfrd_mini


def evaluate_Nion_z(
    user_params: UserParams,
    cosmo_params: CosmoParams,
    astro_params: AstroParams,
    flag_options: FlagOptions,
    M_min: float,
    M_max: float,
    redshifts: Sequence[float],
    log10mturnovers: Sequence[float],
    return_integral: bool = False,
):
    """Evaluates the global ionising emissivity expected at a range of redshifts."""
    lib.Broadcast_struct_global_all(
        user_params.cstruct,
        cosmo_params.cstruct,
        astro_params.cstruct,
        flag_options.cstruct,
    )
    lib.init_ps()

    if user_params.USE_INTERPOLATION_TABLES:
        lib.initialiseSigmaMInterpTable(M_min, M_max)
    if (
        user_params.INTEGRATION_METHOD_ATOMIC == "GAUSS-LEGENDRE"
        or user_params.INTEGRATION_METHOD_MINI == "GAUSS-LEGENDRE"
    ):
        lib.initialise_GL(np.log(M_min), np.log(M_max))

    ap_c = astro_params.cdict

    mlim_fstar_acg = (
        1e10 * ap_c["F_STAR10"] ** (-1.0 / ap_c["ALPHA_STAR"])
        if ap_c["ALPHA_STAR"]
        else 0.0
    )
    mlim_fstar_mcg = (
        1e7 * ap_c["F_STAR7_MINI"] ** (-1.0 / ap_c["ALPHA_STAR_MINI"])
        if ap_c["ALPHA_STAR_MINI"]
        else 0.0
    )
    mlim_fesc_acg = (
        1e10 * ap_c["F_ESC10"] ** (-1.0 / ap_c["ALPHA_ESC"])
        if ap_c["ALPHA_ESC"]
        else 0.0
    )
    mlim_fesc_mcg = (
        1e7 * ap_c["F_ESC7_MINI"] ** (-1.0 / ap_c["ALPHA_ESC"])
        if ap_c["ALPHA_ESC"]
        else 0.0
    )

    sfrd_mini = 0.0
    mcrit_atom = (
        np.vectorize(lib.atomic_cooling_threshold)(redshifts)
        if flag_options.USE_MINI_HALOS
        else ap_c["M_TURN"]
    )
    # Unfortunately we have to do this until we sort out the USE_INTERPOLATION_TABLES flag
    # Since these integrals take forever if the flag is false
    if return_integral:
        sfrd = np.vectorize(lib.Nion_General)(
            redshifts,
            np.log(M_min),
            np.log(M_max),
            mcrit_atom,
            ap_c["ALPHA_STAR"],
            ap_c["ALPHA_ESC"],
            ap_c["F_STAR10"],
            ap_c["F_ESC10"],
            mlim_fstar_acg,
            mlim_fesc_acg,
        )
        if flag_options.USE_MINI_HALOS:
            sfrd_mini = np.vectorize(lib.Nion_General_MINI)(
                redshifts[:, None],
                np.log(M_min),
                np.log(M_max),
                10 ** log10mturnovers[None, :],
                mcrit_atom[:, None],
                ap_c["ALPHA_STAR_MINI"],
                ap_c["ALPHA_ESC"],
                ap_c["F_STAR7_MINI"],
                ap_c["F_ESC7_MINI"],
                mlim_fstar_mcg,
                mlim_fesc_mcg,
            )
        return sfrd, sfrd_mini

    if user_params.USE_INTERPOLATION_TABLES:
        lib.initialise_Nion_Ts_spline(
            400,
            redshifts.min() - 1.01,
            redshifts.max() + 1.01,
            ap_c["ALPHA_STAR"],
            ap_c["ALPHA_STAR_MINI"],
            ap_c["ALPHA_ESC"],
            ap_c["F_STAR10"],
            ap_c["F_ESC10"],
            ap_c["F_STAR7_MINI"],
            ap_c["F_ESC7_MINI"],
            ap_c["M_TURN"],
            flag_options.USE_MINI_HALOS,
        )

    nion = np.vectorize(lib.EvaluateNionTs)(
        redshifts,
        mlim_fstar_acg,
        mlim_fesc_acg,
    )
    nion_mini = np.vectorize(lib.EvaluateNionTs_MINI)(
        redshifts[:, None],
        log10mturnovers[None, :],
        mlim_fstar_mcg,
        mlim_fesc_mcg,
    )
    return nion, nion_mini


def evaluate_SFRD_cond(
    *,
    user_params: UserParams,
    cosmo_params: CosmoParams,
    astro_params: AstroParams,
    flag_options: FlagOptions,
    M_min: float,
    M_max: float,
    redshift: float,
    cond_mass: float,
    densities: Sequence[float],
    l10mturns: Sequence[float],
    return_integral: bool = False,
):
    """Evaluates the conditional star formation rate density expected at a range of densities."""
    lib.Broadcast_struct_global_all(
        user_params.cstruct,
        cosmo_params.cstruct,
        astro_params.cstruct,
        flag_options.cstruct,
    )

    lib.init_ps()
    if user_params.USE_INTERPOLATION_TABLES:
        lib.initialiseSigmaMInterpTable(M_min, M_max)

    if (
        user_params.INTEGRATION_METHOD_ATOMIC == "GAUSS-LEGENDRE"
        or user_params.INTEGRATION_METHOD_MINI == "GAUSS-LEGENDRE"
    ):
        lib.initialise_GL(np.log(M_min), np.log(M_max))

    ap_c = astro_params.cdict

    sigma_cond = lib.EvaluateSigma(np.log(cond_mass))
    mcrit_atom = (
        lib.atomic_cooling_threshold(redshift)
        if flag_options.USE_MINI_HALOS
        else ap_c["M_TURN"]
    )

    mlim_fstar_acg = (
        1e10 * ap_c["F_STAR10"] ** (-1.0 / ap_c["ALPHA_STAR"])
        if ap_c["ALPHA_STAR"]
        else 0.0
    )
    mlim_fstar_mcg = (
        1e7 * ap_c["F_STAR7_MINI"] ** (-1.0 / ap_c["ALPHA_STAR_MINI"])
        if ap_c["ALPHA_STAR_MINI"]
        else 0.0
    )

    growthf = lib.dicke(redshift)
    if not flag_options.USE_MINI_HALOS:
        SFRD_mcg = np.zeros((densities.size, l10mturns.size))
    # Unfortunately we have to do this until we sort out the USE_INTERPOLATION_TABLES flag
    # Since these integrals take forever if the flag is false
    if return_integral:
        SFRD_acg = np.vectorize(lib.Nion_ConditionalM)(
            growthf,
            np.log(M_min),
            np.log(M_max),
            cond_mass,
            sigma_cond,
            densities,
            mcrit_atom,
            ap_c["ALPHA_STAR"],
            0.0,
            ap_c["F_STAR10"],
            1.0,
            mlim_fstar_acg,
            0.0,
            user_params.cdict["INTEGRATION_METHOD_ATOMIC"],
        )
        if flag_options.USE_MINI_HALOS:
            SFRD_mcg = np.vectorize(lib.Nion_ConditionalM_MINI)(
                growthf,
                np.log(M_min),
                np.log(M_max),
                cond_mass,
                sigma_cond,
                densities[:, None],
                10 ** l10mturns[None, :],
                mcrit_atom,
                ap_c["ALPHA_STAR_MINI"],
                0.0,
                ap_c["F_STAR7_MINI"],
                1.0,
                mlim_fstar_mcg,
                0.0,
                user_params.cdict["INTEGRATION_METHOD_MINI"],
            )
        return SFRD_acg, SFRD_mcg

    if user_params.USE_INTERPOLATION_TABLES:
        lib.initialise_SFRD_Conditional_table(
            densities.min() - 0.01,
            densities.max() + 0.01,
            growthf,
            mcrit_atom,
            M_min,
            M_max,
            cond_mass,
            ap_c["ALPHA_STAR"],
            ap_c["ALPHA_STAR_MINI"],
            ap_c["F_STAR10"],
            ap_c["F_STAR7_MINI"],
            user_params.cdict["INTEGRATION_METHOD_ATOMIC"],
            user_params.cdict["INTEGRATION_METHOD_MINI"],
            flag_options.USE_MINI_HALOS,
        )

    SFRD_acg = np.vectorize(lib.EvaluateSFRD_Conditional)(
        densities,
        growthf,
        M_min,
        M_max,
        cond_mass,
        sigma_cond,
        mcrit_atom,
        mlim_fstar_acg,
    )
    if flag_options.USE_MINI_HALOS:
        SFRD_mcg = np.vectorize(lib.EvaluateSFRD_Conditional_MINI)(
            densities[:, None],
            l10mturns[None, :],
            growthf,
            M_min,
            M_max,
            cond_mass,
            sigma_cond,
            mcrit_atom,
            mlim_fstar_mcg,
        )
    return SFRD_acg, SFRD_mcg


def evaluate_Nion_cond(
    *,
    user_params: UserParams,
    cosmo_params: CosmoParams,
    astro_params: AstroParams,
    flag_options: FlagOptions,
    M_min: float,
    M_max: float,
    redshift: float,
    cond_mass: float,
    densities: Sequence[float],
    l10mturns: Sequence[float],
    return_integral: bool = False,
):
    """Evaluates the conditional ionising emissivity expected at a range of densities."""
    lib.Broadcast_struct_global_all(
        user_params.cstruct,
        cosmo_params.cstruct,
        astro_params.cstruct,
        flag_options.cstruct,
    )

    lib.init_ps()
    if user_params.USE_INTERPOLATION_TABLES:
        lib.initialiseSigmaMInterpTable(M_min, M_max)

    if (
        user_params.INTEGRATION_METHOD_ATOMIC == "GAUSS-LEGENDRE"
        or user_params.INTEGRATION_METHOD_MINI == "GAUSS-LEGENDRE"
    ):
        lib.initialise_GL(np.log(M_min), np.log(M_max))

    ap_c = astro_params.cdict

    sigma_cond = lib.EvaluateSigma(np.log(cond_mass))
    mcrit_atom = (
        lib.atomic_cooling_threshold(redshift)
        if flag_options.USE_MINI_HALOS
        else ap_c["M_TURN"]
    )

    mlim_fstar_acg = (
        1e10 * ap_c["F_STAR10"] ** (-1.0 / ap_c["ALPHA_STAR"])
        if ap_c["ALPHA_STAR"]
        else 0.0
    )
    mlim_fstar_mcg = (
        1e7 * ap_c["F_STAR7_MINI"] ** (-1.0 / ap_c["ALPHA_STAR_MINI"])
        if ap_c["ALPHA_STAR_MINI"]
        else 0.0
    )
    mlim_fesc_acg = (
        1e10 * ap_c["F_ESC10"] ** (-1.0 / ap_c["ALPHA_ESC"])
        if ap_c["ALPHA_ESC"]
        else 0.0
    )
    mlim_fesc_mcg = (
        1e7 * ap_c["F_ESC7_MINI"] ** (-1.0 / ap_c["ALPHA_ESC"])
        if ap_c["ALPHA_ESC"]
        else 0.0
    )

    if not flag_options.USE_MINI_HALOS:
        Nion_mcg = np.zeros((densities.size, l10mturns.size))
    growthf = lib.dicke(redshift)
    # Unfortunately we have to do this until we sort out the USE_INTERPOLATION_TABLES flag
    # Since these integrals take forever if the flag is false
    if return_integral:
        Nion_acg = np.vectorize(lib.Nion_ConditionalM)(
            growthf,
            np.log(M_min),
            np.log(M_max),
            cond_mass,
            sigma_cond,
            densities[:, None] if flag_options.USE_MINI_HALOS else densities,
            10 ** l10mturns[None,
                            :] if flag_options.USE_MINI_HALOS else mcrit_atom,
            ap_c["ALPHA_STAR"],
            ap_c["ALPHA_ESC"],
            ap_c["F_STAR10"],
            ap_c["F_ESC10"],
            mlim_fstar_acg,
            mlim_fesc_acg,
            user_params.cdict["INTEGRATION_METHOD_ATOMIC"],
        )
        if flag_options.USE_MINI_HALOS:
            Nion_mcg = np.vectorize(lib.Nion_ConditionalM_MINI)(
                growthf,
                np.log(M_min),
                np.log(M_max),
                cond_mass,
                sigma_cond,
                densities[:, None],
                10 ** l10mturns[None, :],
                mcrit_atom,
                ap_c["ALPHA_STAR_MINI"],
                ap_c["ALPHA_ESC"],
                ap_c["F_STAR7_MINI"],
                ap_c["F_ESC7_MINI"],
                mlim_fstar_mcg,
                mlim_fesc_mcg,
                user_params.cdict["INTEGRATION_METHOD_MINI"],
            )
        return Nion_acg, Nion_mcg

    # TODO: this function needs cleanup
    if user_params.USE_INTERPOLATION_TABLES:
        lib.initialise_Nion_Conditional_spline(
            redshift,
            mcrit_atom,
            densities.min() - 0.01,
            densities.max() + 0.01,
            M_min,
            M_max,
            cond_mass,
            l10mturns.min() * 0.99,
            l10mturns.max() * 1.01,
            l10mturns.min() * 0.99,
            l10mturns.max() * 1.01,
            ap_c["ALPHA_STAR"],
            ap_c["ALPHA_STAR_MINI"],
            ap_c["ALPHA_ESC"],
            ap_c["F_STAR10"],
            ap_c["F_ESC10"],
            mlim_fstar_acg,
            mlim_fesc_acg,
            ap_c["F_STAR7_MINI"],
            ap_c["F_ESC7_MINI"],
            mlim_fstar_mcg,
            mlim_fesc_mcg,
            user_params.cdict["INTEGRATION_METHOD_ATOMIC"],
            user_params.cdict["INTEGRATION_METHOD_MINI"],
            flag_options.USE_MINI_HALOS,
            False,
        )

    Nion_acg = np.vectorize(lib.EvaluateNion_Conditional)(
        densities[:, None] if flag_options.USE_MINI_HALOS else densities,
        l10mturns[None, :] if flag_options.USE_MINI_HALOS else mcrit_atom,
        growthf,
        M_min,
        M_max,
        cond_mass,
        sigma_cond,
        mlim_fstar_acg,
        mlim_fesc_acg,
        False,
    )
    if flag_options.USE_MINI_HALOS:
        Nion_mcg = np.vectorize(lib.EvaluateNion_Conditional_MINI)(
            densities[:, None],
            l10mturns[None, :],
            growthf,
            M_min,
            M_max,
            cond_mass,
            sigma_cond,
            mcrit_atom,
            mlim_fstar_mcg,
            mlim_fesc_mcg,
            False,
        )
    return Nion_acg, Nion_mcg


def halo_sample_test(
    *,
    user_params: UserParams,
    cosmo_params: CosmoParams,
    astro_params: AstroParams,
    flag_options: FlagOptions,
    redshift: float,
    from_cat: bool,
    cond_array,
    seed: int = 12345,
):
    """Constructs a halo sample given a descendant catalogue and redshifts."""
    # fix all zero for coords
    n_cond = cond_array.size
    crd_in = np.zeros(3 * n_cond).astype("i4")
    # HALO MASS CONDITIONS WITH FIXED z-step
    cond_array = cond_array.astype("f4")

    z_prev = -1
    if from_cat:
        z_prev = (1 + redshift) / global_params.ZPRIME_STEP_FACTOR - 1

    # about 500MB total 2e7 * 4 (float) * 4 (mass + 3crd)
    buffer_size = int(3e7)
    nhalo_out = np.zeros(1).astype("i4")
    N_out = np.zeros(n_cond).astype("i4")
    M_out = np.zeros(n_cond).astype("f8")
    exp_M = np.zeros(n_cond).astype("f8")
    exp_N = np.zeros(n_cond).astype("f8")
    halomass_out = np.zeros(buffer_size).astype("f4")
    halocrd_out = np.zeros(int(3 * buffer_size)).astype("i4")

    lib.single_test_sample(
        user_params.cstruct,
        cosmo_params.cstruct,
        astro_params.cstruct,
        flag_options.cstruct,
        12345,
        n_cond,
        # WIP: CFFI Refactor
        # ffi.cast("float *", cond_array.ctypes.data),
        # ffi.cast("int *", crd_in.ctypes.data),
        cond_array.ctypes.data,
        crd_in.ctypes.data,
        redshift,
        z_prev,
        # WIP: CFFI Refactor
        # ffi.cast("int *", nhalo_out.ctypes.data),
        # ffi.cast("int *", N_out.ctypes.data),
        # ffi.cast("double *", exp_N.ctypes.data),
        # ffi.cast("double *", M_out.ctypes.data),
        # ffi.cast("double *", exp_M.ctypes.data),
        # ffi.cast("float *", halomass_out.ctypes.data),
        # ffi.cast("int *", halocrd_out.ctypes.data),
        nhalo_out,
        N_out,
        exp_N,
        M_out,
        exp_M,
        halomass_out,
        halocrd_out,
    )

    return {
        "n_halo_total": nhalo_out[0],
        "halo_masses": halomass_out,
        # 'halo_coords': halocrd_out,
        "n_progenitors": N_out,
        "progenitor_mass": M_out,
        "expected_progenitors": exp_N,
        "expected_progenitor_mass": exp_M,
    }


# TODO: make this able to take a proper HaloField/PerturbHaloField
#    with corresponding Ts/ion/Ic fields for feedback
def convert_halo_properties(
    *,
    redshift: float,
    astro_params: AstroParams,
    flag_options: FlagOptions,
    ics: InitialConditions,
    halo_masses: Sequence[float],
    halo_rng: Sequence[float],
):
    """
    Converts a halo catalogue's mass and RNG fields to halo properties.

    Assumes no feedback (Lyman-Werner, reionization).

    Returns a dict of 12 properties per halo:
        halo mass
        stellar mass (ACG)
        star formation rate (ACG)
        xray luminosity (combined)
        ionising emissivity (combined)
        escape-fraction weighted SFR (combined)
        stellar mass (MCG)
        star formation rate (MCG)
        ACG turnover mass
        MCG turnover mass
        Reionization turnover mass
        Metallicity
    """
    lib.Broadcast_struct_global_all(
        ics.user_params.cstruct,
        ics.cosmo_params.cstruct,
        astro_params.cstruct,
        flag_options.cstruct,
    )
    inputs = InputParameters.from_output_structs(
        [
            ics,
        ],
        astro_params=astro_params,
        flag_options=flag_options,
        redshift=redshift,
    )

    # HACK: Make the fake halo list
    fake_pthalos = PerturbHaloField(
        inputs=inputs,
        buffer_size=halo_masses.size,
    )
    fake_pthalos()  # initialise memory
    fake_pthalos.halo_masses = halo_masses.astype("f4")
    fake_pthalos.halo_corods = np.zeros(halo_masses.size * 3).astype("i4")
    fake_pthalos.star_rng = halo_rng.astype("f4")
    fake_pthalos.sfr_rng = halo_rng.astype("f4")
    fake_pthalos.xray_rng = halo_rng.astype("f4")
    fake_pthalos.n_halos = halo_masses.size

    # TODO: ask Steven if this is a memory leak
    fake_pthalos._init_cstruct()

    # single element zero array to act as the grids (vcb, J_21_LW, z_reion, Gamma12)
    # WIP: CFFI Refactor
    # zero_array = ffi.cast("float *", np.zeros(1).ctypes.data)
    zero_array = np.zeros(1)

    out_buffer = np.zeros(12 * halo_masses.size).astype("f4")
    lib.test_halo_props(
        redshift,
        ics.user_params.cstruct,
        ics.cosmo_params.cstruct,
        astro_params.cstruct,
        flag_options.cstruct,
        zero_array,  # ICs vcb
        zero_array,  # J_21_LW
        zero_array,  # z_re
        zero_array,  # Gamma12
        fake_pthalos(),
        # WIP: CFFI Refactor
        # ffi.cast("float *", out_buffer.ctypes.data),
        out_buffer,
    )

    out_buffer = out_buffer.reshape(fake_pthalos.n_halos, 12)

    return {
        "halo_mass": out_buffer[:, 0],
        "halo_stars": out_buffer[:, 1],
        "halo_sfr": out_buffer[:, 2],
        "halo_xray": out_buffer[:, 3],
        "n_ion": out_buffer[:, 4],
        "halo_wsfr": out_buffer[:, 5],
        "halo_stars_mini": out_buffer[:, 6],
        "halo_sfr_mini": out_buffer[:, 7],
        "mturn_a": out_buffer[:, 8],
        "mturn_m": out_buffer[:, 9],
        "mturn_r": out_buffer[:, 10],
        "metallicity": out_buffer[:, 11],
    }
