"""Low-level python wrappers of C functions."""

import logging
from collections.abc import Callable, Sequence
from functools import cache
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d

from .._cfg import config
from ..c_21cmfast import ffi, lib
from ._utils import _process_exitcode
from .inputs import AstroParams, CosmoParams, FlagOptions, InputParameters, UserParams
from .outputs import InitialConditions, PerturbHaloField

logger = logging.getLogger(__name__)


def broadcast_params(func: Callable) -> Callable:
    """Decorator to broadcast the parameters to the C library before calling the function."""

    def wrapper(*args, **kwargs):
        inputs = kwargs.get("inputs")
        if inputs.astro_params:
            lib.Broadcast_struct_global_all(
                inputs.user_params.cstruct,
                inputs.cosmo_params.cstruct,
                inputs.astro_params.cstruct,
                inputs.flag_options.cstruct,
            )
        else:
            lib.Broadcast_struct_global_noastro(
                inputs.user_params.cstruct,
                inputs.cosmo_params.cstruct,
            )
        return func(*args, **kwargs)

    return wrapper


@broadcast_params
def init_backend_ps(func: Callable) -> Callable:
    """Decorator to initialise the backend PS before calling the function."""

    def wrapper(*args, **kwargs):
        lib.init_ps()
        return func(*args, **kwargs)

    return wrapper


@init_backend_ps
def init_sigma_table(func: Callable) -> Callable:
    """Decorator to initialise the the sigma interpolation table before calling the function."""

    def wrapper(*args, **kwargs):
        sigma_min_mass = kwargs.get("M_min", 1e5)
        sigma_max_mass = kwargs.get("M_max", 1e16)
        if (
            kwargs.get("inputs").user_params.USE_INTERPOLATION_TABLES
            != "no-interpolation"
        ):
            lib.initialiseSigmaMInterpTable(sigma_min_mass, sigma_max_mass)
        return func(*args, **kwargs)

    return wrapper


@init_sigma_table
def init_gl(func: Callable) -> Callable:
    """Decorator to initialise the Gauss-Legendre if required before calling the function."""

    def wrapper(*args, **kwargs):
        if "GAUSS-LEGENDRE" in (
            kwargs.get("inputs").user_params.INTEGRATION_METHOD_ATOMIC,
            kwargs.get("inputs").user_params.INTEGRATION_METHOD_MINI,
        ):
            # no defualt since GL mass limits are strict
            lib.initialise_GL(np.log(kwargs.get("M_min")), np.log(kwargs.get("M_max")))
        return func(*args, **kwargs)

    return wrapper


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
    hbuffer_size = int((hbuffer_size + 1) * config["HALO_CATALOG_MEM_FACTOR"])

    # set a minimum in case of fluctuation at high z
    return int(max(hbuffer_size, min_size))


def compute_tau(
    *,
    redshifts: Sequence[float],
    global_xHI: Sequence[float],
    inputs: InputParameters,
    z_re_HeII: float = 3.0,
) -> float:
    """Compute the optical depth to reionization under the given model.

    Parameters
    ----------
    redshifts : array-like
        Redshifts defining an evolution of the neutral fraction.
    global_xHI : array-like
        The mean neutral fraction at `redshifts`.
    inputs : :class:`~InputParameters`
        Defines the input parameters of the run
    z_re_HeII : float, optional
        The redshift at which helium reionization occurs.

    Returns
    -------
    tau : float
        The optical depth to reionization

    Raises
    ------
    ValueError :
        If `redshifts` and `global_xHI` have inconsistent length or if redshifts are not
        in ascending order.
    """
    if len(redshifts) != len(global_xHI):
        raise ValueError("redshifts and global_xHI must have same length")

    if not np.all(np.diff(redshifts) > 0):
        raise ValueError("redshifts and global_xHI must be in ascending order")

    # Convert the data to the right type
    redshifts = np.array(redshifts, dtype="float32")
    global_xHI = np.array(global_xHI, dtype="float32")

    z = ffi.cast("float *", ffi.from_buffer(redshifts))
    xHI = ffi.cast("float *", ffi.from_buffer(global_xHI))

    # Run the C code
    return lib.ComputeTau(
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        len(redshifts),
        z,
        xHI,
        z_re_HeII,
    )


def compute_luminosity_function(
    *,
    redshifts: Sequence[float],
    inputs: InputParameters,
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
    user_params = inputs.user_params
    cosmo_params = inputs.cosmo_params
    flag_options = inputs.flag_options
    astro_params = inputs.astro_params

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
            np.zeros(len(redshifts), dtype=np.float32) + 10**astro_params.M_TURN
        )
        component = "acg"

    lfunc = np.zeros(len(redshifts) * nbins)
    Muvfunc = np.zeros(len(redshifts) * nbins)
    Mhfunc = np.zeros(len(redshifts) * nbins)

    lfunc.shape = (len(redshifts), nbins)
    Muvfunc.shape = (len(redshifts), nbins)
    Mhfunc.shape = (len(redshifts), nbins)

    c_Muvfunc = ffi.cast("double *", ffi.from_buffer(Muvfunc))
    c_Mhfunc = ffi.cast("double *", ffi.from_buffer(Mhfunc))
    c_lfunc = ffi.cast("double *", ffi.from_buffer(lfunc))

    lfunc_MINI = np.zeros(len(redshifts) * nbins)
    Muvfunc_MINI = np.zeros(len(redshifts) * nbins)
    Mhfunc_MINI = np.zeros(len(redshifts) * nbins)

    lfunc_MINI.shape = (len(redshifts), nbins)
    Muvfunc_MINI.shape = (len(redshifts), nbins)
    Mhfunc_MINI.shape = (len(redshifts), nbins)

    c_Muvfunc_MINI = ffi.cast("double *", ffi.from_buffer(Muvfunc_MINI))
    c_Mhfunc_MINI = ffi.cast("double *", ffi.from_buffer(Mhfunc_MINI))
    c_lfunc_MINI = ffi.cast("double *", ffi.from_buffer(lfunc_MINI))

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
            ffi.cast("float *", ffi.from_buffer(redshifts)),
            ffi.cast("float *", ffi.from_buffer(mturnovers)),
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
            ffi.cast("float *", ffi.from_buffer(redshifts)),
            ffi.cast("float *", ffi.from_buffer(mturnovers_mini)),
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


@init_sigma_table
def evaluate_sigma(
    inputs,
    masses: Sequence[float],
):
    """
    Evaluates the variance of a mass scale.

    Uses the 21cmfast backend
    """
    sigma = np.vectorize(lib.EvaluateSigma)(np.log(masses))
    dsigmasq = np.vectorize(lib.EvaluatedSigmasqdm)(np.log(masses))

    return sigma, dsigmasq


@init_backend_ps
def get_growth_factor(
    inputs: InputParameters,
    redshift: float,
):
    """Gets the growth factor at a given redshift."""
    return lib.dicke(redshift)


def get_delta_crit(inputs, mass, redshift):
    """Gets the critical collapse density given a mass, redshift and parameters."""
    sigma, _ = evaluate_sigma(inputs, mass)
    # evaluate_sigma already broadcasts the paramters so we don't need to repeat
    growth = get_growth_factor(inputs, redshift)
    return get_delta_crit_nu(inputs.user_params, sigma, growth)


def get_delta_crit_nu(user_params, sigma, growth):
    """Uses the nu paramters (sigma and growth factor) to get critical density."""
    # None of the parameter structs are used in this function so we don't need a broadcast
    return np.vectorize(lib.get_delta_crit)(user_params.cdict["HMF"], sigma, growth)


@init_sigma_table
def initialise_dNdM_tables(
    inputs: InputParameters,
    cond_min: float,
    cond_max: float,
    M_min: float,
    M_max: float,
    growth_out: float,
    cond_param: float,
    from_catalog: bool,
):
    """Initialises the dNdM tables for the conditional mass function.

    If from_catalog is True, cond_param is the descendant growth factor and cond_array are halo masses.
    If from_catalog is False, cond_param is the natural log of the cell mass and cond_array are deltas.
    """
    lib.initialise_dNdM_tables(
        cond_min,
        cond_max,
        np.log(M_min),
        np.log(M_max),
        growth_out,
        cond_param,
        from_catalog,
    )


@init_sigma_table
def get_condition_integrals(
    cond_array: Sequence[float],
    growth_out: float,
    M_min: float,
    M_max: float,
    cond_mass: float,
    sigma_cond: float,
    delta: float,
):
    """Gets the expected number and mass of halos given a condition.

    If USE_INTERPOLATION_TABLES is set to 'hmf-interpolation': Will crash if the table
    has not been initialised, only `cond_array` is used,
    and the rest of the arguments are taken from when the table was initialised.
    """
    n_halo = (
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
    m_coll = (
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

    return n_halo, m_coll


@init_sigma_table
def initialise_dNdM_inverse_table(
    cond_min: float,
    cond_max: float,
    M_min: float,
    growth_out: float,
    cond_param: float,
    from_catalog: bool,
):
    """Initialises the inverse dNdM tables for the conditional mass function."""
    lib.initialise_dNdM_inverse_table(
        cond_min,
        cond_max,
        np.log(M_min),
        growth_out,
        cond_param,
        from_catalog,
    )


def evaluate_inverse_table(
    cond_array: Sequence[float],
    probabilities: Sequence[float],
    cond_mass: float,
):
    """Evaluates the inverse cumulative halo mass function.

    Used to verify sampling tables.
    """
    masses = (
        np.vectorize(lib.EvaluateNhaloInv)(
            cond_array,
            probabilities,
        )
        * cond_mass
    )
    return masses


# TODO: I don't think these needs to be separated into initialisation
#   and evaluation with a combined function in integrals.py e.g the halos
#   due to its simplicity, would appreciate some input here
# Considerations:
# - limiting these functions to thin wrappers is desired
# - It would be nice to keep lib calls segregated to this file
# - keeping the initialisation and evaluation together makes mistakes harder
# - Avoiding re-initialisation by calling decoratred functions multiple times is a good idea
@init_sigma_table
def evaluate_FgtrM_cond(
    inputs: InputParameters,
    densities: Sequence[float],
    redshift: float,
    sigma_min: float,
    sigma_cond: float,
    growth_out: float,
):
    """Gets the collapsed fraction from the backend, given a density and condition sigma."""
    if inputs.user_params.USE_INTERPOLATION_TABLES == "hmf-interpolation":
        lib.initialise_FgtrM_delta_tables(
            densities.min(),
            densities.max(),
            redshift,
            lib.dicke(redshift),
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


@init_gl
def evaluate_SFRD_z(
    *,
    inputs: InputParameters,
    M_min: float,
    M_max: float,
    redshifts: Sequence[float],
    log10mturnovers: Sequence[float],
    pl_mass_lims: dict[str, float],
    nbin: int = 400,
):
    """Evaluates the global star formation rate density expected at a range of redshifts."""
    ap_c = inputs.astro_params.cdict
    if inputs.user_params.USE_INTERPOLATION_TABLES == "hmf-interpolation":
        lib.initialise_SFRD_spline(
            nbin,
            redshifts.min(),
            redshifts.max() + 1.01,
            ap_c["ALPHA_STAR"],
            ap_c["ALPHA_STAR_MINI"],
            ap_c["F_STAR10"],
            ap_c["F_STAR7_MINI"],
        )

    sfrd = np.vectorize(lib.EvaluateSFRD)(redshifts, pl_mass_lims["fstar_acg"])
    sfrd_mini = np.zeros((redshifts.size, log10mturnovers.size))
    if inputs.flag_options.USE_MINI_HALOS:
        sfrd_mini = np.vectorize(lib.EvaluateSFRD_MINI)(
            redshifts[:, None],
            log10mturnovers[None, :],
            pl_mass_lims["fstar_mcg"],
        )
    return sfrd, sfrd_mini


@init_gl
def evaluate_Nion_z(
    inputs: InputParameters,
    M_min: float,
    M_max: float,
    redshifts: Sequence[float],
    log10mturnovers: Sequence[float],
    pl_mass_lims: dict[str, float],
    nbins: int = 400,
):
    """Evaluates the global ionising emissivity expected at a range of redshifts."""
    ap_c = inputs.astro_params.cdict
    if inputs.user_params.USE_INTERPOLATION_TABLES == "hmf-interpolation":
        lib.initialise_Nion_Ts_spline(
            nbins,
            redshifts.min(),
            redshifts.max() + 1.01,
            ap_c["ALPHA_STAR"],
            ap_c["ALPHA_STAR_MINI"],
            ap_c["ALPHA_ESC"],
            ap_c["F_STAR10"],
            ap_c["F_ESC10"],
            ap_c["F_STAR7_MINI"],
            ap_c["F_ESC7_MINI"],
        )

    nion = np.vectorize(lib.EvaluateNionTs)(
        redshifts,
        pl_mass_lims["fstar_acg"],
        pl_mass_lims["fesc_acg"],
    )
    nion_mini = np.zeros((redshifts.size, log10mturnovers.size))
    if inputs.flag_options.USE_MINI_HALOS:
        nion_mini = np.vectorize(lib.EvaluateNionTs_MINI)(
            redshifts[:, None],
            log10mturnovers[None, :],
            pl_mass_lims["fstar_mcg"],
            pl_mass_lims["fesc_mcg"],
        )
    return nion, nion_mini


@init_gl
def evaluate_SFRD_cond(
    *,
    inputs: InputParameters,
    M_min: float,
    M_max: float,
    redshift: float,
    cond_mass: float,
    densities: Sequence[float],
    l10mturns: Sequence[float],
    mturn_acg: float,
    pl_mass_lims: dict[str, float],
):
    """Evaluates the conditional star formation rate density expected at a range of densities."""
    ap_c = inputs.astro_params.cdict

    # NOTE: I'm still using the lib functions to avoid double-initialisation
    sigma_cond = lib.EvaluateSigma(np.log(cond_mass))
    acg_thresh = lib.atomic_cooling_threshold(redshift)

    growthf = lib.dicke(redshift)
    if inputs.user_params.USE_INTERPOLATION_TABLES:
        lib.initialise_SFRD_Conditional_table(
            redshift,
            densities.min(),
            densities.max() + 0.01,
            M_min,
            M_max,
            cond_mass,
            ap_c["ALPHA_STAR"],
            ap_c["ALPHA_STAR_MINI"],
            ap_c["F_STAR10"],
            ap_c["F_STAR7_MINI"],
        )

    SFRD_acg = np.vectorize(lib.EvaluateSFRD_Conditional)(
        densities,
        growthf,
        M_min,
        M_max,
        cond_mass,
        sigma_cond,
        mturn_acg,
        pl_mass_lims["fstar_acg"],
    )
    if inputs.flag_options.USE_MINI_HALOS:
        SFRD_mcg = np.vectorize(lib.EvaluateSFRD_Conditional_MINI)(
            densities[:, None],
            l10mturns[None, :],
            growthf,
            M_min,
            M_max,
            cond_mass,
            sigma_cond,
            acg_thresh,
            pl_mass_lims["fstar_mcg"],
        )
    return SFRD_acg, SFRD_mcg


@init_gl
def evaluate_Nion_cond(
    *,
    inputs: InputParameters,
    M_min: float,
    M_max: float,
    redshift: float,
    cond_mass: float,
    densities: Sequence[float],
    l10mturns: Sequence[float],
    pl_mass_lims: dict[str, float],
):
    """Evaluates the conditional ionising emissivity expected at a range of densities."""
    ap_c = inputs.astro_params.cdict

    sigma_cond = lib.EvaluateSigma(np.log(cond_mass))
    acg_thresh = lib.atomic_cooling_threshold(redshift)
    growthf = lib.dicke(redshift)

    if inputs.user_params.USE_INTERPOLATION_TABLES:
        lib.initialise_Nion_Conditional_spline(
            redshift,
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
            ap_c["F_STAR7_MINI"],
            ap_c["F_ESC7_MINI"],
            False,
        )

    Nion_acg = np.vectorize(lib.EvaluateNion_Conditional)(
        densities[:, None] if inputs.flag_options.USE_MINI_HALOS else densities,
        l10mturns[None, :] if inputs.flag_options.USE_MINI_HALOS else ap_c["M_TURN"],
        growthf,
        M_min,
        M_max,
        cond_mass,
        sigma_cond,
        pl_mass_lims["fstar_acg"],
        pl_mass_lims["fesc_acg"],
        False,
    )

    Nion_mcg = np.zeros((densities.size, l10mturns.size))
    if inputs.flag_options.USE_MINI_HALOS:
        Nion_mcg = np.vectorize(lib.EvaluateNion_Conditional_MINI)(
            densities[:, None],
            l10mturns[None, :],
            growthf,
            M_min,
            M_max,
            cond_mass,
            sigma_cond,
            acg_thresh,
            pl_mass_lims["fstar_mcg"],
            pl_mass_lims["fesc_mcg"],
            False,
        )
    return Nion_acg, Nion_mcg


@init_gl
def evaluate_Xray_cond(
    *,
    inputs: InputParameters,
    M_min: float,
    M_max: float,
    redshift: float,
    cond_mass: float,
    densities: Sequence[float],
    l10mturns: Sequence[float],
    pl_mass_lims: dict[str, float],
):
    """Evaluates the conditional ionising emissivity expected at a range of densities."""
    ap_c = inputs.astro_params.cdict

    sigma_cond = lib.EvaluateSigma(np.log(cond_mass))
    acg_thresh = lib.atomic_cooling_threshold(redshift)
    growthf = lib.dicke(redshift)
    t_h = (1 / inputs.cosmo_params.cosmo.H(redshift)).to("s").value

    if inputs.user_params.USE_INTERPOLATION_TABLES:
        lib.initialise_Xray_Conditional_table(
            densities.min() - 0.01,
            densities.max() + 0.01,
            redshift,
            M_min,
            M_max,
            cond_mass,
            ap_c["ALPHA_STAR"],
            ap_c["ALPHA_STAR_MINI"],
            ap_c["F_STAR10"],
            ap_c["F_STAR7_MINI"],
            ap_c["L_X"],
            ap_c["L_X_MINI"],
            t_h,
            ap_c["t_STAR"],
        )

    Xray = np.vectorize(lib.EvaluateXray_Conditional)(
        densities[:, None] if inputs.flag_options.USE_MINI_HALOS else densities,
        l10mturns[None, :] if inputs.flag_options.USE_MINI_HALOS else ap_c["M_TURN"],
        redshift,
        growthf,
        M_min,
        M_max,
        cond_mass,
        sigma_cond,
        max(ap_c["M_TURN"], acg_thresh),
        t_h,
        pl_mass_lims["fstar_acg"],
        pl_mass_lims["fstar_mcg"],
    )
    return Xray


def halo_sample_test(
    *,
    inputs: InputParameters,
    redshift: float,
    from_cat: bool,
    cond_array,
    redshift_prev: float | None = None,
    seed: int = 12345,
    buffer_size: int = 3e7,
):
    """Constructs a halo sample given a descendant catalogue and redshifts."""
    # fix all zero for coords
    n_cond = cond_array.size
    crd_in = np.zeros(3 * n_cond).astype("i4")

    # HALO MASS CONDITIONS WITH FIXED z-step
    cond_array = cond_array.astype("f4")
    z_prev = -1 if redshift_prev is None else redshift_prev

    nhalo_out = np.zeros(1).astype("i4")
    N_out = np.zeros(n_cond).astype("i4")
    M_out = np.zeros(n_cond).astype("f8")
    exp_M = np.zeros(n_cond).astype("f8")
    exp_N = np.zeros(n_cond).astype("f8")
    halomass_out = np.zeros(buffer_size).astype("f4")
    halocrd_out = np.zeros(int(3 * buffer_size)).astype("i4")

    lib.single_test_sample(
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct,
        seed,
        n_cond,
        ffi.cast("float *", cond_array.ctypes.data),
        ffi.cast("int *", crd_in.ctypes.data),
        redshift,
        z_prev,
        ffi.cast("int *", nhalo_out.ctypes.data),
        ffi.cast("int *", N_out.ctypes.data),
        ffi.cast("double *", exp_N.ctypes.data),
        ffi.cast("double *", M_out.ctypes.data),
        ffi.cast("double *", exp_M.ctypes.data),
        ffi.cast("float *", halomass_out.ctypes.data),
        ffi.cast("int *", halocrd_out.ctypes.data),
    )

    return {
        "n_halo_total": nhalo_out[0],
        "halo_masses": halomass_out,
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
    inputs: InputParameters,
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
    zero_array = ffi.cast("float *", np.zeros(1).ctypes.data)

    out_buffer = np.zeros(12 * halo_masses.size).astype("f4")
    lib.test_halo_props(
        redshift,
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct,
        zero_array,  # ICs vcb
        zero_array,  # J_21_LW
        zero_array,  # z_re
        zero_array,  # Gamma12
        fake_pthalos(),
        ffi.cast("float *", out_buffer.ctypes.data),
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
