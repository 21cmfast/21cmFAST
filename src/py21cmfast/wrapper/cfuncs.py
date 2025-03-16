"""Low-level python wrappers of C functions."""

import logging
import warnings
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

# Ideally, backend functions that we access here should do all the broadcasting/initialisation themselves
# These decorators are for lower functions which are called directly in one or two lines, like delta_crit

# TODO: a lot of these assume input as numpy arrays via use of .shape, explicitly require this


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


def init_backend_ps(func: Callable) -> Callable:
    """Decorator to initialise the backend PS before calling the function."""

    @broadcast_params
    def wrapper(*args, **kwargs):
        lib.init_ps()
        return func(*args, **kwargs)

    return wrapper


def init_sigma_table(func: Callable) -> Callable:
    """Decorator to initialise the the sigma interpolation table before calling the function."""

    @init_backend_ps
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


def init_gl(func: Callable) -> Callable:
    """Decorator to initialise the Gauss-Legendre if required before calling the function."""

    @init_sigma_table
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
    *redshift: float,
    inputs: InputParameters,
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
    return lib.expected_nhalo(
        redshift, inputs.user_params.cstruct, inputs.cosmo_params.cstruct
    )


def get_halo_list_buffer_size(
    *redshift: float,
    inputs: InputParameters,
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
    hbuffer_size = get_expected_nhalo(redshift=redshift, inputs=inputs)
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
    inputs: :class:`~InputParameters`
        The input parameters defining the simulation run.
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


def evaluate_sigma(
    inputs,
    masses: Sequence[float],
):
    """
    Evaluates the variance of a mass scale.

    Uses the 21cmfast backend
    """
    sigma = np.zeros(len(masses), dtype="f8")
    dsigmasq = np.zeros(len(masses), dtype="f8")
    masses = np.array(masses, dtype="f8")

    lib.get_sigma(
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        len(masses),
        ffi.cast("double *", ffi.from_buffer(masses)),
        ffi.cast("double *", ffi.from_buffer(sigma)),
        ffi.cast("double *", ffi.from_buffer(dsigmasq)),
    )

    return sigma, dsigmasq


@init_backend_ps
def get_growth_factor(
    *,
    inputs: InputParameters,
    redshift: float,
):
    """Gets the growth factor at a given redshift."""
    return lib.dicke(redshift)


def get_condition_mass(inputs, R):
    """Convenience function for determining condition masses for backend routines."""
    rhocrit = (
        inputs.cosmo_params.cosmo.critical_density(0).to("M_sun Mpc-3").value
        * inputs.cosmo_params.OMm
    )
    if R == "cell":
        volume = (inputs.user_params.BOX_LEN / inputs.user_params.HII_DIM) ** 3
    else:
        volume = 4.0 / 3.0 * np.pi * R**3

    return volume * rhocrit


def get_delta_crit(inputs, mass, redshift):
    """Gets the critical collapse density given a mass, redshift and parameters."""
    sigma, _ = evaluate_sigma(
        inputs,
        [
            mass,
        ],
    )
    # evaluate_sigma already broadcasts the paramters so we don't need to repeat
    growth = get_growth_factor(inputs=inputs, redshift=redshift)
    return get_delta_crit_nu(inputs.user_params, sigma, growth)


def get_delta_crit_nu(user_params, sigma, growth):
    """Uses the nu paramters (sigma and growth factor) to get critical density."""
    # None of the parameter structs are used in this function so we don't need a broadcast
    return lib.get_delta_crit(user_params.cdict["HMF"], sigma, growth)


def evaluate_condition_integrals(
    inputs: InputParameters,
    cond_array: Sequence[float],
    redshift: float,
    redshift_prev: float | None = None,
):
    """Gets the expected number and mass of halos given a condition.

    If USE_INTERPOLATION_TABLES is set to 'hmf-interpolation': Will crash if the table
    has not been initialised, only `cond_array` is used,
    and the rest of the arguments are taken from when the table was initialised.
    """
    orig_shape = cond_array.shape
    cond_array = np.array(cond_array, dtype="f8").flatten()
    n_halo = np.zeros_like(cond_array)
    m_coll = np.zeros_like(cond_array)

    lib.get_condition_integrals(
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct,
        redshift,
        redshift_prev if redshift_prev is not None else -1,
        len(cond_array),
        ffi.cast("double *", ffi.from_buffer(cond_array)),
        ffi.cast("double *", ffi.from_buffer(n_halo)),
        ffi.cast("double *", ffi.from_buffer(m_coll)),
    )

    return np.reshape(n_halo, orig_shape), np.reshape(m_coll, orig_shape)


def integrate_chmf_interval(
    inputs: InputParameters,
    redshift: float,
    lnM_lower: Sequence[float],
    lnM_upper: Sequence[float],
    cond_values: Sequence[float],
    redshift_prev: float | None = None,
):
    """Evaluates conditional mass function integrals at a range of mass intervals."""
    if lnM_lower.shape != lnM_upper.shape:
        raise ValueError("the shapes of the two mass-limit arrays must be equal")

    out_prob = np.zeros(len(lnM_lower) * len(cond_values), dtype="f8")
    cond_values = cond_values.astype("f8")
    lnM_lower = lnM_lower.astype("f8")
    lnM_upper = lnM_upper.astype("f8")

    lib.get_halo_chmf_interval(
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct,
        redshift,
        redshift_prev if redshift_prev is not None else -1,
        len(cond_values),
        ffi.cast("double *", ffi.from_buffer(cond_values)),
        len(lnM_lower),
        ffi.cast("double *", ffi.from_buffer(lnM_lower)),
        ffi.cast("double *", ffi.from_buffer(lnM_upper)),
        ffi.cast("double *", ffi.from_buffer(out_prob)),
    )

    return np.reshape(out_prob, (len(cond_values), len(lnM_lower)))


def evaluate_inverse_table(
    inputs: InputParameters,
    cond_array: Sequence[float],
    probabilities: Sequence[float],
    redshift: float,
    redshift_prev: float | None = None,
):
    """Gets the expected number and mass of halos given a condition."""
    if cond_array.shape != probabilities.shape:
        raise ValueError(
            "the shapes of the input arrays `cond_array` and `probabilities"
            " must be equal."
        )

    if redshift_prev is None:
        redshift_prev = -1

    orig_shape = cond_array.shape
    cond_array = np.array(cond_array, dtype="f8").flatten()
    probabilities = np.array(probabilities, dtype="f8").flatten()
    masses = np.zeros_like(cond_array)

    lib.get_halomass_at_probability(
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct,
        redshift,
        redshift_prev,
        len(cond_array),
        ffi.cast("double *", ffi.from_buffer(cond_array)),
        ffi.cast("double *", ffi.from_buffer(probabilities)),
        ffi.cast("double *", ffi.from_buffer(masses)),
    )

    return np.reshape(masses, orig_shape)


def evaluate_FgtrM_cond(
    inputs: InputParameters,
    densities: Sequence[float],
    redshift: float,
    R: float,
):
    """Gets the collapsed fraction from the backend, given a density and condition sigma."""
    orig_shape = densities.shape
    densities = np.array(densities, dtype="f8").flatten()
    fcoll = np.zeros_like(densities)
    dfcoll = np.zeros_like(densities)

    lib.get_conditional_FgtrM(
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct,
        redshift,
        R,
        len(densities),
        ffi.cast("double *", ffi.from_buffer(densities)),
        ffi.cast("double *", ffi.from_buffer(fcoll)),
        ffi.cast("double *", ffi.from_buffer(dfcoll)),
    )
    return np.reshape(fcoll, orig_shape), np.reshape(dfcoll, orig_shape)


def evaluate_SFRD_z(
    *,
    inputs: InputParameters,
    redshifts: Sequence[float],
    log10mturns: Sequence[float],
):
    """Evaluates the global star formation rate density expected at a range of redshifts."""
    if redshifts.shape != log10mturns.shape:
        raise ValueError(
            f"the shapes of the input arrays `redshifts` {redshifts.shape} and `log10mturns` {log10mturns.shape}"
            " must be equal."
        )

    orig_shape = redshifts.shape
    redshifts = np.array(redshifts, dtype="f8").flatten()
    log10mturns = np.array(log10mturns, dtype="f8").flatten()
    sfrd = np.zeros_like(redshifts)
    sfrd_mini = np.zeros_like(redshifts)

    lib.get_global_SFRD_z(
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct,
        len(redshifts),
        ffi.cast("double *", ffi.from_buffer(redshifts)),
        ffi.cast("double *", ffi.from_buffer(log10mturns)),
        ffi.cast("double *", ffi.from_buffer(sfrd)),
        ffi.cast("double *", ffi.from_buffer(sfrd_mini)),
    )

    return np.reshape(sfrd, orig_shape), np.reshape(sfrd_mini, orig_shape)


def evaluate_Nion_z(
    *,
    inputs: InputParameters,
    redshifts: Sequence[float],
    log10mturns: Sequence[float],
):
    """Evaluates the global ionising emissivity expected at a range of redshifts."""
    if redshifts.shape != log10mturns.shape:
        raise ValueError(
            "the shapes of the input arrays `redshifts` and `log10mturns`"
            " must be equal."
        )

    orig_shape = redshifts.shape
    redshifts = np.array(redshifts, dtype="f8").flatten()
    log10mturns = np.array(log10mturns, dtype="f8").flatten()
    nion = np.zeros_like(redshifts)
    nion_mini = np.zeros_like(redshifts)

    lib.get_global_Nion_z(
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct,
        len(redshifts),
        ffi.cast("double *", ffi.from_buffer(redshifts)),
        ffi.cast("double *", ffi.from_buffer(log10mturns)),
        ffi.cast("double *", ffi.from_buffer(nion)),
        ffi.cast("double *", ffi.from_buffer(nion_mini)),
    )

    return np.reshape(nion, orig_shape), np.reshape(nion_mini, orig_shape)


def evaluate_SFRD_cond(
    *,
    inputs: InputParameters,
    redshift: float,
    radius: float,
    densities: Sequence[float],
    log10mturns: Sequence[float],
):
    """Evaluates the conditional star formation rate density expected at a range of densities."""
    if densities.shape != log10mturns.shape:
        raise ValueError(
            "the shapes of the input arrays `densities` and `log10mturns` must be equal"
        )

    orig_shape = densities.shape
    densities = np.array(densities, dtype="f8").flatten()
    log10mturns = np.array(log10mturns, dtype="f8").flatten()
    sfrd = np.zeros_like(densities)
    sfrd_mini = np.zeros_like(densities)

    lib.get_conditional_SFRD(
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct,
        redshift,
        radius,
        len(densities),
        ffi.cast("double *", ffi.from_buffer(densities)),
        ffi.cast("double *", ffi.from_buffer(log10mturns)),
        ffi.cast("double *", ffi.from_buffer(sfrd)),
        ffi.cast("double *", ffi.from_buffer(sfrd_mini)),
    )

    return np.reshape(sfrd, orig_shape), np.reshape(sfrd_mini, orig_shape)


def evaluate_Nion_cond(
    *,
    inputs: InputParameters,
    redshift: float,
    radius: float,
    densities: Sequence[float],
    l10mturns_acg: Sequence[float],
    l10mturns_mcg: Sequence[float],
):
    """Evaluates the conditional ionising emissivity expected at a range of densities."""
    if not (densities.shape == l10mturns_mcg.shape == l10mturns_acg.shape):
        raise ValueError(
            "the shapes of the input arrays `densities` and `log10mturns_x` must be equal"
        )

    orig_shape = densities.shape
    densities = np.array(densities, dtype="f8").flatten()
    l10mturns_acg = np.array(l10mturns_acg, dtype="f8").flatten()
    l10mturns_mcg = np.array(l10mturns_mcg, dtype="f8").flatten()
    nion = np.zeros_like(densities)
    nion_mini = np.zeros_like(densities)

    lib.get_conditional_Nion(
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct,
        redshift,
        radius,
        len(densities),
        ffi.cast("double *", ffi.from_buffer(densities)),
        ffi.cast("double *", ffi.from_buffer(l10mturns_acg)),
        ffi.cast("double *", ffi.from_buffer(l10mturns_mcg)),
        ffi.cast("double *", ffi.from_buffer(nion)),
        ffi.cast("double *", ffi.from_buffer(nion_mini)),
    )

    return np.reshape(nion, orig_shape), np.reshape(nion_mini, orig_shape)


def evaluate_Xray_cond(
    *,
    inputs: InputParameters,
    redshift: float,
    radius: float,
    densities: Sequence[float],
    log10mturns: Sequence[float],
):
    """Evaluates the conditional star formation rate density expected at a range of densities."""
    if densities.shape != log10mturns.shape:
        raise ValueError(
            "the shapes of the input arrays `cond_array` and `probabilities"
            " must be equal."
        )

    orig_shape = densities.shape
    densities = np.array(densities, dtype="f8").flatten()
    log10mturns = np.array(log10mturns, dtype="f8").flatten()
    xray = np.zeros_like(densities)

    lib.get_conditional_Xray(
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct,
        redshift,
        radius,
        len(densities),
        ffi.cast("double *", ffi.from_buffer(densities)),
        ffi.cast("double *", ffi.from_buffer(log10mturns)),
        ffi.cast("double *", ffi.from_buffer(xray)),
    )

    return np.reshape(xray, orig_shape)


def halo_sample_test(
    *,
    inputs: InputParameters,
    redshift: float,
    cond_array,
    redshift_prev: float | None = None,
    buffer_size: int | None = None,
):
    """Constructs a halo sample given a descendant catalogue and redshifts."""
    z_prev = -1 if redshift_prev is None else redshift_prev
    if buffer_size is None:
        buffer_size = get_expected_nhalo(inputs=inputs, redshift=redshift)

    n_cond = cond_array.size
    # all coordinates zero
    crd_in = np.zeros(3 * n_cond).astype("i4")

    cond_array = cond_array.astype("f4")
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
        inputs.random_seed,
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
    halo_masses: Sequence[float],
    star_rng: Sequence[float],
    sfr_rng: Sequence[float],
    xray_rng: Sequence[float],
    halo_coords: Sequence[float] | None = None,
    vcb_grid: Sequence[float] | None = None,
    J_21_LW_grid: Sequence[float] | None = None,
    z_re_grid: Sequence[float] | None = None,
    Gamma12_grid: Sequence[float] | None = None,
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
    # single element zero array to act as the grids (vcb, J_21_LW, z_reion, Gamma12)
    out_buffer = np.zeros(12 * halo_masses.size).astype("f4")
    lo_dim = (inputs.user_params.HII_DIM,) * 3
    n_halos = len(halo_masses)

    if halo_coords is None:
        halo_coords = np.zeros(3 * halo_masses.size, dtype="i4")
    else:
        halo_coords = np.array(halo_coords, dtype="i4")

    if vcb_grid is None:
        vcb_grid = np.zeros(lo_dim, dtype="f4")
    else:
        vcb_grid = np.array(vcb_grid, dtype="f4")

    if J_21_LW_grid is None:
        J_21_LW_grid = np.zeros(lo_dim, dtype="f4")
    else:
        J_21_LW_grid = np.array(J_21_LW_grid, dtype="f4")

    if z_re_grid is None:
        z_re_grid = np.zeros(lo_dim, dtype="f4")
    else:
        z_re_grid = np.array(z_re_grid, dtype="f4")

    if Gamma12_grid is None:
        Gamma12_grid = np.zeros(lo_dim, dtype="f4")
    else:
        Gamma12_grid = np.array(Gamma12_grid, dtype="f4")

    lib.test_halo_props(
        redshift,
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct,
        vcb_grid,
        J_21_LW_grid,
        z_re_grid,
        Gamma12_grid,
        n_halos,
        halo_masses,
        halo_coords,
        star_rng,
        sfr_rng,
        xray_rng,
        ffi.cast("float *", out_buffer.ctypes.data),
    )

    out_buffer = out_buffer.reshape(n_halos, 12)

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
