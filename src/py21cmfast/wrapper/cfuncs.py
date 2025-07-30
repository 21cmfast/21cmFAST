"""Low-level python wrappers of C functions."""

import logging
from collections.abc import Callable, Sequence
from functools import cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

import py21cmfast.c_21cmfast as lib

from .._cfg import config
from ._utils import _process_exitcode
from .inputs import (
    InputParameters,
)

logger = logging.getLogger(__name__)

# Ideally, backend functions that we access here should do all the broadcasting/initialisation themselves
# These decorators are for lower functions which are called directly in one or two lines, like delta_crit

# NOTE: On casting to C pointers:
# -------------------------------
# Currently our wrapper functions directly take C type pointers, which
# requires us to cast the data to the correct type before passing it to the C.
# This is made annoying by the fact that CAMB (which is indirectly imported somewhere)
# appears to have overwritten the ctypes library pointer types which cause errors.
# We will use the nanobind ndarray casters, which allow us to pass
# numpy arrays directly to C++ functions, with size and type information.
# We will have to translate the `integral_wrapper.c` functions to C++ and (maybe?) define
# some wrapper layer functions in C++ for the output struct functions to parse the array data.


def broadcast_input_struct(inputs: InputParameters):
    """Broadcast the parameters to the C library."""
    lib.Broadcast_struct_global_all(
        inputs.simulation_options.cstruct,
        inputs.matter_options.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.astro_options.cstruct,
    )


def broadcast_params(func: Callable) -> Callable:
    """Broadcast the parameters to the C library before calling the function.

    This should be added as a decorator to any function which accesses the
    21cmFAST if it does not directly call `broadcast_input_struct`.
    """

    def wrapper(*args, inputs: InputParameters, **kwargs):
        broadcast_input_struct(inputs)
        return func(*args, inputs=inputs, **kwargs)

    return wrapper


def init_backend_ps(func: Callable) -> Callable:
    """Initialise the backend power-spectrum before calling the function.

    This should be added as a decorator to any function which uses the cosmology
    from the 21cmFAST backend without passing through our regular functions.
    """

    @broadcast_params
    def wrapper(*args, **kwargs):
        lib.init_ps()
        return func(*args, **kwargs)

    return wrapper


def init_sigma_table(func: Callable) -> Callable:
    """Initialise the the sigma interpolation table before calling the function.

    This should be added as a decorator to any function which calls lib.EvaluateSigma
    or the sigma tables directly.
    """

    @init_backend_ps
    def wrapper(*args, inputs: InputParameters, **kwargs):
        sigma_min_mass = kwargs.get("M_min", 1e5)
        sigma_max_mass = kwargs.get("M_max", 1e16)
        if inputs.matter_options.USE_INTERPOLATION_TABLES != "no-interpolation":
            lib.initialiseSigmaMInterpTable(sigma_min_mass, sigma_max_mass)
        return func(*args, inputs=inputs, **kwargs)

    return wrapper


def init_gl(func: Callable) -> Callable:
    """Initialise the Gauss-Legendre integration if required before calling the function.

    Calculates the abcissae weights and stores them as arrays in the backend when either of the
    HMF integrals is set to use Gauss-Legendre integration. This should be added as a decorator to
    any function which calls backend integrals directly.
    """

    @init_sigma_table
    def wrapper(*args, inputs: InputParameters, **kwargs):
        if "GAUSS-LEGENDRE" in (
            inputs.astro_options.INTEGRATION_METHOD_ATOMIC,
            inputs.astro_options.INTEGRATION_METHOD_MINI,
        ):
            # no defualt since GL mass limits are strict
            lib.initialise_GL(np.log(kwargs.get("M_min")), np.log(kwargs.get("M_max")))
        return func(*args, inputs=inputs, **kwargs)

    return wrapper


@broadcast_params
def get_expected_nhalo(
    *,
    redshift: float,
    inputs: InputParameters,
) -> int:
    """Get the expected number of halos in a given box.

    Parameters
    ----------
    redshift : float
        The redshift at which to calculate the halo list.
    inputs: :class:`~InputParameters`
        The input parameters of the run
    """
    return lib.expected_nhalo(
        redshift,
    )


@broadcast_params
def get_halo_list_buffer_size(
    *,
    redshift: float,
    inputs: InputParameters,
    min_size: int = 1000000,
) -> int:
    """Compute the required size of the memory buffer to hold a halo list.

    Parameters
    ----------
    redshift : float
        The redshift at which to calculate the halo list.
    inputs: :class:`~InputParameters`
        The input parameters of the run
    min_size : int, optional
        A minimum size to be used as the buffer.
    """
    # find the buffer size from expected halos in the box
    hbuffer_size = get_expected_nhalo(redshift=redshift, inputs=inputs)
    hbuffer_size = int((hbuffer_size + 1) * config["HALO_CATALOG_MEM_FACTOR"])

    # set a minimum in case of fluctuation at high z
    return int(max(hbuffer_size, min_size))


@broadcast_params
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

    # Run the C code
    return lib.ComputeTau(
        redshifts,
        global_xHI,
        z_re_HeII,
    )


@broadcast_params
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
    astro_options = inputs.astro_options
    astro_params = inputs.astro_params

    redshifts = np.array(redshifts, dtype="float32")
    if astro_options.USE_MINI_HALOS:
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

    lfunc = np.zeros((len(redshifts), nbins))
    Muvfunc = np.zeros((len(redshifts), nbins))
    Mhfunc = np.zeros((len(redshifts), nbins))

    lfunc_MINI = np.zeros((len(redshifts), nbins))
    Muvfunc_MINI = np.zeros((len(redshifts), nbins))
    Mhfunc_MINI = np.zeros((len(redshifts), nbins))

    if component in ("both", "acg"):
        # Run the C code
        errcode = lib.ComputeLF(
            1,
            nbins,
            redshifts,
            mturnovers,
            Muvfunc,
            Mhfunc,
            lfunc,
        )

        _process_exitcode(
            errcode,
            lib.ComputeLF,
            (
                1,
                nbins,
            ),
        )

    if component in ("both", "mcg"):
        # Run the C code
        errcode = lib.ComputeLF(
            2,
            nbins,
            redshifts,
            mturnovers_mini,
            Muvfunc_MINI,
            Mhfunc_MINI,
            lfunc_MINI,
        )

        _process_exitcode(
            errcode,
            lib.ComputeLF,
            (
                2,
                nbins,
            ),
        )

    if component == "both":
        # redo the Muv range using the faintest (most likely MINI) and the brightest (most likely massive)
        lfunc_all = np.zeros((len(redshifts), nbins))
        Muvfunc_all = np.zeros((len(redshifts), nbins))
        Mhfunc_all = np.zeros((len(redshifts), nbins, 2))

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
@broadcast_params
def construct_fftw_wisdoms(
    *,
    inputs: InputParameters,
) -> int:
    """Construct all necessary FFTW wisdoms.

    Parameters
    ----------
    inputs : :class:`~inputs.InputParameters`
        Parameters defining the simulation run.

    """
    # Run the C code
    if inputs.matter_options.USE_FFTW_WISDOM:
        return lib.CreateFFTWWisdoms()
    else:
        return 0


@init_backend_ps
def get_matter_power_values(
    *,
    inputs: InputParameters,
    k_values: Sequence[float],
):
    """Evaluate the power at a certain scale from the 21cmFAST backend."""
    return np.vectorize(lib.power_in_k)(k_values)


@broadcast_params
def evaluate_sigma(
    *,
    inputs: InputParameters,
    masses: NDArray[np.float64],
):
    """
    Evaluate the variance of a mass scale.

    Uses the 21cmfast backend
    """
    masses = masses.astype("f8")
    sigma = np.zeros_like(masses, dtype="f8")
    dsigmasq = np.zeros_like(masses, dtype="f8")

    lib.get_sigma(
        masses,
        sigma,
        dsigmasq,
    )

    return sigma, dsigmasq


@init_backend_ps
def get_growth_factor(
    *,
    inputs: InputParameters,
    redshift: float,
):
    """Get the growth factor at a given redshift."""
    return lib.dicke(redshift)


def get_condition_mass(inputs: InputParameters, R: float):
    """Determine condition masses for backend routines.

    Returns either mass contained within a radius,
    or mass of the Lagrangian cell on HII_DIM
    """
    rhocrit = (
        inputs.cosmo_params.cosmo.critical_density(0).to("M_sun Mpc-3").value
        * inputs.cosmo_params.OMm
    )
    if R == "cell":
        volume = (
            inputs.simulation_options.BOX_LEN / inputs.simulation_options.HII_DIM
        ) ** 3
    else:
        volume = 4.0 / 3.0 * np.pi * R**3

    return volume * rhocrit


@broadcast_params
def get_delta_crit(*, inputs: InputParameters, mass: float, redshift: float):
    """Get the critical collapse density given a mass, redshift and parameters."""
    sigma, _ = evaluate_sigma(inputs=inputs, masses=np.array([mass]))
    growth = get_growth_factor(inputs=inputs, redshift=redshift)
    return get_delta_crit_nu(inputs.matter_options.cdict["HMF"], sigma, growth)


def get_delta_crit_nu(hmf_int_flag: int, sigma: float, growth: float):
    """Get the critical density from sigma and growth factor."""
    # None of the parameter structs are used in this function so we don't need a broadcast
    return lib.get_delta_crit(hmf_int_flag, sigma, growth)


@broadcast_params
def evaluate_condition_integrals(
    inputs: InputParameters,
    cond_array: NDArray[np.float64],
    redshift: float,
    redshift_prev: float | None = None,
):
    """Get the expected number and mass of halos given a condition.

    If USE_INTERPOLATION_TABLES is set to 'hmf-interpolation': Will crash if the table
    has not been initialised, only `cond_array` is used,
    and the rest of the arguments are taken from when the table was initialised.
    """
    cond_array = cond_array.astype("f8")
    n_halo = np.zeros_like(cond_array)
    m_coll = np.zeros_like(cond_array)

    lib.get_condition_integrals(
        redshift,
        redshift_prev if redshift_prev is not None else -1,
        cond_array,
        n_halo,
        m_coll,
    )

    return n_halo, m_coll


@broadcast_params
def integrate_chmf_interval(
    inputs: InputParameters,
    redshift: float,
    lnm_lower: NDArray[np.float64],
    lnm_upper: NDArray[np.float64],
    cond_values: NDArray[np.float64],
    redshift_prev: float | None = None,
):
    """Evaluate conditional mass function integrals at a range of mass intervals."""
    if lnm_lower.shape != lnm_upper.shape:
        raise ValueError("the shapes of the two mass-limit arrays must be equal")
    assert np.all(lnm_lower < lnm_upper)

    out_prob = np.zeros((len(cond_values), len(lnm_lower)), dtype="f8")
    cond_values = cond_values.astype("f8")
    lnm_lower = lnm_lower.astype("f8")
    lnm_upper = lnm_upper.astype("f8")

    lib.get_halo_chmf_interval(
        redshift,
        redshift_prev if redshift_prev is not None else -1,
        cond_values,
        lnm_lower,
        lnm_upper,
        out_prob,
    )

    return out_prob


@broadcast_params
def evaluate_inverse_table(
    inputs: InputParameters,
    cond_array: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    redshift: float,
    redshift_prev: float | None = None,
):
    """Get the expected number and mass of halos given a condition."""
    if cond_array.shape != probabilities.shape:
        raise ValueError(
            "the shapes of the input arrays `cond_array` and `probabilities"
            " must be equal."
        )

    if redshift_prev is None:
        redshift_prev = -1

    cond_array = cond_array.astype("f8")
    probabilities = probabilities.astype("f8")
    masses = np.zeros_like(cond_array)

    lib.get_halomass_at_probability(
        redshift,
        redshift_prev,
        cond_array,
        probabilities,
        masses,
    )

    return masses


@broadcast_params
def evaluate_FgtrM_cond(
    inputs: InputParameters,
    densities: NDArray[np.float64],
    redshift: float,
    R: float,
):
    """Get the collapsed fraction from the backend, given a density and condition sigma."""
    densities = densities.astype("f8")
    fcoll = np.zeros_like(densities)
    dfcoll = np.zeros_like(densities)

    lib.get_conditional_FgtrM(
        redshift,
        R,
        densities,
        fcoll,
        dfcoll,
    )
    return fcoll, dfcoll


@broadcast_params
def evaluate_SFRD_z(
    *,
    inputs: InputParameters,
    redshifts: NDArray[np.float64],
    log10mturns: NDArray[np.float64],
):
    """Evaluate the global star formation rate density expected at a range of redshifts."""
    if redshifts.shape != log10mturns.shape:
        raise ValueError(
            f"the shapes of the input arrays `redshifts` {redshifts.shape} and `log10mturns` {log10mturns.shape}"
            " must be equal."
        )

    redshifts = redshifts.astype("f8")
    log10mturns = log10mturns.astype("f8")
    sfrd = np.zeros_like(redshifts)
    sfrd_mini = np.zeros_like(redshifts)

    lib.get_global_SFRD_z(
        redshifts,
        log10mturns,
        sfrd,
        sfrd_mini,
    )

    return sfrd, sfrd_mini


@broadcast_params
def evaluate_Nion_z(
    *,
    inputs: InputParameters,
    redshifts: NDArray[np.float64],
    log10mturns: NDArray[np.float64],
):
    """Evaluate the global ionising emissivity expected at a range of redshifts."""
    if redshifts.shape != log10mturns.shape:
        raise ValueError(
            "the shapes of the input arrays `redshifts` and `log10mturns`"
            " must be equal."
        )

    redshifts = redshifts.astype("f8")
    log10mturns = log10mturns.astype("f8")
    nion = np.zeros_like(redshifts)
    nion_mini = np.zeros_like(redshifts)

    lib.get_global_Nion_z(
        redshifts,
        log10mturns,
        nion,
        nion_mini,
    )

    return nion, nion_mini


@broadcast_params
def evaluate_SFRD_cond(
    *,
    inputs: InputParameters,
    redshift: float,
    radius: float,
    densities: NDArray[np.float64],
    log10mturns: NDArray[np.float64],
):
    """Evaluate the conditional star formation rate density expected at a range of densities."""
    if densities.shape != log10mturns.shape:
        raise ValueError(
            "the shapes of the input arrays `densities` and `log10mturns` must be equal"
        )

    densities = densities.astype("f8")
    log10mturns = log10mturns.astype("f8")
    sfrd = np.zeros_like(densities)
    sfrd_mini = np.zeros_like(densities)

    lib.get_conditional_SFRD(
        redshift,
        radius,
        densities,
        log10mturns,
        sfrd,
        sfrd_mini,
    )

    return sfrd, sfrd_mini


@broadcast_params
def evaluate_Nion_cond(
    *,
    inputs: InputParameters,
    redshift: float,
    radius: float,
    densities: NDArray[np.float64],
    l10mturns_acg: NDArray[np.float64],
    l10mturns_mcg: NDArray[np.float64],
):
    """Evaluate the conditional ionising emissivity expected at a range of densities."""
    if not (densities.shape == l10mturns_mcg.shape == l10mturns_acg.shape):
        raise ValueError(
            "the shapes of the input arrays `densities` and `log10mturns_x` must be equal"
        )

    densities = densities.astype("f8")
    l10mturns_acg = l10mturns_acg.astype("f8")
    l10mturns_mcg = l10mturns_mcg.astype("f8")
    nion = np.zeros_like(densities)
    nion_mini = np.zeros_like(densities)

    lib.get_conditional_Nion(
        redshift,
        radius,
        densities,
        l10mturns_acg,
        l10mturns_mcg,
        nion,
        nion_mini,
    )

    return nion, nion_mini


@broadcast_params
def evaluate_Xray_cond(
    *,
    inputs: InputParameters,
    redshift: float,
    radius: float,
    densities: NDArray[np.float64],
    log10mturns: NDArray[np.float64],
):
    """Evaluate the conditional star formation rate density expected at a range of densities."""
    if densities.shape != log10mturns.shape:
        raise ValueError(
            "the shapes of the input arrays `cond_array` and `probabilities"
            " must be equal."
        )

    densities = densities.astype("f8")
    log10mturns = log10mturns.astype("f8")
    xray = np.zeros_like(densities)

    lib.get_conditional_Xray(
        redshift,
        radius,
        densities,
        log10mturns,
        xray,
    )

    return xray


@broadcast_params
def sample_halos_from_conditions(
    *,
    inputs: InputParameters,
    redshift: float,
    cond_array,
    redshift_prev: float | None = None,
    buffer_size: int | None = None,
):
    """Construct a halo sample given a descendant catalogue and redshifts."""
    z_prev = -1 if redshift_prev is None else redshift_prev
    if buffer_size is None:
        buffer_size = get_halo_list_buffer_size(inputs=inputs, redshift=redshift)

    n_cond = cond_array.size
    # all coordinates zero
    crd_in = np.zeros((n_cond, 3)).astype("f4")
    cond_array = cond_array.astype("f4")
    nhalo_out = np.zeros(1).astype("i4")
    N_out = np.zeros(n_cond).astype("i4")
    M_out = np.zeros(n_cond).astype("f8")
    exp_M = np.zeros(n_cond).astype("f8")
    exp_N = np.zeros(n_cond).astype("f8")
    halomass_out = np.zeros(buffer_size).astype("f4")
    halocrd_out = np.zeros((buffer_size, 3)).astype("i4")

    lib.single_test_sample(
        inputs.random_seed,
        cond_array,
        crd_in,
        redshift,
        z_prev,
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
        "n_progenitors": N_out,
        "progenitor_mass": M_out,
        "expected_progenitors": exp_N,
        "expected_progenitor_mass": exp_M,
    }


@broadcast_params
def convert_halo_properties(
    *,
    redshift: float,
    inputs: InputParameters,
    halo_masses: NDArray[np.float64],
    star_rng: NDArray[np.float64],
    sfr_rng: NDArray[np.float64],
    xray_rng: NDArray[np.float64],
    halo_coords: NDArray[np.float64] | None = None,
    vcb_grid: NDArray[np.float64] | None = None,
    J_21_LW_grid: NDArray[np.float64] | None = None,
    z_re_grid: NDArray[np.float64] | None = None,
    Gamma12_grid: NDArray[np.float64] | None = None,
):
    """
    Convert a halo catalogue's mass and RNG fields to halo properties.

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
    if not (halo_masses.shape == star_rng.shape == sfr_rng.shape == xray_rng.shape):
        raise ValueError("Halo masses and rng shapes must be identical.")

    n_halos = halo_masses.size
    orig_shape = halo_masses.shape
    out_buffer = np.zeros((n_halos, 12), dtype="f4")
    lo_dim = (inputs.simulation_options.HII_DIM,) * 3

    if halo_coords is None:
        halo_coords = np.zeros((n_halos, 3))
    if vcb_grid is None:
        vcb_grid = np.zeros(lo_dim)
    if J_21_LW_grid is None:
        J_21_LW_grid = np.zeros(lo_dim)
    if z_re_grid is None:
        z_re_grid = np.zeros(lo_dim)
    if Gamma12_grid is None:
        Gamma12_grid = np.zeros(lo_dim)

    vcb_grid = vcb_grid.astype("f4")
    J_21_LW_grid = J_21_LW_grid.astype("f4")
    z_re_grid = z_re_grid.astype("f4")
    Gamma12_grid = Gamma12_grid.astype("f4")

    halo_masses = halo_masses.reshape(n_halos).astype("f4")
    halo_coords = halo_coords.reshape(n_halos, 3).astype("f4")
    star_rng = star_rng.reshape(n_halos).astype("f4")
    sfr_rng = sfr_rng.reshape(n_halos).astype("f4")
    xray_rng = xray_rng.reshape(n_halos).astype("f4")

    lib.test_halo_props(
        redshift,
        vcb_grid,
        J_21_LW_grid,
        z_re_grid,
        Gamma12_grid,
        halo_masses,
        halo_coords,
        star_rng,
        sfr_rng,
        xray_rng,
        out_buffer,
    )

    out_buffer = out_buffer.reshape(n_halos, 12)

    return {
        "halo_mass": out_buffer[:, 0].reshape(orig_shape),
        "halo_stars": out_buffer[:, 1].reshape(orig_shape),
        "halo_sfr": out_buffer[:, 2].reshape(orig_shape),
        "halo_xray": out_buffer[:, 3].reshape(orig_shape),
        "n_ion": out_buffer[:, 4].reshape(orig_shape),
        "halo_wsfr": out_buffer[:, 5].reshape(orig_shape),
        "halo_stars_mini": out_buffer[:, 6].reshape(orig_shape),
        "halo_sfr_mini": out_buffer[:, 7].reshape(orig_shape),
        "mturn_a": out_buffer[:, 8].reshape(orig_shape),
        "mturn_m": out_buffer[:, 9].reshape(orig_shape),
        "mturn_r": out_buffer[:, 10].reshape(orig_shape),
        "metallicity": out_buffer[:, 11].reshape(orig_shape),
    }


@init_sigma_table
def return_uhmf_value(
    *,
    inputs: InputParameters,
    redshift: float,
    mass_values: Sequence[float],
):
    """Return the value of the unconditional halo mass function at given parameters.

    Parameters
    ----------
    inputs : InputParameters
        The input parameters defining the simulation run.
    redshift : float
        The redshift at which to evaluate the halo mass function.
    mass_values : float
        The mass values at which to evaluate the halo mass function.
    """
    growthf = lib.dicke(redshift)
    return np.vectorize(lib.unconditional_hmf)(
        growthf, np.log(mass_values), redshift, inputs.matter_options.cdict["HMF"]
    )


@init_sigma_table
def return_chmf_value(
    *,
    inputs: InputParameters,
    redshift: float,
    mass_values: Sequence[float],
    delta_values: Sequence[float],
    condmass_values: Sequence[float],
):
    """Return the value of the conditional halo mass function at given parameters.

    Parameters
    ----------
    inputs : InputParameters
        The input parameters defining the simulation run.
    redshift : float
        The redshift at which to evaluate the halo mass function.
    mass_values : float
        The mass values at which to evaluate the halo mass function.
    delta : float
        The overdensity at which to evaluate the halo mass function.
    cond_mass : float
        The condition mass at which to evaluate the halo mass function.
    """
    growthf = lib.dicke(redshift)
    sigma = np.vectorize(lib.sigma_z0)(condmass_values)

    return np.vectorize(lib.conditional_hmf)(
        growthf,
        np.log(mass_values[None, None, :]),
        delta_values[:, None, None],
        sigma[None, :, None],
        inputs.matter_options.cdict["HMF"],
    )
