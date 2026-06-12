"""Low-level python wrappers of C functions."""

import logging
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from .._cfg import config
from ..c_21cmfast import ffi, lib
from ..drivers._global_initialization import init_c_state
from ..drivers.global_evolution import GlobalEvolution, run_global_evolution
from ..drivers.lightcone import LightCone
from ._utils import _process_exitcode
from .inputs import InputParameters

logger = logging.getLogger(__name__)

# TODO: a lot of these assume input as numpy arrays via use of .shape, explicitly require this


@init_c_state(sigma=True)
def get_expected_nhalo(*, redshift: float, inputs: InputParameters) -> int:
    """Get the expected number of halos in a given box.

    Parameters
    ----------
    redshift : float
        The redshift at which to calculate the halo list.
    inputs: :class:`~InputParameters`
        The input parameters of the run

    Returns
    -------
    n_halo : float
        The expected number of halos in the box at the given redshift under the given model.

    Raises
    ------
    ValueError :
        If the matter options do not have a discrete halo model.
    """
    if not inputs.matter_options.has_discrete_halos:
        raise ValueError(
            "SOURCE_MODEL must have a discrete halo model in order to calculate the expected number of halos in the box. "
            "Change SOURCE_MODEL to either 'DEXM-ESF' or 'CHMF-SAMPLER' in order to use this function."
        )
    return lib.expected_nhalo(
        redshift,
    )


@init_c_state(sigma=True)
def get_halo_catalog_buffer_size(
    *, redshift: float, inputs: InputParameters, min_size: int = 1000000
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
    hbuffer_size = get_expected_nhalo(
        redshift=redshift,
        inputs=inputs,
    )
    hbuffer_size = int((hbuffer_size + 1) * config["HALO_CATALOG_MEM_FACTOR"])

    # set a minimum in case of fluctuation at high z
    return int(max(hbuffer_size, min_size))


@init_c_state(broadcast_inputs=True)
def compute_mturns(
    *,
    inputs: InputParameters,
    redshifts: float | Sequence[float],
    J_LW_21: float | Sequence[float],
    v_cb: float | Sequence[float],
    ionisation_rate_G12: float | Sequence[float],
    z_reion: float | Sequence[float],
) -> tuple[float, float]:
    """
    Compute the turnover masses for both ACGs and MCGs at a given redshift.

    Parameters
    ----------
    redshifts : array-like
        The redshifts at which to compute the turnover masses.
    J_LW_21 : array-like
        The Lyman-Werner flux in units of 1e-21 erg/s/Hz/cm^2/sr at the given redshifts.
    v_cb : array-like
        The amplitude of the relative velocity between dark matter and baryons in units of km/s at the given redshifts.
    ionisation_rate_G12 : array-like
        The ionisation rate in units of 1e-12 s^-1 at the given redshifts.
    z_reion : array-like
        The reionisation redshift at the given redshifts.

    Returns
    -------
    M_turn_a : array-like
        The turnover mass for atomic cooling halos at the given redshifts.
    M_turn_m : array-like
        The turnover mass for molecular cooling halos at the given redshifts.

    Raises
    ------
    ValueError :
        If the input arrays do not have the same shape.
    """
    inputs_to_check = {
        "J_LW_21": J_LW_21,
        "v_cb": v_cb,
        "ionisation_rate_G12": ionisation_rate_G12,
        "z_reion": z_reion,
    }

    redshifts_shape = np.asarray(redshifts).shape
    for name, value in inputs_to_check.items():
        current_shape = np.asarray(value).shape
        if current_shape != redshifts_shape:
            raise ValueError(
                f"The shapes of redshifts and {name} are not the same! "
                f"Got {redshifts_shape} and {current_shape}."
            )

    M_turn_a_ffi = ffi.new("double *")
    M_turn_m_ffi = ffi.new("double *")

    def _scalar_call(z, j, v, g, zr):
        lib.compute_mturns(z, j, v, g, zr, M_turn_a_ffi, M_turn_m_ffi)
        return M_turn_a_ffi[0], M_turn_m_ffi[0]

    vfunc = np.vectorize(_scalar_call, otypes=[np.float64, np.float64])
    M_turn_a, M_turn_m = vfunc(redshifts, J_LW_21, v_cb, ionisation_rate_G12, z_reion)

    if M_turn_a.ndim == 0:  # scalar input case
        return float(M_turn_a), float(M_turn_m)
    return M_turn_a, M_turn_m


@init_c_state(broadcast_inputs=True)
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
        len(redshifts),
        z,
        xHI,
        z_re_HeII,
    )


@init_c_state(sigma=True)
def compute_luminosity_function(
    *,
    redshifts: Sequence[float],
    inputs: InputParameters,
    nbins: int = 100,
    mturnovers: np.ndarray | None = None,
    mturnovers_mini: np.ndarray | None = None,
    lightcone: LightCone | None = None,
    global_evolution: GlobalEvolution | None = None,
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
    lightcone : :class:`~LightCone` or None, optional
        The lightcone object to use for the computation.
        If None, the function will consider `global_evolution` for the global m_turnover values,
        otherwise they will be extracted from the given lightcone.
    global_evolution : :class:`~GlobalEvolution` or None, optional
        The global evolution object to use for the computation.
        If None, the function will run a global evolution to estimate the global m_turnover values,
        otherwise they will be extracted from the given global evolution.
    component : str, {'both', 'acg', 'mcg}
        The component of the LF to be calculated. Forced to be 'acg' if USE_MINI_HALOS is False.

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

    redshifts = np.array(redshifts, dtype="float32")

    if (mturnovers is not None) or (mturnovers_mini is not None):
        raise TypeError(
            "`mturnovers` and `mturnovers_mini` have been removed. "
            "Please use the `lightcone` or `global_evolution` arguments instead, "
            "or leave unspecified and they will be estimated automatically."
        )

    if not astro_options.USE_MINI_HALOS and component != "acg":
        warnings.warn(
            "USE_MINI_HALOS is False, so only ACG LFs are computed.",
            stacklevel=2,
        )
        component = "acg"

    if lightcone is not None:
        mturnovers_global = pow(10.0, lightcone.global_quantities["log10_mturn_acg"])
        mturnovers_mini_global = pow(
            10.0, lightcone.global_quantities["log10_mturn_mcg"]
        )
    else:
        # If lightcone is not provided, we estimate the turnover masses from the global evolution
        if global_evolution is None:
            global_evolution = run_global_evolution(inputs=inputs)
        mturnovers_global = pow(10.0, global_evolution.quantities["log10_mturn_acg"])
        mturnovers_mini_global = pow(
            10.0, global_evolution.quantities["log10_mturn_mcg"]
        )

    # Interpolate the mturnover arrays at the input redshifts
    mturnovers = np.interp(
        redshifts, inputs.node_redshifts[::-1], mturnovers_global[::-1]
    )
    mturnovers_mini = np.interp(
        redshifts, inputs.node_redshifts[::-1], mturnovers_mini_global[::-1]
    )

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
                1,
                len(redshifts),
            ),
        )

    if component in ("both", "mcg"):
        # Run the C code
        errcode = lib.ComputeLF(
            nbins,
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


@init_c_state(ps=True)
def get_matter_power_values(
    *,
    inputs: InputParameters,
    k_values: Sequence[float],
):
    """Evaluate the matter density power spectrum (at z=0) at a certain scale from the 21cmFAST backend."""
    return np.vectorize(lib.power_in_k)(k_values)


@init_c_state(ps=True)
def get_vcb_power_values(
    *,
    inputs: InputParameters,
    k_values: Sequence[float],
):
    """Evaluate the vcb power spectrum (at kinematic decoupling) at a certain scale from the 21cmFAST backend."""
    if inputs.matter_options.USE_RELATIVE_VELOCITIES:
        return np.vectorize(lib.power_in_vcb)(k_values)
    else:
        raise ValueError(
            "inputs.matter_options.USE_RELATIVE_VELOCITIES must be True in order to compute the v_cb power spectrum."
        )


@init_c_state(sigma=True)
def evaluate_sigma(
    *,
    inputs: InputParameters,
    masses: NDArray[np.floating],
):
    """
    Evaluate the variance of a mass scale.

    Uses the 21cmfast backend
    """
    masses = masses.astype("f8")
    sigma = np.zeros_like(masses)
    dsigmasq = np.zeros_like(masses)

    lib.get_sigma(
        masses.size,
        ffi.cast("double *", ffi.from_buffer(masses)),
        ffi.cast("double *", ffi.from_buffer(sigma)),
        ffi.cast("double *", ffi.from_buffer(dsigmasq)),
    )

    return sigma, dsigmasq


@init_c_state(broadcast_inputs=True)
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


@init_c_state(broadcast_inputs=True)
def get_delta_crit(*, inputs: InputParameters, mass: float, redshift: float):
    """Get the critical collapse density given a mass, redshift and parameters."""
    sigma, _ = evaluate_sigma(inputs=inputs, masses=np.array([mass]))
    growth = get_growth_factor(inputs=inputs, redshift=redshift)
    return get_delta_crit_nu(inputs.matter_options.cdict["HMF"], sigma[0], growth)


def get_delta_crit_nu(hmf_int_flag: int, sigma: float, growth: float):
    """Get the critical density from sigma and growth factor."""
    # None of the parameter structs are used in this function so we don't need a broadcast
    return lib.get_delta_crit(hmf_int_flag, sigma, growth)


@init_c_state(sigma=True)
def evaluate_condition_integrals(
    inputs: InputParameters,
    cond_array: NDArray[np.floating],
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
        cond_array.size,
        ffi.cast("double *", ffi.from_buffer(cond_array)),
        ffi.cast("double *", ffi.from_buffer(n_halo)),
        ffi.cast("double *", ffi.from_buffer(m_coll)),
    )

    return n_halo, m_coll


@init_c_state(sigma=True)
def integrate_chmf_interval(
    inputs: InputParameters,
    redshift: float,
    lnm_lower: NDArray[np.floating],
    lnm_upper: NDArray[np.floating],
    cond_values: NDArray[np.floating],
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
        len(cond_values),
        ffi.cast("double *", ffi.from_buffer(cond_values)),
        len(lnm_lower),
        ffi.cast("double *", ffi.from_buffer(lnm_lower)),
        ffi.cast("double *", ffi.from_buffer(lnm_upper)),
        ffi.cast("double *", ffi.from_buffer(out_prob)),
    )

    return out_prob


@init_c_state(sigma=True)
def evaluate_inverse_table(
    inputs: InputParameters,
    cond_array: NDArray[np.floating],
    probabilities: NDArray[np.floating],
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
        cond_array.size,
        ffi.cast("double *", ffi.from_buffer(cond_array)),
        ffi.cast("double *", ffi.from_buffer(probabilities)),
        ffi.cast("double *", ffi.from_buffer(masses)),
    )

    return masses


@init_c_state(sigma=True)
def evaluate_FgtrM_cond(
    inputs: InputParameters,
    densities: NDArray[np.floating],
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
        densities.size,
        ffi.cast("double *", ffi.from_buffer(densities)),
        ffi.cast("double *", ffi.from_buffer(fcoll)),
        ffi.cast("double *", ffi.from_buffer(dfcoll)),
    )
    return fcoll, dfcoll


@init_c_state(sigma=True)
def evaluate_SFRD_z(
    *,
    inputs: InputParameters,
    redshifts: NDArray[np.floating],
    log10mturns: NDArray[np.floating] | None = None,
    lightcone: LightCone | None = None,
    global_evolution: GlobalEvolution | None = None,
):
    """
    Evaluate the global star formation rate density (in units of M_sun/s/Mpc^3) expected at a range of redshifts.

    Parameters
    ----------
    inputs: :class:`~InputParameters`
        The input parameters defining the simulation run.
    redshifts : array-like
        The redshifts at which to compute the SFRD.
    lightcone : :class:`~LightCone` or None, optional
        The lightcone object to use for the computation.
        If None, the function will consider `global_evolution` for the global m_turnover values,
        otherwise they will be extracted from the given lightcone.
    global_evolution : :class:`~GlobalEvolution` or None, optional
        The global evolution object to use for the computation.
        If None, the function will run a global evolution to estimate the global m_turnover values,
        otherwise they will be extracted from the given global evolution.

    Returns
    -------
    sfrd : np.ndarray
        The global star formation rate density at the given redshifts for ACGs.
    sfrd_mini : np.ndarray or None
        The global star formation rate density at the given redshifts for MCGs.
        Will be None if `USE_MINI_HALOS` is False.
    """
    if log10mturns is not None:
        raise TypeError(
            "`log10mturns` has been removed. "
            "Please use the `lightcone` or `global_evolution` arguments instead, "
            "or leave unspecified and they will be estimated automatically."
        )

    if inputs.astro_options.USE_MINI_HALOS:
        if lightcone is not None:
            log10mturns_mini_global = lightcone.global_quantities["log10_mturn_mcg"]
        else:
            # If lightcone is not provided, we estimate the turnover masses from the global evolution
            if global_evolution is None:
                global_evolution = run_global_evolution(inputs=inputs)
            log10mturns_mini_global = global_evolution.quantities["log10_mturn_mcg"]

        log10mturns_mini = np.interp(
            redshifts, inputs.node_redshifts[::-1], log10mturns_mini_global[::-1]
        )
    else:
        log10mturns_mini = np.zeros_like(redshifts)  # dummy value for no mini halos

    redshifts = np.asarray(redshifts).astype("f8")
    log10mturns_mini = log10mturns_mini.astype("f8")
    sfrd = np.zeros_like(redshifts)
    sfrd_mini = np.zeros_like(redshifts)

    lib.get_global_SFRD_z(
        redshifts.size,
        ffi.cast("double *", ffi.from_buffer(redshifts)),
        ffi.cast("double *", ffi.from_buffer(log10mturns_mini)),
        ffi.cast("double *", ffi.from_buffer(sfrd)),
        ffi.cast("double *", ffi.from_buffer(sfrd_mini)),
    )
    if not inputs.astro_options.USE_MINI_HALOS:
        sfrd_mini = None

    return sfrd, sfrd_mini


@init_c_state(sigma=True)
def evaluate_Nion_z(
    *,
    inputs: InputParameters,
    redshifts: NDArray[np.floating],
    log10mturns: NDArray[np.floating] | None = None,
    lightcone: LightCone | None = None,
    global_evolution: GlobalEvolution | None = None,
):
    """
    Evaluate the global number of ionising photons per baryon, expected at a range of redshifts.

    Parameters
    ----------
    inputs: :class:`~InputParameters`
        The input parameters defining the simulation run.
    redshifts : array-like
        The redshifts at which to compute Nion.
    lightcone : :class:`~LightCone` or None, optional
        The lightcone object to use for the computation.
        If None, the function will consider `global_evolution` for the global m_turnover values,
        otherwise they will be extracted from the given lightcone.
    global_evolution : :class:`~GlobalEvolution` or None, optional
        The global evolution object to use for the computation.
        If None, the function will run a global evolution to estimate the global m_turnover values,
        otherwise they will be extracted from the given global evolution.

    Returns
    -------
    nion : np.ndarray
        The global number of ionising photons per baryon at the given redshifts for ACGs.
    nion_mini : np.ndarray or None
        The global number of ionising photons per baryon at the given redshifts for MCGs.
        Will be None if `USE_MINI_HALOS` is False.
    """
    if log10mturns is not None:
        raise TypeError(
            "`log10mturns` has been removed. "
            "Please use the `lightcone` or `global_evolution` arguments instead, "
            "or leave unspecified and they will be estimated automatically."
        )

    if inputs.astro_options.USE_MINI_HALOS:
        if lightcone is not None:
            log10mturns_mini_global = lightcone.global_quantities["log10_mturn_mcg"]
        else:
            # If lightcone is not provided, we estimate the turnover masses from the global evolution
            if global_evolution is None:
                global_evolution = run_global_evolution(inputs=inputs)
            log10mturns_mini_global = global_evolution.quantities["log10_mturn_mcg"]

        log10mturns_mini = np.interp(
            redshifts, inputs.node_redshifts[::-1], log10mturns_mini_global[::-1]
        )
    else:
        log10mturns_mini = np.zeros_like(redshifts)  # dummy value for no mini halos

    redshifts = np.asarray(redshifts).astype("f8")
    log10mturns_mini = log10mturns_mini.astype("f8")
    nion = np.zeros_like(redshifts)
    nion_mini = np.zeros_like(redshifts)

    lib.get_global_Nion_z(
        redshifts.size,
        ffi.cast("double *", ffi.from_buffer(redshifts)),
        ffi.cast("double *", ffi.from_buffer(log10mturns_mini)),
        ffi.cast("double *", ffi.from_buffer(nion)),
        ffi.cast("double *", ffi.from_buffer(nion_mini)),
    )

    if not inputs.astro_options.USE_MINI_HALOS:
        nion_mini = None

    return nion, nion_mini


@init_c_state(sigma=True)
def evaluate_SFRD_cond(
    *,
    inputs: InputParameters,
    redshift: float,
    radius: float,
    densities: NDArray[np.floating],
    log10mturns: NDArray[np.floating] | None = None,
    lightcone: LightCone | None = None,
    global_evolution: GlobalEvolution | None = None,
):
    """
    Evaluate the conditional star formation rate density (in units of M_sun/s/Mpc^3) expected at a range of densities.

    Parameters
    ----------
    inputs: :class:`~InputParameters`
        The input parameters defining the simulation run.
    redshift : float
        The redshift at which to compute the SFRD.
    radius : float
        The radius of the region at which to compute the conditional SFRD.
    densities : array-like
        The densities at which to compute the conditional SFRD.
    lightcone : :class:`~LightCone` or None, optional
        The lightcone object to use for the computation.
        If None, the function will consider `global_evolution` for the global m_turnover values,
        otherwise they will be extracted from the given lightcone.
    global_evolution : :class:`~GlobalEvolution` or None, optional
        The global evolution object to use for the computation.
        If None, the function will run a global evolution to estimate the global m_turnover values,
        otherwise they will be extracted from the given global evolution.

    Returns
    -------
    sfrd : np.ndarray
        The conditional star formation rate density at the given redshift and radius for ACGs.
    sfrd_mini : np.ndarray or None
        The conditional star formation rate density at the given redshift and radius for MCGs.
        Will be None if `USE_MINI_HALOS` is False.

    Notes
    -----
    This function estimates the conditional SFRD by using the global turnover masses.
    In reality, these turnover masses do not depend solely on redshift, but also on the local
    density field, as well on its environment and history. Since it is impossible to well-define
    the conditional SFRD in given region by only providing redshift and density, we approximate
    the used turnover masses in this calculation to be the global ones.
    """
    if log10mturns is not None:
        raise TypeError(
            "`log10mturns` has been removed. "
            "Please use the `lightcone` or `global_evolution` arguments instead, "
            "or leave unspecified and they will be estimated automatically."
        )

    if inputs.astro_options.USE_MINI_HALOS:
        if lightcone is not None:
            log10mturns_mini_global = lightcone.global_quantities["log10_mturn_mcg"]
        else:
            # If lightcone is not provided, we estimate the turnover masses from the global evolution
            if global_evolution is None:
                global_evolution = run_global_evolution(inputs=inputs)
            log10mturns_mini_global = global_evolution.quantities["log10_mturn_mcg"]

        log10mturn_mini = np.interp(
            redshift, inputs.node_redshifts[::-1], log10mturns_mini_global[::-1]
        )
    else:
        log10mturn_mini = 0.0  # dummy value for no mini halos

    densities = densities.astype("f8")
    sfrd = np.zeros_like(densities)
    sfrd_mini = np.zeros_like(densities)

    lib.get_conditional_SFRD(
        redshift,
        radius,
        densities.size,
        ffi.cast("double *", ffi.from_buffer(densities)),
        log10mturn_mini,
        ffi.cast("double *", ffi.from_buffer(sfrd)),
        ffi.cast("double *", ffi.from_buffer(sfrd_mini)),
    )

    if not inputs.astro_options.USE_MINI_HALOS:
        sfrd_mini = None

    return sfrd, sfrd_mini


@init_c_state(sigma=True)
def evaluate_Nion_cond(
    *,
    inputs: InputParameters,
    redshift: float,
    radius: float,
    densities: NDArray[np.floating],
    l10mturns_acg: NDArray[np.floating] | None = None,
    l10mturns_mcg: NDArray[np.floating] | None = None,
    lightcone: LightCone | None = None,
    global_evolution: GlobalEvolution | None = None,
):
    """
    Evaluate the global number of ionising photons per baryon, expected at a range of densities.

    Parameters
    ----------
    inputs: :class:`~InputParameters`
        The input parameters defining the simulation run.
    redshift : float
        The redshift at which to compute Nion.
    radius : float
        The radius of the region at which to compute the conditional Nion.
    densities : array-like
        The densities at which to compute the conditional Nion.
    lightcone : :class:`~LightCone` or None, optional
        The lightcone object to use for the computation.
        If None, the function will consider `global_evolution` for the global m_turnover values,
        otherwise they will be extracted from the given lightcone.
    global_evolution : :class:`~GlobalEvolution` or None, optional
        The global evolution object to use for the computation.
        If None, the function will run a global evolution to estimate the global m_turnover values,
        otherwise they will be extracted from the given global evolution.

    Returns
    -------
    nion : np.ndarray
        The conditional number of ionising photons per baryon at the given redshift and radius for ACGs.
    nion_mini : np.ndarray or None
        The conditional number of ionising photons per baryon at the given redshift and radius for MCGs.
        Will be None if `USE_MINI_HALOS` is False.

    Notes
    -----
    This function estimates the conditional N_ion by using the global turnover masses.
    In reality, these turnover masses do not depend solely on redshift, but also on the local
    density field, as well on its environment and history. Since it is impossible to well-define
    the conditional N_ion in given region by only providing redshift and density, we approximate
    the used turnover masses in this calculation to be the global ones.
    """
    if (l10mturns_acg is not None) or (l10mturns_mcg is not None):
        raise TypeError(
            "`l10mturns_acg` and `l10mturns_mcg` have been removed. "
            "Please use the `lightcone` or `global_evolution` arguments instead, "
            "or leave unspecified and they will be estimated automatically."
        )

    # TODO: Why this function is the only one that needs the global mturnover values for ACGs?
    if lightcone is not None:
        log10mturns_global = lightcone.global_quantities["log10_mturn_acg"]
        log10mturns_mini_global = lightcone.global_quantities["log10_mturn_mcg"]
    else:
        # If lightcone is not provided, we estimate the turnover masses from the global evolution
        if global_evolution is None:
            global_evolution = run_global_evolution(inputs=inputs)
        log10mturns_global = global_evolution.quantities["log10_mturn_acg"]
        log10mturns_mini_global = global_evolution.quantities["log10_mturn_mcg"]

    log10mturn_acg = np.interp(
        redshift, inputs.node_redshifts[::-1], log10mturns_global[::-1]
    )
    log10mturn_mcg = np.interp(
        redshift, inputs.node_redshifts[::-1], log10mturns_mini_global[::-1]
    )

    densities = densities.astype("f8")
    nion = np.zeros_like(densities)
    nion_mini = np.zeros_like(densities)

    lib.get_conditional_Nion(
        redshift,
        radius,
        densities.size,
        ffi.cast("double *", ffi.from_buffer(densities)),
        log10mturn_acg,
        log10mturn_mcg,
        ffi.cast("double *", ffi.from_buffer(nion)),
        ffi.cast("double *", ffi.from_buffer(nion_mini)),
    )

    if not inputs.astro_options.USE_MINI_HALOS:
        nion_mini = None

    return nion, nion_mini


@init_c_state(sigma=True)
def evaluate_Xray_cond(
    *,
    inputs: InputParameters,
    redshift: float,
    radius: float,
    densities: NDArray[np.floating],
    log10mturns: NDArray[np.floating] | None = None,
    lightcone: LightCone | None = None,
    global_evolution: GlobalEvolution | None = None,
):
    """
    Evaluate the conditional X-ray emissivity (in units of erg/s/Mpc^3) expected at a range of densities.

    Parameters
    ----------
    inputs: :class:`~InputParameters`
        The input parameters defining the simulation run.
    redshift : float
        The redshift at which to compute the conditional X-ray emissivity.
    radius : float
        The radius of the region at which to compute the conditional X-ray emissivity.
    densities : array-like
        The densities at which to compute the conditional X-ray emissivity.
    lightcone : :class:`~LightCone` or None, optional
        The lightcone object to use for the computation.
        If None, the function will consider `global_evolution` for the global m_turnover values,
        otherwise they will be extracted from the given lightcone.
    global_evolution : :class:`~GlobalEvolution` or None, optional
        The global evolution object to use for the computation.
        If None, the function will run a global evolution to estimate the global m_turnover values,
        otherwise they will be extracted from the given global evolution.

    Returns
    -------
    xray_emissivity : np.ndarray
        The conditional X-ray emissivity at the given redshift and radius for ACGs and MCGs combined.

    Notes
    -----
    This function estimates the conditional X-ray emissivity by using the global turnover masses.
    In reality, these turnover masses do not depend solely on redshift, but also on the local
    density field, as well on its environment and history. Since it is impossible to well-define
    the conditional X-ray emissivity in given region by only providing redshift and density, we approximate
    the used turnover masses in this calculation to be the global ones.

    """
    if log10mturns is not None:
        raise TypeError(
            "`log10mturns` has been removed. "
            "Please use the `lightcone` or `global_evolution` arguments instead, "
            "or leave unspecified and they will be estimated automatically."
        )

    if inputs.astro_options.USE_MINI_HALOS:
        if lightcone is not None:
            log10mturns_mini_global = lightcone.global_quantities["log10_mturn_mcg"]
        else:
            # If lightcone is not provided, we estimate the turnover masses from the global evolution
            if global_evolution is None:
                global_evolution = run_global_evolution(inputs=inputs)
            log10mturns_mini_global = global_evolution.quantities["log10_mturn_mcg"]

        log10mturn_mini = np.interp(
            redshift, inputs.node_redshifts[::-1], log10mturns_mini_global[::-1]
        )
    else:
        log10mturn_mini = 0.0  # dummy value for no mini halos

    densities = densities.astype("f8")
    xray_emissivity = np.zeros_like(densities)

    lib.get_conditional_Xray(
        redshift,
        radius,
        densities.size,
        ffi.cast("double *", ffi.from_buffer(densities)),
        log10mturn_mini,
        ffi.cast("double *", ffi.from_buffer(xray_emissivity)),
    )

    return xray_emissivity


@init_c_state(sigma=True)
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
        buffer_size = get_halo_catalog_buffer_size(inputs=inputs, redshift=redshift)

    n_cond = cond_array.size
    # all coordinates zero
    crd_in = np.zeros(3 * n_cond).astype("f4")

    cond_array = cond_array.astype("f4")
    nhalo_out = np.zeros(1).astype("i4")
    N_out = np.zeros(n_cond).astype("i4")
    M_out = np.zeros(n_cond).astype("f8")
    exp_M = np.zeros(n_cond).astype("f8")
    exp_N = np.zeros(n_cond).astype("f8")
    halomass_out = np.zeros(buffer_size).astype("f4")
    halocrd_out = np.zeros(int(3 * buffer_size)).astype("i4")

    lib.single_test_sample(
        inputs.random_seed,
        n_cond,
        ffi.cast("float *", cond_array.ctypes.data),
        ffi.cast("float *", crd_in.ctypes.data),
        redshift,
        z_prev,
        ffi.cast("int *", nhalo_out.ctypes.data),
        ffi.cast("int *", N_out.ctypes.data),
        ffi.cast("double *", exp_N.ctypes.data),
        ffi.cast("double *", M_out.ctypes.data),
        ffi.cast("double *", exp_M.ctypes.data),
        ffi.cast("float *", halomass_out.ctypes.data),
        ffi.cast("float *", halocrd_out.ctypes.data),
    )

    return {
        "n_halo_total": nhalo_out[0],
        "halo_masses": halomass_out,
        "n_progenitors": N_out,
        "progenitor_mass": M_out,
        "expected_progenitors": exp_N,
        "expected_progenitor_mass": exp_M,
    }


@init_c_state(broadcast_inputs=True)
def convert_halo_properties(
    *,
    redshift: float,
    inputs: InputParameters,
    halo_masses: NDArray[np.floating],
    star_rng: NDArray[np.floating],
    sfr_rng: NDArray[np.floating],
    xray_rng: NDArray[np.floating],
    halo_coords: NDArray[np.floating] | None = None,
    vcb_grid: NDArray[np.floating] | None = None,
    J_21_LW_grid: NDArray[np.floating] | None = None,
    z_re_grid: NDArray[np.floating] | None = None,
    Gamma12_grid: NDArray[np.floating] | None = None,
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
    out_buffer = np.zeros((n_halos, 12), dtype="f4")
    lo_dim = (inputs.simulation_options.HII_DIM,) * 3

    if halo_coords is None:
        halo_coords = np.zeros(3 * n_halos)
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

    halo_masses = halo_masses.astype("f4")
    halo_coords = halo_coords.astype("f4")
    star_rng = star_rng.astype("f4")
    sfr_rng = sfr_rng.astype("f4")
    xray_rng = xray_rng.astype("f4")

    lib.test_halo_props(
        redshift,
        ffi.cast("float *", vcb_grid.ctypes.data),
        ffi.cast("float *", J_21_LW_grid.ctypes.data),
        ffi.cast("float *", z_re_grid.ctypes.data),
        ffi.cast("float *", Gamma12_grid.ctypes.data),
        n_halos,
        ffi.cast("float *", halo_masses.ctypes.data),
        ffi.cast("float *", halo_coords.ctypes.data),
        ffi.cast("float *", star_rng.ctypes.data),
        ffi.cast("float *", sfr_rng.ctypes.data),
        ffi.cast("float *", xray_rng.ctypes.data),
        ffi.cast("float *", out_buffer.ctypes.data),
    )

    out_buffer = out_buffer.reshape(n_halos, 12)

    return {
        "halo_mass": out_buffer[:, 0].reshape(halo_masses.shape),
        "halo_stars": out_buffer[:, 1].reshape(halo_masses.shape),
        "halo_sfr": out_buffer[:, 2].reshape(halo_masses.shape),
        "halo_xray": out_buffer[:, 3].reshape(halo_masses.shape),
        "n_ion": out_buffer[:, 4].reshape(halo_masses.shape),
        "halo_wsfr": out_buffer[:, 5].reshape(halo_masses.shape),
        "halo_stars_mini": out_buffer[:, 6].reshape(halo_masses.shape),
        "halo_sfr_mini": out_buffer[:, 7].reshape(halo_masses.shape),
        "mturn_a": out_buffer[:, 8].reshape(halo_masses.shape),
        "mturn_m": out_buffer[:, 9].reshape(halo_masses.shape),
        "mturn_r": out_buffer[:, 10].reshape(halo_masses.shape),
        "metallicity": out_buffer[:, 11].reshape(halo_masses.shape),
    }


@init_c_state(sigma=True)
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


@init_c_state(sigma=True)
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
