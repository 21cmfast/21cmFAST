"""
Module for the photon conservation models.

The excursion set reionisation model applied in ionize_box does not conserve photons.
As a result there is an offset between the expected global ionized fraction and the value
calculated from the ionized bubble maps. These models apply approximate corrections in order
to bring the bubble maps more in line with the expected global values.

The application of the model is controlled by the flag option PHOTON_CONS_TYPE, which
can take 4 values:

0. No correction is applied
1. We use the ionizing emissivity grids from a different redshift when calculating
   ionized bubble maps, this mapping from one redshift to another is obtained by
   performing a calibration simulation and measuring its redshift difference with the
   expected global evolution.
2. The power-law slope of the ionizing escape fraction is adjusted, using a fit
   ALPHA_ESC -> X + Y*Q(z), where Q is the expected global ionized fraction. This
   relation is fit by performing a calibration simulation as in (1), and comparing it to
   a range of expected global histories with different power-law slopes.
3. The normalisation of the ionizing escape fraction is adjusted, using a fit
   F_ESC10 -> X + Y*Q(z), where Q is the expected global ionized fraction. This relation
   is fit by performing a calibration simulation as in (1), and taking its ratio with
   the expected global evolution.

Notes
-----
The function map for the photon conservation model looks like::

    wrapper.run_lightcone/coeval()
        setup_photon_cons_correction()
            calibrate_photon_cons_correction()
                _init_photon_conservation_correction() --> computes and stores global evolution
                -->perfoms calibration simulation
                _calibrate_photon_conservation_correction() --> stores calibration evolution
            IF PHOTON_CONS_TYPE=='z-photoncons':
                lib.ComputeIonizedBox()
                    lib.adjust_redshifts_for_photoncons()
                        lib.determine_deltaz_for_photoncons() (first time only) --> calculates the deltaz array with some smoothing
                        --> does more smoothing and returns the adjusted redshift
            ELIF PHOTON_CONS_TYPE=='alpha-photoncons':
                photoncons_alpha() --> computes and stores ALPHA_ESC shift vs global neutral fraction
                lib.ComputeIonizedBox()
                    get_fesc_fit() --> applies the fit to the current redshift
            ELIF PHOTON_CONS_TYPE=='f-photoncons':
                photoncons_fesc() --> computes and stores F_ESC10 shift vs global neutral fraction
                lib.ComputeIonizedBox()
                    get_fesc_fit() --> applies the fit to the current redshift

"""

import logging

import attrs
import numpy as np
from scipy.optimize import curve_fit

from ..c_21cmfast import ffi, lib
from ._utils import _process_exitcode
from .cfuncs import broadcast_params
from .inputs import InputParameters
from .outputs import InitialConditions

logger = logging.getLogger(__name__)


# NOTE: if/when we move the photoncons model data to python we should use this
#   structure to hold z, xHI and parameter delta arrays
@attrs.define(kw_only=True)
class _PhotonConservationState:
    """Singleton class which contains the state of the photon-conservation model."""

    calibration_inputs: InputParameters | None = None

    @property
    def c_memory_allocated(self) -> bool:
        """Whether the memory for the parameter shifts has been allocated in the backend."""
        return lib.photon_cons_allocated

    @c_memory_allocated.setter
    def c_memory_allocated(self, val):
        lib.photon_cons_allocated = ffi.cast("bool", val)


_photoncons_state = _PhotonConservationState()


@broadcast_params
def _init_photon_conservation_correction(*, inputs, **kwargs):
    # This function calculates the global expected evolution of reionisation and saves
    #   it to C global arrays z_Q and Q_value (as well as other non-global confusingly named arrays),
    #   and constructs a GSL interpolator z_at_Q_spline
    return lib.InitialisePhotonCons()


def _calibrate_photon_conservation_correction(
    *, redshifts_estimate, nf_estimate, NSpline
):
    # This function passes the calibration simulation results to C,
    #       Storing a clipped version in global arrays nf_vals and z_vals,
    #       and constructing the GSL interpolator z_NFHistory_spline
    redshifts_estimate = np.array(redshifts_estimate, dtype="float64")
    nf_estimate = np.array(nf_estimate, dtype="float64")

    z = ffi.cast("double *", ffi.from_buffer(redshifts_estimate))
    xHI = ffi.cast("double *", ffi.from_buffer(nf_estimate))

    logger.debug(f"PhotonCons nf estimates: {nf_estimate}")
    return lib.PhotonCons_Calibration(z, xHI, NSpline)


def _calc_zstart_photon_cons():
    # gets the starting redshift of the z-based photon conservation model
    #   Set by neutral fraction astro_params.PHOTONCONS_ZSTART
    from ._utils import _call_c_simple

    return _call_c_simple(lib.ComputeZstart_PhotonCons)


def _get_photon_nonconservation_data() -> dict:
    """
    Access C global data representing the photon-nonconservation corrections.

    .. note::  If photon conservation is switched off via PHOTON_CONS_TYPE='no-photoncons' or the
               initialisation for photon conservation has not been performed yet, this
               will return None.

    Returns
    -------
    dict :
      z_analytic: array of redshifts defining the analytic ionized fraction
      Q_analytic: array of analytic  ionized fractions corresponding to `z_analytic`
      z_calibration: array of redshifts defining the ionized fraction from 21cmFAST without
      recombinations
      nf_calibration: array of calibration ionized fractions corresponding to `z_calibration`
      delta_z_photon_cons: the change in redshift required to calibrate 21cmFAST, as a function
      of z_calibration
      nf_photoncons: the neutral fraction as a function of redshift
    """
    # Check if photon conservation has been initialised at all
    if not _photoncons_state.c_memory_allocated:
        return {}

    arbitrary_large_size = 2000

    data = np.zeros((6, arbitrary_large_size))

    IntVal1 = np.array(np.zeros(1), dtype="int32")
    IntVal2 = np.array(np.zeros(1), dtype="int32")
    IntVal3 = np.array(np.zeros(1), dtype="int32")

    c_z_at_Q = ffi.cast("double *", ffi.from_buffer(data[0]))
    c_Qval = ffi.cast("double *", ffi.from_buffer(data[1]))
    c_z_cal = ffi.cast("double *", ffi.from_buffer(data[2]))
    c_nf_cal = ffi.cast("double *", ffi.from_buffer(data[3]))
    c_PC_nf = ffi.cast("double *", ffi.from_buffer(data[4]))
    c_PC_deltaz = ffi.cast("double *", ffi.from_buffer(data[5]))

    c_int_NQ = ffi.cast("int *", ffi.from_buffer(IntVal1))
    c_int_NC = ffi.cast("int *", ffi.from_buffer(IntVal2))
    c_int_NP = ffi.cast("int *", ffi.from_buffer(IntVal3))

    # Run the C code
    errcode = lib.ObtainPhotonConsData(
        c_z_at_Q,
        c_Qval,
        c_int_NQ,
        c_z_cal,
        c_nf_cal,
        c_int_NC,
        c_PC_nf,
        c_PC_deltaz,
        c_int_NP,
    )

    _process_exitcode(errcode, lib.ObtainPhotonConsData, ())

    ArrayIndices = [
        IntVal1[0],
        IntVal1[0],
        IntVal2[0],
        IntVal2[0],
        IntVal3[0],
        IntVal3[0],
    ]

    data_list = [
        "z_analytic",
        "Q_analytic",
        "z_calibration",
        "nf_calibration",
        "nf_photoncons",
        "delta_z_photon_cons",
    ]

    return {
        name: d[:index]
        for name, d, index in zip(data_list, data, ArrayIndices, strict=True)
    }


def setup_photon_cons(
    initial_conditions: InitialConditions,
    inputs: InputParameters | None = None,
    **kwargs,
):
    r"""
    Set up the photon non-conservation correction.

    First performs a simplified calibration simulation and saves its neutral fraction history,
    to be matched to the analytic expression from solving the filling factor
    ODE.

    This matching can happen via a redshift adjustment, or an adjustment to the escape fraction
    power-law parameters


    Parameters
    ----------
    initial_conditions : :class:`~InitialConditions`,
        The InitialConditions instance to use for the photonconservation calculation
    inputs : :class:`~InputParameters`, optional
        An InputParameters instance. If not given will taken from initial_conditions.

    Other Parameters
    ----------------
    Any other parameters able to be passed to :func:`compute_initial_conditions`.
    """
    if inputs is None:
        inputs = initial_conditions.inputs

    if inputs.astro_options.PHOTON_CONS_TYPE == "no-photoncons":
        return {}

    from ..drivers._param_config import check_consistency_of_outputs_with_inputs

    check_consistency_of_outputs_with_inputs(inputs, [initial_conditions])

    # calculate global and calibration simulation xH histories and save them in C
    calibrate_photon_cons(
        inputs=inputs,
        initial_conditions=initial_conditions,
        **kwargs,
    )
    _photoncons_state.calibration_inputs = inputs

    # The PHOTON_CONS_TYPE == 1 case is handled in C (for now....), but we get the data anyway
    if inputs.astro_options.PHOTON_CONS_TYPE == "z-photoncons":
        photoncons_data = _get_photon_nonconservation_data()

    if inputs.astro_options.PHOTON_CONS_TYPE == "alpha-photoncons":
        photoncons_data = photoncons_alpha(inputs, **kwargs)

    if inputs.astro_options.PHOTON_CONS_TYPE == "f-photoncons":
        photoncons_data = photoncons_fesc(inputs)

    return photoncons_data


def calibrate_photon_cons(
    inputs: InputParameters,
    initial_conditions: InitialConditions,
    **kwargs,
):
    r"""
    Perform a photon conservation calibration simulation.

    Parameters
    ----------
    initial_conditions : :class:`~InitialConditions`,
        The InitialConditions instance to use for the photonconservation calculation
    inputs : :class:`~InputParameters`,
        An InputParameters instance.

    Other Parameters
    ----------------
    See docs of :func:`compute_initial_conditions` for more information.
    """
    # avoiding circular imports by importing here
    from ..drivers.single_field import compute_ionization_field, perturb_field

    # Create a new astro_params and astro_options just for the photon_cons correction
    # NOTE: Since the calibration cannot do INHOMO_RECO, we set the R_BUBBLE_MAX
    #   to the default w/o recombinations ONLY when the original box has INHOMO_RECO enabled.
    # TODO: figure out if it's possible to find a "closest" Rmax, since the correction fails when
    # the histories are too different.

    # Using the halo sampling twice would be very slow, so switch to L-INTEGRAL for the calibration
    source_model_calibration = {
        "E-INTEGRAL": "E-INTEGRAL",
        "L-INTEGRAL": "L-INTEGRAL",
        "DEXM-ESF": "L-INTEGRAL",
        "CHMF-SAMPLER": "L-INTEGRAL",
    }
    inputs_calibration = inputs.evolve_input_structs(
        USE_TS_FLUCT=False,
        INHOMO_RECO=False,
        USE_MINI_HALOS=False,
        SOURCE_MODEL=source_model_calibration[inputs.matter_options.SOURCE_MODEL],
        PHOTON_CONS_TYPE="no-photoncons",
        R_BUBBLE_MAX=(
            15 if inputs.astro_options.INHOMO_RECO else inputs.astro_params.R_BUBBLE_MAX
        ),
    )
    ib = None
    prev_perturb = None

    # Arrays for redshift and neutral fraction for the calibration curve
    neutral_fraction_photon_cons = []

    # Initialise the analytic expression for the reionisation history
    logger.info("About to start photon conservation correction")
    _init_photon_conservation_correction(inputs=inputs, **kwargs)
    # Determine the starting redshift to start scrolling through to create the
    # calibration reionisation history
    logger.info("Calculating photon conservation zstart")
    z = _calc_zstart_photon_cons()

    fast_node_redshifts = [z]

    # NOTE: Not checking any redshift consistency for the calibration run
    #   Since the z-step is Q-dependent, we can't predict the redshifts
    inputs_calibration = inputs_calibration.clone(node_redshifts=None)

    while z > inputs.astro_params.PHOTONCONS_CALIBRATION_END:
        # Determine the ionisation box with recombinations, spin temperature etc.
        # turned off.
        this_perturb = perturb_field(
            redshift=z,
            inputs=inputs_calibration,
            initial_conditions=initial_conditions,
            **kwargs,
        )

        ib2 = compute_ionization_field(
            inputs=inputs_calibration,
            previous_ionized_box=ib,
            initial_conditions=initial_conditions,
            perturbed_field=this_perturb,
            previous_perturbed_field=prev_perturb,
            **kwargs,
        )

        mean_nf = np.mean(ib2.get("neutral_fraction"))

        # Save mean/global quantities
        neutral_fraction_photon_cons.append(mean_nf)

        # Can speed up sampling in regions where the evolution is slower
        if 0.3 < mean_nf <= 0.9:
            z -= 0.15
        elif 0.01 < mean_nf <= 0.3:
            z -= 0.05
        else:
            z -= 0.5

        ib = ib2
        if inputs.astro_options.USE_MINI_HALOS:
            prev_perturb = this_perturb

        fast_node_redshifts.append(z)

    fast_node_redshifts = np.array(fast_node_redshifts[::-1])
    neutral_fraction_photon_cons = np.array(neutral_fraction_photon_cons[::-1])

    # Construct the spline for the calibration curve
    logger.info("Calibrating photon conservation correction")
    _calibrate_photon_conservation_correction(
        redshifts_estimate=fast_node_redshifts,
        nf_estimate=neutral_fraction_photon_cons,
        NSpline=len(fast_node_redshifts),
    )


# (Jdavies): I needed a function to access the delta z from the wrapper
# get_photoncons_data does not have the edge cases that adjust_redshifts_for_photoncons does
@broadcast_params
def get_photoncons_dz(inputs, redshift, **kwargs):
    """Access the delta z arrays from the photon conservation model in C."""
    deltaz = np.zeros(1).astype("f4")
    redshift_pc_in = np.array([redshift]).astype("f4")
    stored_redshift_pc_in = np.array([redshift]).astype("f4")
    lib.adjust_redshifts_for_photoncons(
        ffi.cast("float *", redshift_pc_in.ctypes.data),
        ffi.cast("float *", stored_redshift_pc_in.ctypes.data),
        ffi.cast("float *", deltaz.ctypes.data),
    )

    return redshift_pc_in[0], stored_redshift_pc_in[0], deltaz[0]


# NOTE: alpha_func here MUST MATCH the C version TODO: remove one of them
def alpha_func(Q, a_const, a_slope):
    """Linear Function to fit in the simpler photon conservation model."""
    return a_const + a_slope * Q


# (jdavies): this will be a very hacky way to make a (d_alphastar vs z) array
# for a photoncons done by ALPHA_STAR instead of redshift
# This will work by taking the calibration simulation, plotting a RANGE of analytic
# Q vs z curves for different ALPHA_STAR, and then finding the aloha star which has the inverse ratio
# with the reference analytic as the calibration
# TODO: don't rely on the photoncons functions since they do a bunch of other stuff in C
def photoncons_alpha(inputs, **kwargs):
    """Run the Simpler photons conservation model using ALPHA_ESC.

    This adjusts the slope of the escape fraction instead of redshifts to match a global
    evolution.
    """
    # HACK: I need to allocate the deltaz arrays so I can return the other ones properly, this isn't a great solution
    # TODO: Move the deltaz interp tables to python
    if not _photoncons_state.c_memory_allocated:
        lib.determine_deltaz_for_photoncons()
        _photoncons_state.c_memory_allocated = True

    # Q(analytic) limits to fit the curve
    max_q_fit = 0.99
    min_q_fit = 0.2

    ap_c = inputs.astro_params.cdict

    ref_pc_data = _get_photon_nonconservation_data()
    z = ref_pc_data["z_calibration"]
    alpha_arr = (
        np.linspace(-2.0, 1.0, num=31) + ap_c["ALPHA_ESC"]
    )  # roughly -0.1 steps for an extended range of alpha
    test_pc_data = np.zeros((alpha_arr.size, ref_pc_data["z_calibration"].size))

    # fit to the same z-array
    ref_interp = np.interp(
        ref_pc_data["z_calibration"],
        ref_pc_data["z_analytic"],
        ref_pc_data["Q_analytic"],
    )
    for i, a in enumerate(alpha_arr):
        # alter astro params with new alpha
        inputs_photoncons = inputs.evolve_input_structs(ALPHA_ESC=a)

        # find the analytic curve wth that alpha
        # TODO: Theres a small memory leak here since global arrays are allocated (for some reason)
        # TODO: use ffi to free them?
        #       This will be fixed by moving the photoncons to python
        _init_photon_conservation_correction(inputs=inputs_photoncons, **kwargs)

        # save it
        pcd_buf = _get_photon_nonconservation_data()

        # interpolate to the calibration redshifts
        test_pc_data[i, ...] = np.interp(
            ref_pc_data["z_calibration"], pcd_buf["z_analytic"], pcd_buf["Q_analytic"]
        )

    # filling factors sometimes go above 1, this causes problems in late-time ratios
    # I want this in the test alphas to get the right photon ratio, but not in the reference analytic
    ref_interp[ref_interp > 1] = 1.0

    # ratio of each alpha with calibration
    ratio_test = (test_pc_data) / ref_interp[None, ...]
    # ratio of given alpha with calibration
    ratio_ref = (1 - ref_pc_data["nf_calibration"]) / ref_interp

    ratio_diff = ratio_test - 1 / ratio_ref[None, :]  # find N(alpha)/ref == ref/cal
    diff_test = (
        (test_pc_data)
        + (1 - ref_pc_data["nf_calibration"])[None, ...]
        - 2 * ref_interp[None, ...]
    )  # find N(alpha) - ref == ref - cal
    reverse_test = (
        test_pc_data - (1 - ref_pc_data["nf_calibration"])[None, ...]
    )  # find NF(alpha) == cal, then apply - alpha

    alpha_estimate_ratio = np.zeros_like(z)
    alpha_estimate_diff = np.zeros_like(z)
    alpha_estimate_reverse = np.zeros_like(z)

    # find the roots of each function
    roots_ratio_idx = np.where(np.diff(np.sign(ratio_diff), axis=0))
    roots_diff_idx = np.where(np.diff(np.sign(diff_test), axis=0))
    roots_reverse_idx = np.where(np.diff(np.sign(reverse_test), axis=0))

    # find alpha estimates
    for arr_in, roots_arr, arr_out in zip(
        [ratio_diff, diff_test, reverse_test],
        [roots_ratio_idx, roots_diff_idx, roots_reverse_idx],
        [alpha_estimate_ratio, alpha_estimate_diff, alpha_estimate_reverse],
        strict=True,
    ):
        last_alpha = ap_c["ALPHA_ESC"]
        for i in range(z.size)[::-1]:
            # get the roots at this redshift
            roots_z = np.where(roots_arr[1] == i)
            # if there are no roots, assign nan
            if roots_z[0].size == 0:
                arr_out[i] = np.nan
                continue

            alpha_idx = roots_arr[0][roots_z]

            # interpolate
            y0 = arr_in[alpha_idx, i]
            y1 = arr_in[alpha_idx + 1, i]
            x0 = alpha_arr[alpha_idx]
            x1 = alpha_arr[alpha_idx + 1]
            guesses = -y0 * (x1 - x0) / (y1 - y0) + x0

            # choose the root which gives the smoothest alpha vs z curve
            arr_out[i] = guesses[np.argmin(np.fabs(guesses - last_alpha))]
            last_alpha = arr_out[i]

    # initialise the output structure before the fits
    results = {
        "z_calibration": ref_pc_data["z_calibration"],
        "z_analytic": ref_pc_data["z_analytic"],
        "Q_analytic": ref_pc_data["Q_analytic"],
        "nf_photoncons": 1 - ref_interp,
        "Q_alpha": test_pc_data,
        "nf_calibration": ref_pc_data["nf_calibration"],
        "alpha_ratio": alpha_estimate_ratio,
        "alpha_diff": alpha_estimate_diff,
        "alpha_reverse": alpha_estimate_reverse,
        "alpha_arr": alpha_arr,
        "fit_yint": ap_c["ALPHA_ESC"],
        "fit_slope": 0,  # start with no correction
        "found_alpha": False,
    }

    # adjust the reverse one (we found the alpha which is close to the calibration sim, undo it)
    alpha_estimate_reverse = 2 * ap_c["ALPHA_ESC"] - alpha_estimate_reverse

    # fit to the curve
    # make sure there's an estimate and Q isn't too high/low
    fit_alpha = alpha_estimate_ratio
    sel = np.isfinite(fit_alpha) & (ref_interp < max_q_fit) & (ref_interp > min_q_fit)

    # if there are no alpha roots found, it's likely this is a strange reionisation history
    # but we can't apply the alpha correction so throw an error

    # if we can't fit due to not enough ionisation, catch that here
    if ref_interp.max() < min_q_fit:
        results["fit_yint"] = last_alpha
        logger.warning(
            f"These parameters result in little ionisation, running with flat alpha correction {last_alpha}"
        )

    elif np.count_nonzero(sel) == 1:
        results["fit_yint"] = last_alpha
        logger.warning(
            f"ONLY ONE REDSHIFT HAD AN ALPHA FIT WITHIN THE RANGE, running with flat alpha correction {last_alpha}"
        )

    # here there are no fits, meaning we likely have a strange reionisaiton history where the photon conservation is worse than alpha can correct
    elif np.count_nonzero(sel) == 0:
        logger.warning(
            "No alpha within the range can fit the global history for any z, running without alpha correction"
        )
        logger.warning(
            "THIS REIONISAITON HISTORY IS LIKELY HIGHLY NON-CONSERVING OF PHOTONS"
        )

    else:
        popt, _pcov = curve_fit(alpha_func, ref_interp[sel], fit_alpha[sel])
        # pass to C
        logger.info(f"ALPHA_ESC Original = {ap_c['ALPHA_ESC']:.3f}")
        logger.info(f"Running with ALPHA_ESC = {popt[0]:.2f} + {popt[1]:.2f} * Q")

        results["fit_yint"] = popt[0]
        results["fit_slope"] = popt[1]
        results["found_alpha"] = True

    lib.set_alphacons_params(results["fit_yint"], results["fit_slope"])

    return results


def photoncons_fesc(inputs):
    """Run the Even Simpler photon conservation model using F_ESC10.

    Adjusts the normalisation of the escape fraction to match a global evolution.
    """
    # HACK: I need to allocate the deltaz arrays so I can return the other ones properly, this isn't a great solution
    if not _photoncons_state.c_memory_allocated:
        lib.determine_deltaz_for_photoncons()
        _photoncons_state.c_memory_allocated = True

    # Q(analytic) limits to fit the curve
    max_q_fit = 0.99
    min_q_fit = 0.2

    ref_pc_data = _get_photon_nonconservation_data()

    # fit to the same z-array
    ref_interp = np.interp(
        ref_pc_data["z_calibration"],
        ref_pc_data["z_analytic"],
        ref_pc_data["Q_analytic"],
    )

    ap_c = inputs.astro_params.cdict
    # filling factors sometimes go above 1, this causes problems in late-time ratios
    # I want this in the test alphas to get the right photon ratio, but not in the reference analytic
    ref_interp[ref_interp > 1] = 1.0

    # ratio of each alpha with calibration
    ratio_ref = ref_interp / (1 - ref_pc_data["nf_calibration"])

    fit_fesc = ratio_ref * ap_c["F_ESC10"]
    sel = np.isfinite(fit_fesc) & (ref_interp < max_q_fit) & (ref_interp > min_q_fit)

    popt, _pcov = curve_fit(alpha_func, ref_interp[sel], fit_fesc[sel])
    # pass to C
    logger.info(f"F_ESC10 Original = {ap_c['F_ESC10']:.3f}")
    logger.info(f"Running with F_ESC10 = {popt[0]:.2f} + {popt[1]:.2f} * Q")

    # initialise the output structure before the fits
    results = {
        "z_calibration": ref_pc_data["z_calibration"],
        "z_analytic": ref_pc_data["z_analytic"],
        "Q_analytic": ref_pc_data["Q_analytic"],
        "nf_calibration": ref_pc_data["nf_calibration"],
        "nf_photoncons": 1 - ref_interp,
        "Q_ratio": ratio_ref,
        "fesc_target": fit_fesc,
        "fit_yint": popt[0],
        "fit_slope": popt[1],  # start with no correction
    }

    lib.set_alphacons_params(results["fit_yint"], results["fit_slope"])

    return results
