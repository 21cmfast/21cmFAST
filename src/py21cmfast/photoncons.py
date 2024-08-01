"""
Module for the photon conservation models.

The excursion set reionisation model applied in ionize_box does not conserve photons.
as a result there is an offset between the expected global ionized fraction and the value
calculated from the ionized bubble maps. These models apply approximate corrections in order
to bring the bubble maps more in line with the expected global values.

The application of the model is controlled by the flag option PHOTON_CONS_TYPE, which can take 4 values:

0: No correction is applied

1: We use the ionizing emissivity grids from a different redshift when calculating ionized bubble maps, this
    mapping from one redshift to another is obtained by performing a calibration simulation and measuring its
    redshift difference with the expected global evolution.

2: The power-law slope of the ionizing escape fraction is adjusted, using a fit ALPHA_ESC -> X + Y*Q(z), where Q is the
    expected global ionized fraction. This relation is fit by performing a calibration simulation as in (1), and
    comparing it to a range of expected global histories with different power-law slopes

3: The normalisation of the ionizing escape fraction is adjusted, using a fit F_ESC10 -> X + Y*Q(z), where Q is the
    expected global ionized fraction. This relation is fit by performing a calibration simulation as in (1), and
    taking its ratio with the expected global evolution

"""

import logging
import numpy as np
from copy import deepcopy
from scipy.optimize import curve_fit

from .c_21cmfast import ffi, lib
from .wrapper._utils import _process_exitcode
from .wrapper.inputs import (
    AstroParams,
    CosmoParams,
    FlagOptions,
    UserParams,
    global_params,
    validate_all_inputs,
)

logger = logging.getLogger(__name__)


"""
NOTES:
    The function map for the photon conservation model looks like:

    wrapper.run_lightcone/coeval()
        setup_photon_cons_correction()
            calibrate_photon_cons_correction()
                _init_photon_conservation_correction() --> computes and stores global evolution
                -->perfoms calibration simulation
                _calibrate_photon_conservation_correction() --> stores calibration evolution
            IF PHOTON_CONS_TYPE==1:
                lib.ComputeIonizedBox()
                    lib.adjust_redshifts_for_photoncons()
                        lib.determine_deltaz_for_photoncons() (first time only) --> calculates the deltaz array with some smoothing
                        --> does more smoothing and returns the adjusted redshift
            ELIF PHOTON_CONS_TYPE==1:
                photoncons_alpha() --> computes and stores ALPHA_ESC shift vs global neutral fraction
                lib.ComputeIonizedBox()
                    get_fesc_fit() --> applies the fit to the current redshift
            ELIF PHOTON_CONS_TYPE==2:
                photoncons_fesc() --> computes and stores F_ESC10 shift vs global neutral fraction
                lib.ComputeIonizedBox()
                    get_fesc_fit() --> applies the fit to the current redshift

"""


def _init_photon_conservation_correction(
    *, user_params=None, cosmo_params=None, astro_params=None, flag_options=None
):
    # This function calculates the global expected evolution of reionisation and saves
    #   it to C global arrays z_Q and Q_value (as well as other non-global confusingly named arrays),
    #   and constructs a GSL interpolator z_at_Q_spline
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    astro_params = AstroParams(astro_params)
    flag_options = FlagOptions(flag_options)

    return lib.InitialisePhotonCons(
        user_params(), cosmo_params(), astro_params(), flag_options()
    )


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
    #   Set by neutral fraction global_params.PhotonConsStart
    from .wrapper import _call_c_simple

    return _call_c_simple(lib.ComputeZstart_PhotonCons)


def _get_photon_nonconservation_data():
    """
    Access C global data representing the photon-nonconservation corrections.

    .. note::  if not using ``PHOTON_CONS`` (in :class:`~FlagOptions`), *or* if the
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
    if not lib.photon_cons_allocated:
        return None

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

    return {name: d[:index] for name, d, index in zip(data_list, data, ArrayIndices)}


def setup_photon_cons(
    astro_params,
    flag_options,
    regenerate,
    hooks,
    direc,
    init_boxes=None,
    user_params=None,
    cosmo_params=None,
    **global_kwargs,
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
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.
    astro_params : :class:`~AstroParams`, optional
        Defines the astrophysical parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    flag_options: :class:`~FlagOptions`, optional
        Options concerning how the reionization process is run, eg. if spin temperature
        fluctuations are required.
    init_boxes : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not be
        re-calculated.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Other Parameters
    ----------------
    regenerate, write
        See docs of :func:`initial_conditions` for more information.
    """
    from .wrapper import _get_config_options

    direc, regenerate, hooks = _get_config_options(direc, regenerate, None, hooks)

    if flag_options.PHOTON_CONS_TYPE == 0:
        return

    if init_boxes is not None:
        cosmo_params = init_boxes.cosmo_params
        user_params = init_boxes.user_params

    if cosmo_params is None or user_params is None:
        raise ValueError(
            "user_params and cosmo_params must be given if init_boxes is not"
        )

    # calculate global and calibration simulation xH histories and save them in C
    calibrate_photon_cons(
        astro_params=astro_params,
        flag_options=flag_options,
        regenerate=regenerate,
        hooks=hooks,
        direc=direc,
        init_boxes=init_boxes,
        user_params=user_params,
        cosmo_params=cosmo_params,
        **global_kwargs,
    )

    # The PHOTON_CONS_TYPE == 1 case is handled in C (for now....), but we get the data anyway
    if flag_options.PHOTON_CONS_TYPE == 1:
        photoncons_data = _get_photon_nonconservation_data()

    if flag_options.PHOTON_CONS_TYPE == 2:
        photoncons_data = photoncons_alpha(
            cosmo_params, user_params, astro_params, flag_options
        )

    if flag_options.PHOTON_CONS_TYPE == 3:
        photoncons_data = photoncons_fesc(
            cosmo_params, user_params, astro_params, flag_options
        )

    return photoncons_data


def calibrate_photon_cons(
    astro_params,
    flag_options,
    regenerate,
    hooks,
    direc,
    init_boxes=None,
    user_params=None,
    cosmo_params=None,
    **global_kwargs,
):
    r"""
    Performs the photon conservation calibration simulation.

    Parameters
    ----------
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.
    astro_params : :class:`~AstroParams`, optional
        Defines the astrophysical parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    flag_options: :class:`~FlagOptions`, optional
        Options concerning how the reionization process is run, eg. if spin temperature
        fluctuations are required.
    init_boxes : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not be
        re-calculated.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Other Parameters
    ----------------
    regenerate, write
        See docs of :func:`initial_conditions` for more information.
    """
    # avoiding circular imports by importing here
    from .wrapper import ionize_box, perturb_field

    with global_params.use(**global_kwargs):
        # Create a new astro_params and flag_options just for the photon_cons correction
        astro_params_photoncons = deepcopy(astro_params)
        astro_params_photoncons._R_BUBBLE_MAX = astro_params.R_BUBBLE_MAX

        flag_options_photoncons = FlagOptions(
            USE_MASS_DEPENDENT_ZETA=flag_options.USE_MASS_DEPENDENT_ZETA,
            M_MIN_in_Mass=flag_options.M_MIN_in_Mass,
        )

        ib = None
        prev_perturb = None

        # Arrays for redshift and neutral fraction for the calibration curve
        z_for_photon_cons = []
        neutral_fraction_photon_cons = []

        # Initialise the analytic expression for the reionisation history
        logger.info("About to start photon conservation correction")
        _init_photon_conservation_correction(
            user_params=user_params,
            cosmo_params=cosmo_params,
            astro_params=astro_params,
            flag_options=flag_options,
        )

        # Determine the starting redshift to start scrolling through to create the
        # calibration reionisation history
        logger.info("Calculating photon conservation zstart")
        z = _calc_zstart_photon_cons()

        while z > global_params.PhotonConsEndCalibz:
            # Determine the ionisation box with recombinations, spin temperature etc.
            # turned off.
            this_perturb = perturb_field(
                redshift=z,
                init_boxes=init_boxes,
                regenerate=regenerate,
                hooks=hooks,
                direc=direc,
            )

            ib2 = ionize_box(
                redshift=z,
                previous_ionize_box=ib,
                init_boxes=init_boxes,
                perturbed_field=this_perturb,
                previous_perturbed_field=prev_perturb,
                astro_params=astro_params_photoncons,
                flag_options=flag_options_photoncons,
                spin_temp=None,
                regenerate=regenerate,
                hooks=hooks,
                direc=direc,
            )

            mean_nf = np.mean(ib2.xH_box)

            # Save mean/global quantities
            neutral_fraction_photon_cons.append(mean_nf)
            z_for_photon_cons.append(z)

            # Can speed up sampling in regions where the evolution is slower
            if 0.3 < mean_nf <= 0.9:
                z -= 0.15
            elif 0.01 < mean_nf <= 0.3:
                z -= 0.05
            else:
                z -= 0.5

            ib = ib2
            if flag_options.USE_MINI_HALOS:
                prev_perturb = this_perturb

        z_for_photon_cons = np.array(z_for_photon_cons[::-1])
        neutral_fraction_photon_cons = np.array(neutral_fraction_photon_cons[::-1])

        # Construct the spline for the calibration curve
        logger.info("Calibrating photon conservation correction")
        _calibrate_photon_conservation_correction(
            redshifts_estimate=z_for_photon_cons,
            nf_estimate=neutral_fraction_photon_cons,
            NSpline=len(z_for_photon_cons),
        )


# (Jdavies): I needed a function to access the delta z from the wrapper
# get_photoncons_data does not have the edge cases that adjust_redshifts_for_photoncons does
def get_photoncons_dz(astro_params, flag_options, redshift):
    """Accesses the delta z arrays from the photon conservation model in C."""
    deltaz = np.zeros(1).astype("f4")
    redshift_pc_in = np.array([redshift]).astype("f4")
    stored_redshift_pc_in = np.array([redshift]).astype("f4")
    lib.adjust_redshifts_for_photoncons(
        astro_params(),
        flag_options(),
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
def photoncons_alpha(cosmo_params, user_params, astro_params, flag_options):
    """The Simpler photons conservation model using ALPHA_ESC, which adjusts the slope of the escape fraction instead of redshifts to match a global evolution."""
    # HACK: I need to allocate the deltaz arrays so I can return the other ones properly, this isn't a great solution
    # TODO: Move the deltaz interp tables to python
    if not lib.photon_cons_allocated:
        lib.determine_deltaz_for_photoncons()
        lib.photon_cons_allocated = ffi.cast("bool", True)

    # Q(analytic) limits to fit the curve
    max_q_fit = 0.99
    min_q_fit = 0.2

    ref_pc_data = _get_photon_nonconservation_data()
    z = ref_pc_data["z_calibration"]
    alpha_arr = (
        np.linspace(-2.0, 1.0, num=31) + astro_params.ALPHA_ESC
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
        astro_params_photoncons = deepcopy(astro_params)
        astro_params_photoncons.ALPHA_ESC = a

        # find the analytic curve wth that alpha
        # TODO: Theres a small memory leak here since global arrays are allocated (for some reason)
        # TODO: use ffi to free them?
        #       This will be fixed by moving the photoncons to python
        _init_photon_conservation_correction(
            user_params=user_params,
            cosmo_params=cosmo_params,
            astro_params=astro_params_photoncons,
            flag_options=flag_options,
        )

        # save it
        pcd_buf = _get_photon_nonconservation_data()

        # interpolate to the calibration redshifts
        test_pc_data[i, ...] = np.interp(
            ref_pc_data["z_calibration"], pcd_buf["z_analytic"], pcd_buf["Q_analytic"]
        )

    # filling factors sometimes go above 1, this causes problems in late-time ratios
    # I want this in the test alphas to get the right photon ratio, but not in the reference analytic
    # test_pc_data[test_pc_data > 1.] = 1.
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
    ):
        last_alpha = astro_params.ALPHA_ESC
        # logger.info('calculating alpha roots')
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
            # logger.info(f'roots at alpha={x1} ')

            # choose the root which gives the smoothest alpha vs z curve
            # arr_out[i] = guesses[np.argmin(np.fabs(guesses - astro_params.ALPHA_ESC))]
            arr_out[i] = guesses[np.argmin(np.fabs(guesses - last_alpha))]
            last_alpha = arr_out[i]

    # initialise the output structure before the fits
    results = {
        "z_cal": ref_pc_data["z_calibration"],
        "Q_ana": ref_interp,
        "Q_alpha": test_pc_data,
        "Q_cal": (1 - ref_pc_data["nf_calibration"]),
        "alpha_ratio": alpha_estimate_ratio,
        "alpha_diff": alpha_estimate_diff,
        "alpha_reverse": alpha_estimate_reverse,
        "alpha_arr": alpha_arr,
        "fit_yint": astro_params.ALPHA_ESC,
        "fit_slope": 0,  # start with no correction
        "found_alpha": False,
    }

    # adjust the reverse one (we found the alpha which is close to the calibration sim, undo it)
    alpha_estimate_reverse = 2 * astro_params.ALPHA_ESC - alpha_estimate_reverse

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
        popt, pcov = curve_fit(alpha_func, ref_interp[sel], fit_alpha[sel])
        # pass to C
        logger.info(f"ALPHA_ESC Original = {astro_params.ALPHA_ESC:.3f}")
        logger.info(f"Running with ALPHA_ESC = {popt[0]:.2f} + {popt[1]:.2f} * Q")

        results["fit_yint"] = popt[0]
        results["fit_slope"] = popt[1]
        results["found_alpha"] = True

    lib.set_alphacons_params(results["fit_yint"], results["fit_slope"])

    return results


def photoncons_fesc(cosmo_params, user_params, astro_params, flag_options):
    """The Even Simpler photon conservation model using F_ESC10.

    Adjusts the normalisation of the escape fraction to match a global evolution.
    """
    # HACK: I need to allocate the deltaz arrays so I can return the other ones properly, this isn't a great solution
    if not lib.photon_cons_allocated:
        lib.determine_deltaz_for_photoncons()
        lib.photon_cons_allocated = ffi.cast("bool", True)

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

    # filling factors sometimes go above 1, this causes problems in late-time ratios
    # I want this in the test alphas to get the right photon ratio, but not in the reference analytic
    # test_pc_data[test_pc_data > 1.] = 1.
    ref_interp[ref_interp > 1] = 1.0

    # ratio of each alpha with calibration
    ratio_ref = ref_interp / (1 - ref_pc_data["nf_calibration"])

    fit_fesc = ratio_ref * 10**astro_params.F_ESC10
    sel = np.isfinite(fit_fesc) & (ref_interp < max_q_fit) & (ref_interp > min_q_fit)

    popt, pcov = curve_fit(alpha_func, ref_interp[sel], fit_fesc[sel])
    # pass to C
    logger.info(f"F_ESC10 Original = {10**astro_params.F_ESC10:.3f}")
    logger.info(f"Running with F_ESC10 = {popt[0]:.2f} + {popt[1]:.2f} * Q")

    # initialise the output structure before the fits
    results = {
        "z_cal": ref_pc_data["z_calibration"],
        "Q_ana": ref_interp,
        "Q_cal": (1 - ref_pc_data["nf_calibration"]),
        "Q_ratio": ratio_ref,
        "fit_target": fit_fesc,
        "fit_yint": popt[0],
        "fit_slope": popt[1],  # start with no correction
    }

    lib.set_alphacons_params(results["fit_yint"], results["fit_slope"])

    return results
