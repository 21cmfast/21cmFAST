"""Function to estimate total memory usage."""

import logging
import numpy as np
from copy import deepcopy

from .inputs import AstroParams, CosmoParams, FlagOptions, UserParams, global_params
from .outputs import (
    BrightnessTemp,
    InitialConditions,
    IonizedBox,
    PerturbedField,
    TsBox,
)
from .wrapper import _logscroll_redshifts, _setup_lightcone

logger = logging.getLogger("21cmFAST")

# Constants defining interpolation table lengths (from C side)
# Not ideal to be here, but unlikely to ever change (names in lower case)
nmass = 300.0
nsfr_high = 200.0
nsfr_low = 250.0
ngl_sfr = 100.0
nmturn = 50.0

zpp_interp_points_sfr = 400.0
dens_ninterp = 400.0
erfc_num_points = 10000.0

x_int_nxhii = 14.0
rr_z_npts = 300.0
rr_lngamma_npts = 250.0


def estimate_memory_coeval(
    *,
    redshift=None,
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
):
    r"""
    Compute an estimate of the requisite memory needed by the user for a run_coeval call.

    Note, this is an upper-limit as it assumes all requisite data needs to be generated.
    Actual memory usage may be less than this estimate if using pre-computed data.

    Parameters
    ----------
    redshift : array_like
        A single redshift, or multiple redshift, at which to return results. The minimum of these
        will define the log-scrolling behaviour (if necessary).
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.
    astro_params : :class:`~AstroParams`, optional
        Defines the astrophysical parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    flag_options : :class:`~FlagOptions`, optional
        Options concerning how the reionization process is run, eg. if spin temperature
        fluctuations are required.

    Returns
    -------
    dict :
        ics_python: Estimate of python allocated memory for initial conditions (in Bytes)
        ics_c: Estimate of C allocated memory for initial conditions (in Bytes)
        pf_python: Estimate of python allocated memory for a single perturb field (in Bytes)
        pf_c: Estimate of C allocated memory for a single perturb field (in Bytes)
        hf_python: Estimate of python allocated memory for determine halo list (in Bytes)
        hf_c: Estimate of C allocated memory for determine halo list (in Bytes)
        phf_python: Estimate of python allocated memory for a single perturb halo list (in Bytes)
        phf_c: Estimate of C allocated memory for a single perturb halo list (in Bytes)
        ib_python: Estimate of python allocated memory for a single ionized box (in Bytes)
        ib_c: Estimate of C allocated memory for a single ionized box (in Bytes)
        st_python: Estimate of python allocated memory for a single spin temperature box (in Bytes)
        st_c_init: Estimate of retained memory for any spin temperature box (in Bytes)
        st_c_per_z: Estimate of C allocated memory for a single spin temperature box (in Bytes)
        bt_python: Estimate of python allocated memory for a single brightness temperature box (in Bytes)
        bt_c: Estimate of C allocated memory for a single brightness temperature box (in Bytes)
        peak_memory: As estimate of the peak memory usage for running a lightcone (generating all data) (in Bytes)

    """
    # Deal with AstroParams and INHOMO_RECO
    astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

    # First, calculate the memory usage for the initial conditions
    memory_ics = mem_initial_conditions(user_params=user_params)

    memory_data = {"ics_%s" % k: memory_ics[k] for k in memory_ics.keys()}

    # Maximum memory while running ICs
    peak_memory = memory_ics["c"] + memory_ics["python"]

    # Now the perturb field
    memory_pf = mem_perturb_field(user_params=user_params)

    memory_data.update({"pf_%s" % k: memory_pf[k] for k in memory_pf.keys()})

    # If we are using the photon non-conservation correction
    if flag_options.PHOTON_CONS:
        # First need to create new structs for photon the photon-conservation
        astro_params_photoncons = deepcopy(astro_params)

        flag_options_photoncons = FlagOptions(
            USE_MASS_DEPENDENT_ZETA=flag_options.USE_MASS_DEPENDENT_ZETA,
            M_MIN_in_Mass=flag_options.M_MIN_in_Mass,
            USE_VELS_AUX=user_params.USE_RELATIVE_VELOCITIES,
        )

        # First perturb_field
        memory_pf = mem_perturb_field(user_params=user_params)

        # First ionize_box
        memory_ib = mem_ionize_box(
            user_params=user_params,
            astro_params=astro_params_photoncons,
            flag_options=flag_options_photoncons,
        )

        # As we iterate through we storing the python memory of two
        # perturb_field and ionize_boxes plus the C memory of either
        # of the ionize_box or perturb field as it is being calculated
        peak_memory_photoncons = memory_ics[
            "python"
        ]  # We have the initial conditions in memory
        peak_memory_photoncons += 2 * (
            memory_pf["python"] + memory_ib["python"]
        )  # The python memory
        peak_memory_photoncons += (
            memory_pf["c"] if memory_pf["c"] > memory_ib["c"] else memory_ib["c"]
        )  # Maximum C memory, as it is freed after usage

        # Check if the memory required to do the photon non-conservation correction exceeds
        # current peak memory usage
        peak_memory = (
            peak_memory
            if peak_memory > peak_memory_photoncons
            else peak_memory_photoncons
        )

    # Determine the number of redshifts passed
    if not hasattr(redshift, "__len__"):
        redshift = [redshift]

    n_redshifts = len(redshift)

    current_memory = memory_ics["python"]  # We have the initial conditions in memory
    current_memory += (
        n_redshifts * memory_pf["python"]
    )  # Python memory for all perturb fields
    current_memory += memory_pf["c"]  # Plus C memory for a single perturb field

    # Check if running perturb_field requires more memory than generating ICs
    peak_memory = peak_memory if peak_memory > current_memory else current_memory

    # Now start generating estimates for all other data products

    # Calculate the memory for a determine_halo_list call
    memory_hf = mem_halo_field(user_params=user_params)

    memory_data.update({"hf_%s" % k: memory_hf[k] for k in memory_hf.keys()})

    # Calculate the memory for a perturb_halo_list call
    memory_phf = mem_perturb_halo(user_params=user_params)

    memory_data.update({"phf_%s" % k: memory_phf[k] for k in memory_phf.keys()})

    # Calculate the memory for an ionize_box call
    memory_ib = mem_ionize_box(
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )

    memory_data.update({"ib_%s" % k: memory_ib[k] for k in memory_ib.keys()})

    # Calculate the memory for a spin_temperature call
    memory_st = mem_spin_temperature(
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )

    memory_data.update({"st_%s" % k: memory_st[k] for k in memory_st.keys()})

    # Calculate the memory for a brightness_temperature call
    memory_bt = mem_brightness_temperature(user_params=user_params)

    memory_data.update({"bt_%s" % k: memory_bt[k] for k in memory_bt.keys()})

    # All the data kept in memory at this point in Python
    stored_memory = memory_ics["python"]  # We have the initial conditions in memory
    stored_memory += (
        n_redshifts * memory_pf["python"]
    )  # Python memory for all perturb fields

    current_memory = stored_memory

    # If using haloes, then need to add all of these too.
    if flag_options.USE_HALO_FIELD:
        current_memory += n_redshifts * memory_phf["python"] + memory_hf["python"]

        if memory_phf["c"] > memory_hf["c"]:
            current_memory += memory_phf["c"]
        else:
            current_memory += memory_hf["c"]

        # Check if running perturb_halos requires more memory than generating ICs
        peak_memory = peak_memory if peak_memory > current_memory else current_memory

        # We keep all the python memory for the haloes
        stored_memory += n_redshifts * memory_phf["python"] + memory_hf["python"]

    if flag_options.USE_MINI_HALOS:
        # If using minihaloes we (might) need 2 perturbs, so two python + one in C
        current_memory = stored_memory + 2.0 * memory_pf["python"] + memory_pf["c"]

        # Check if our peak memory usage has been exceeded
        peak_memory = peak_memory if peak_memory > current_memory else current_memory

        stored_memory += 2.0 * memory_pf["python"]  # We (might) need two perturb's

    # Add the two ionized boxes
    stored_memory += 2.0 * memory_ib["python"]

    # Add (if necessary) the two spin temperature boxes
    if flag_options.USE_TS_FLUCT:
        stored_memory += 2.0 * memory_st["python"]

        # We also have initialied C memory that is retained until the end of the calculation
        stored_memory += memory_st["c_init"]

    # Add the two brightness_temperature boxes
    stored_memory += 2.0 * memory_bt["python"]

    # Now we have an estimate for the data retained in memory, now we just need to check
    # the peak memory usage (which includes the additional C memory that is allocated
    # and then freed on a single redshift call)
    # First check perturb field
    current_memory = stored_memory + memory_pf["c"]  # Add the temporary C memory

    # Check if our peak memory usage has been exceeded
    peak_memory = peak_memory if peak_memory > current_memory else current_memory

    # Check spin temperature
    if flag_options.USE_TS_FLUCT:
        current_memory = stored_memory + memory_st["c_per_z"]

        # Check if our peak memory usage has been exceeded
        peak_memory = peak_memory if peak_memory > current_memory else current_memory

    # Check ionized box
    current_memory = stored_memory + memory_ib["c"]

    # Check if our peak memory usage has been exceeded
    peak_memory = peak_memory if peak_memory > current_memory else current_memory

    # Check brightness temperature
    current_memory = stored_memory + memory_bt["c"]

    # Check if our peak memory usage has been exceeded
    peak_memory = peak_memory if peak_memory > current_memory else current_memory

    # I have been finding that remaining memory (variables etc.) consistently appear
    # to contribute about 0.1 Gb. So I'll add this amount. Only important for small
    # box sizes.
    peak_memory += 0.1 * 1024**3

    memory_data.update({"peak_memory": peak_memory})

    print_memory_estimate(
        memory_data=memory_data,
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )

    return memory_data


def estimate_memory_lightcone(
    *,
    redshift=None,
    max_redshift=None,
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
    lightcone_quantities=("brightness_temp",),
):
    r"""
    Compute an estimate of the requisite memory needed by the user for a run_lightcone call.

    Note, this is an upper-limit as it assumes all requisite data needs to be generated.
    Actual memory usage may be less than this estimate if using pre-computed data.

    Parameters
    ----------
    redshift : float
        The minimum redshift of the lightcone.
    max_redshift : float, optional
        The maximum redshift at which to keep lightcone information. By default, this is equal to
        `z_heat_max`. Note that this is not *exact*, but will be typically slightly exceeded.
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.
    astro_params : :class:`~AstroParams`, optional
        Defines the astrophysical parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    flag_options : :class:`~FlagOptions`, optional
        Options concerning how the reionization process is run, eg. if spin temperature
        fluctuations are required.
    lightcone_quantities : tuple of str, optional
        The quantities to form into a lightcone. By default, just the brightness
        temperature. Note that these quantities must exist in one of the output
        structures:

        * :class:`~InitialConditions`
        * :class:`~PerturbField`
        * :class:`~TsBox`
        * :class:`~IonizedBox`
        * :class:`BrightnessTemp`

        To get a full list of possible quantities, run :func:`get_all_fieldnames`.

    Returns
    -------
    dict :
        ics_python: Estimate of python allocated memory for initial conditions (in Bytes)
        ics_c: Estimate of C allocated memory for initial conditions (in Bytes)
        pf_python: Estimate of python allocated memory for a single perturb field (in Bytes)
        pf_c: Estimate of C allocated memory for a single perturb field (in Bytes)
        hf_python: Estimate of python allocated memory for determine halo list (in Bytes)
        hf_c: Estimate of C allocated memory for determine halo list (in Bytes)
        phf_python: Estimate of python allocated memory for a single perturb halo list (in Bytes)
        phf_c: Estimate of C allocated memory for a single perturb halo list (in Bytes)
        ib_python: Estimate of python allocated memory for a single ionized box (in Bytes)
        ib_c: Estimate of C allocated memory for a single ionized box (in Bytes)
        st_python: Estimate of python allocated memory for a single spin temperature box (in Bytes)
        st_c_init: Estimate of retained memory for any spin temperature box (in Bytes)
        st_c_per_z: Estimate of C allocated memory for a single spin temperature box (in Bytes)
        bt_python: Estimate of python allocated memory for a single brightness temperature box (in Bytes)
        bt_c: Estimate of C allocated memory for a single brightness temperature box (in Bytes)
        python_lc: Estimate of python allocated memory for all provided lightcones (in Bytes)
        peak_memory: As estimate of the peak memory usage for running a lightcone (generating all data) (in Bytes)

    """
    # Deal with AstroParams and INHOMO_RECO
    astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

    # First, calculate the memory usage for the initial conditions
    memory_ics = mem_initial_conditions(user_params=user_params)

    memory_data = {"ics_%s" % k: memory_ics[k] for k in memory_ics.keys()}

    # Maximum memory while running ICs
    peak_memory = memory_ics["c"] + memory_ics["python"]

    # Now the perturb field
    memory_pf = mem_perturb_field(user_params=user_params)

    memory_data.update({"pf_%s" % k: memory_pf[k] for k in memory_pf.keys()})

    # Stored ICs in python + allocated C and Python memory for perturb_field
    current_memory = memory_ics["python"] + memory_pf["python"] + memory_pf["c"]

    # Check if running perturb_field requires more memory than generating ICs
    peak_memory = peak_memory if peak_memory > current_memory else current_memory

    # If we are using the photon non-conservation correction
    if flag_options.PHOTON_CONS:
        # First need to create new structs for photon the photon-conservation
        astro_params_photoncons = deepcopy(astro_params)

        flag_options_photoncons = FlagOptions(
            USE_MASS_DEPENDENT_ZETA=flag_options.USE_MASS_DEPENDENT_ZETA,
            M_MIN_in_Mass=flag_options.M_MIN_in_Mass,
            USE_VELS_AUX=user_params.USE_RELATIVE_VELOCITIES,
        )

        # First perturb_field
        memory_pf = mem_perturb_field(user_params=user_params)

        # First ionize_box
        memory_ib = mem_ionize_box(
            user_params=user_params,
            astro_params=astro_params_photoncons,
            flag_options=flag_options_photoncons,
        )

        # As we iterate through we storing the python memory of two
        # perturb_field and ionize_boxes plus the C memory of either
        # of the ionize_box or perturb field as it is being calculated
        peak_memory_photoncons = memory_ics[
            "python"
        ]  # We have the initial conditions in memory
        peak_memory_photoncons += 2 * (
            memory_pf["python"] + memory_ib["python"]
        )  # The python memory
        peak_memory_photoncons += (
            memory_pf["c"] if memory_pf["c"] > memory_ib["c"] else memory_ib["c"]
        )  # Maximum C memory, as it is freed after usage

        # Check if the memory required to do the photon non-conservation correction exceeds
        # current peak memory usage
        peak_memory = (
            peak_memory
            if peak_memory > peak_memory_photoncons
            else peak_memory_photoncons
        )

    # Now need to determine the size of the light-cone and how many types are to be stored in memory.
    # Below are taken from the run_lightcone function.
    # Determine the maximum redshift (starting point) for the light-cone.
    max_redshift = (
        global_params.Z_HEAT_MAX
        if (
            flag_options.INHOMO_RECO
            or flag_options.USE_TS_FLUCT
            or max_redshift is None
        )
        else max_redshift
    )

    # Get the redshift through which we scroll and evaluate the ionization field.
    scrollz = _logscroll_redshifts(
        redshift, global_params.ZPRIME_STEP_FACTOR, max_redshift
    )

    # Obtain the size of the light-cone object (n_lightcone)
    d_at_redshift, lc_distances, n_lightcone = _setup_lightcone(
        cosmo_params,
        max_redshift,
        redshift,
        scrollz,
        user_params,
        global_params.ZPRIME_STEP_FACTOR,
    )

    # Total number of light-cones to be stored in memory
    num_lightcones = len(lightcone_quantities)

    # Calculate memory footprint for all light-cones
    size_lightcones = n_lightcone * (user_params.HII_DIM) ** 2
    size_lightcones = (np.float32(1.0).nbytes) * num_lightcones * size_lightcones

    memory_data.update({"python_lc": size_lightcones})

    # All the data kept in memory at this point in Python
    current_memory = memory_ics["python"] + memory_pf["python"] + size_lightcones

    # Check if we now exceed the peak memory usage thus far
    peak_memory = peak_memory if peak_memory > current_memory else current_memory

    # Now start generating the data to populate the light-cones

    # Calculate the memory for a determine_halo_list call
    memory_hf = mem_halo_field(user_params=user_params)

    memory_data.update({"hf_%s" % k: memory_hf[k] for k in memory_hf.keys()})

    # Calculate the memory for a perturb_halo_list call
    memory_phf = mem_perturb_halo(user_params=user_params)

    memory_data.update({"phf_%s" % k: memory_phf[k] for k in memory_phf.keys()})

    # Calculate the memory for an ionize_box call
    memory_ib = mem_ionize_box(
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )

    memory_data.update({"ib_%s" % k: memory_ib[k] for k in memory_ib.keys()})

    # Calculate the memory for a spin_temperature call
    memory_st = mem_spin_temperature(
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )

    memory_data.update({"st_%s" % k: memory_st[k] for k in memory_st.keys()})

    # Calculate the memory for a brightness_temperature call
    memory_bt = mem_brightness_temperature(user_params=user_params)

    memory_data.update({"bt_%s" % k: memory_bt[k] for k in memory_bt.keys()})

    # Now, when calculating the light-cone we always need two concurrent boxes (current and previous redshift)

    # This is the all the data currently in memory at this point
    stored_memory = current_memory  # Corresponds to ICs, one perturb_field and the allocated light-cones

    # Add second perturb_field
    stored_memory += memory_pf["python"]

    # Add the memory from perturb_haloes
    if flag_options.USE_HALO_FIELD:
        # We'll have two copies of this at any one time, one of the original haloes
        # and one for the current redshift we are evaluating
        # Note this is approximate as we can only guess the total memory due to the
        # fact that the amount of memory required is proportional to the number of haloes
        # found.
        stored_memory += 2.0 * memory_phf["c"]

    # Add the two ionized boxes
    stored_memory += 2.0 * memory_ib["python"]

    # Add (if necessary) the two spin temperature boxes
    if flag_options.USE_TS_FLUCT:
        stored_memory += 2.0 * memory_st["python"]

        # We also have initialied C memory that is retained until the end of the calculation
        stored_memory += memory_st["c_init"]

    # Add the two brightness_temperature boxes
    stored_memory += 2.0 * memory_bt["python"]

    # Now we have an estimate for the data retained in memory, now we just need to check
    # the peak memory usage (which includes the additional C memory that is allocated
    # and then freed on a single redshift call)
    # First check perturb field
    current_memory = stored_memory + memory_pf["c"]  # Add the temporary C memory

    # Check if our peak memory usage has been exceeded
    peak_memory = peak_memory if peak_memory > current_memory else current_memory

    # Check spin temperature
    if flag_options.USE_TS_FLUCT:
        current_memory = stored_memory + memory_st["c_per_z"]

        # Check if our peak memory usage has been exceeded
        peak_memory = peak_memory if peak_memory > current_memory else current_memory

    # Check ionized box
    current_memory = stored_memory + memory_ib["c"]

    # Check if our peak memory usage has been exceeded
    peak_memory = peak_memory if peak_memory > current_memory else current_memory

    # Check brightness temperature
    current_memory = stored_memory + memory_bt["c"]

    # Check if our peak memory usage has been exceeded
    peak_memory = peak_memory if peak_memory > current_memory else current_memory

    # I have been finding that remaining memory (variables etc.) consistently appear
    # to contribute about 0.1 Gb. So I'll add this amount. Only important for small
    # box sizes.
    peak_memory += 0.1 * 1024**3

    memory_data.update({"peak_memory": peak_memory})

    print_memory_estimate(
        memory_data=memory_data,
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )

    return memory_data


def estimate_memory_ics(
    *,
    user_params=None,
):
    r"""
    Compute an estimate of the requisite memory needed just for the initial conditions.

    Note, this is an upper-limit

    Parameters
    ----------
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.

    Returns
    -------
    dict :
        ics_python: Estimate of python allocated memory for initial conditions (in Bytes)
        ics_c: Estimate of C allocated memory for initial conditions (in Bytes)
        peak_memory: As estimate of the peak memory usage for running a lightcone (generating all data) (in Bytes)
    """
    # Calculate the memory usage for the initial conditions
    memory_ics = mem_initial_conditions(user_params=user_params)

    memory_data = {"ics_%s" % k: memory_ics[k] for k in memory_ics.keys()}

    # Maximum memory while running ICs
    peak_memory = memory_ics["c"] + memory_ics["python"]

    # I have been finding that remaining memory (variables etc.) consistently appear
    # to contribute about 0.1 Gb. So I'll add this amount. Only important for small
    # box sizes.
    peak_memory += 0.1 * 1024**3

    memory_data.update({"peak_memory": peak_memory})

    print_memory_estimate(
        memory_data=memory_data,
        user_params=user_params,
    )

    return memory_data


def estimate_memory_perturb(
    *,
    user_params=None,
):
    r"""
    Compute an estimate of the requisite memory needed by the user for a single perturb_field call.

    Note, this is an upper-limit

    Parameters
    ----------
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.

    Returns
    -------
    dict :
        ics_python: Estimate of python allocated memory for initial conditions (in Bytes)
        ics_c: Estimate of C allocated memory for initial conditions (in Bytes)
        pf_python: Estimate of python allocated memory for a single perturb field (in Bytes)
        pf_c: Estimate of C allocated memory for a single perturb field (in Bytes)
        peak_memory: As estimate of the peak memory usage for running a lightcone (generating all data) (in Bytes)

    """
    # First, calculate the memory usage for the initial conditions
    memory_ics = mem_initial_conditions(user_params=user_params)

    memory_data = {"ics_%s" % k: memory_ics[k] for k in memory_ics.keys()}

    # Maximum memory while running ICs
    peak_memory = memory_ics["c"] + memory_ics["python"]

    # Now the perturb field
    memory_pf = mem_perturb_field(user_params=user_params)

    memory_data.update({"pf_%s" % k: memory_pf[k] for k in memory_pf.keys()})

    # Stored ICs in python + allocated C and Python memory for perturb_field
    current_memory = memory_ics["python"] + memory_pf["python"] + memory_pf["c"]

    # Check if running perturb_field requires more memory than generating ICs
    peak_memory = peak_memory if peak_memory > current_memory else current_memory

    # I have been finding that remaining memory (variables etc.) consistently appear
    # to contribute about 0.1 Gb. So I'll add this amount. Only important for small
    # box sizes.
    peak_memory += 0.1 * 1024**3

    memory_data.update({"peak_memory": peak_memory})

    print_memory_estimate(
        memory_data=memory_data,
        user_params=user_params,
    )

    return memory_data


def mem_initial_conditions(
    *,
    user_params=None,
):
    """A function to estimate total memory usage of an initial_conditions call."""
    # Memory usage of Python InitialConditions class.
    init = InitialConditions(user_params=user_params)
    size_py = 0
    for key in init._array_structure:
        size_py += np.prod(init._array_structure[key])

    # These are all float arrays
    size_py = (np.float32(1.0).nbytes) * size_py

    # Memory usage within GenerateICs
    kspace_num_pixels = (
        float(user_params.DIM) * user_params.NON_CUBIC_FACTOR / 2.0 + 1.0
    ) * (user_params.DIM) ** 2

    # All declared DIM boxes
    # HIRES_box, HIRES_box_saved
    num_c_boxes = 2

    # All declared 2LPT boxes (DIM)
    if user_params.USE_2LPT:
        num_c_boxes += 1

    # These are all fftwf complex arrays (thus 2 * size)
    size_c = (2.0 * (np.float32(1.0).nbytes)) * num_c_boxes * kspace_num_pixels

    # Storage of large number of integers (for seeding in multithreading)
    # Note, this is somewhat hard coded as it is C compiler/architecture specific (INT_MAX)
    size_c_RNG = (
        134217727.0 * 4.0
    )  # First, is INT_MAX/16, factor of 4. is for an unsigned int

    if size_c_RNG > size_c:
        size_c = size_c_RNG

    return {"python": size_py, "c": size_c}


def mem_perturb_field(
    *,
    user_params=None,
    cosmo_params=None,
):
    """A function to estimate total memory usage of a perturb_field call."""
    # Memory usage of Python InitialConditions class. (assume single redshift)
    pt_box = PerturbedField(
        redshift=9.0, user_params=user_params, cosmo_params=cosmo_params
    )
    size_py = 0
    for key in pt_box._array_structure:
        size_py += np.prod(pt_box._array_structure[key])

    # These are all float arrays
    size_py = (np.float32(1.0).nbytes) * size_py

    # Memory usage within PerturbField.c
    kspace_num_pixels = (
        float(user_params.DIM) * user_params.NON_CUBIC_FACTOR / 2.0 + 1.0
    ) * (user_params.DIM) ** 2
    hii_kspace_num_pixels = (
        float(user_params.HII_DIM) * user_params.NON_CUBIC_FACTOR / 2.0 + 1.0
    ) * (user_params.HII_DIM) ** 2

    tot_num_pixels = (
        float(user_params.DIM)
        * user_params.NON_CUBIC_FACTOR
        * float(user_params.DIM) ** 2
    )
    hii_tot_num_pixels = (
        float(user_params.HII_DIM)
        * user_params.NON_CUBIC_FACTOR
        * float(user_params.HII_DIM) ** 2
    )

    # LOWRES_density_perturb, LOWRES_density_perturb_saved
    num_c_boxes_HII_DIM = 2

    # These are all fftwf complex arrays (thus 2 * size)
    size_c = (
        (2.0 * (np.float32(1.0).nbytes)) * num_c_boxes_HII_DIM * hii_kspace_num_pixels
    )

    num_c_boxes_DIM = 0
    if user_params.PERTURB_ON_HIGH_RES:
        # HIRES_density_perturb, HIRES_density_perturb_saved
        num_c_boxes_DIM += 2

        # These are all fftwf complex arrays (thus 2 * size)
        size_c += (2.0 * (np.float32(1.0).nbytes)) * num_c_boxes_DIM * kspace_num_pixels

        # For resampled_box (double)
        size_c += (np.float64(1.0).nbytes) * tot_num_pixels
    else:
        # For resampled_box (double)
        size_c += (np.float64(1.0).nbytes) * hii_tot_num_pixels

    return {"python": size_py, "c": size_c}


def mem_ionize_box(
    *,
    user_params=None,
    astro_params=None,
    flag_options=None,
):
    """A function to estimate total memory usage of an ionize_box call."""
    # Memory usage of Python IonizedBox class.
    ibox = IonizedBox(
        redshift=9.0,
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )
    size_py = 0
    for key in ibox._array_structure:
        if isinstance(
            ibox._array_structure[key], dict
        ):  # It can have embedded dictionaries
            for alt_key in ibox._array_structure[key]:
                if "shape" in alt_key:
                    size_py += np.prod(ibox._array_structure[key][alt_key])
        else:
            size_py += np.prod(ibox._array_structure[key])

    # These are all float arrays
    size_py = (np.float32(1.0).nbytes) * size_py

    # Memory usage within IonisationBox.c
    hii_kspace_num_pixels = (
        float(user_params.HII_DIM) * user_params.NON_CUBIC_FACTOR / 2.0 + 1.0
    ) * (user_params.HII_DIM) ** 2

    # deltax_unfiltered, delta_unfiltered_original, deltax_filtered
    num_c_boxes = 3.0

    if flag_options.USE_MINI_HALOS:
        # prev_deltax_unfiltered, prev_deltax_filtered
        num_c_boxes += 2.0

    if flag_options.USE_TS_FLUCT:
        # xe_unfiltered, xe_filtered
        num_c_boxes += 2.0

    if flag_options.INHOMO_RECO:
        # N_rec_unfiltered, N_rec_filtered
        num_c_boxes += 2.0

    if flag_options.USE_MASS_DEPENDENT_ZETA and flag_options.USE_MINI_HALOS:
        # log10_Mturnover_unfiltered, log10_Mturnover_filtered, log10_Mturnover_MINI_unfiltered, log10_Mturnover_MINI_filtered
        num_c_boxes += 4.0

    if flag_options.USE_HALO_FIELD:
        # M_coll_unfiltered, M_coll_filtered
        num_c_boxes += 2.0

    # Various interpolation tables
    tables_float = tables_double = 0.0
    if flag_options.USE_MASS_DEPENDENT_ZETA:
        tables_float += 2.0 * (ngl_sfr + 1)  # xi_SFR, wi_SFR

        if user_params.USE_INTERPOLATION_TABLES:
            tables_double += nsfr_low  # log10_overdense_spline_SFR,
            tables_float += (
                2.0 * nsfr_high + nsfr_low
            )  # Overdense_spline_SFR, Nion_spline, log10_Nion_spline

            if flag_options.USE_MINI_HALOS:
                tables_double += nsfr_low  # prev_log10_overdense_spline_SFR
                tables_float += nsfr_high  # prev_Overdense_spline_SFR
                tables_float += (
                    4.0 * nsfr_low * nmturn
                )  # log10_Nion_spline, log10_Nion_spline_MINI, prev_log10_Nion_spline, prev_log10_Nion_spline_MINI
                tables_float += (
                    4.0 * nsfr_high * nmturn
                )  # Nion_spline, Nion_spline_MINI, prev_Nion_spline, prev_Nion_spline_MINI

        if flag_options.USE_MINI_HALOS:
            tables_float += 2.0 * nmturn  # Mturns, Mturns_MINI
    else:
        tables_double += 2.0 * erfc_num_points  # ERFC_VALS, ERFC_VALS_DIFF

    if flag_options.INHOMO_RECO:
        tables_double += (
            rr_z_npts * rr_lngamma_npts + rr_lngamma_npts
        )  # RR_table, lnGamma_values

    # These can only exist in ionisation box if spin temperature is not being computed (otherwise it exists there)
    if user_params.USE_INTERPOLATION_TABLES and not flag_options.USE_TS_FLUCT:
        tables_float += (
            3.0 * nmass
        )  # Mass_InterpTable, Sigma_InterpTable, dSigmadm_InterpTable

    # These are all fftwf complex arrays (thus 2 * size)
    size_c = (2.0 * (np.float32(1.0).nbytes)) * num_c_boxes * hii_kspace_num_pixels

    # Now add in the additional interpolation tables
    size_c += (np.float32(1.0).nbytes) * tables_float + (
        np.float64(1.0).nbytes
    ) * tables_double

    return {"python": size_py, "c": size_c}


def mem_spin_temperature(
    *,
    user_params=None,
    astro_params=None,
    flag_options=None,
):
    """A function to estimate total memory usage of a spin_temperature call."""
    # Memory usage of Python TsBox class.
    ts_box = TsBox(
        redshift=9.0,
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )
    size_py = 0
    for key in ts_box._array_structure:
        size_py += np.prod(ts_box._array_structure[key])

    # These are all float arrays
    size_py = (np.float32(1.0).nbytes) * size_py

    # Memory usage within SpinTemperatureBox.c
    hii_kspace_num_pixels = (
        float(user_params.HII_DIM) * user_params.NON_CUBIC_FACTOR / 2.0 + 1.0
    ) * (user_params.HII_DIM) ** 2

    # box, unfiltered_box
    num_c_boxes = 2.0
    num_c_boxes_alt = 0.0
    num_c_boxes_initialised = 0.0

    if flag_options.USE_MINI_HALOS:
        # log10_Mcrit_LW_unfiltered, log10_Mcrit_LW_filtered
        num_c_boxes += 2.0

        # log10_Mcrit_LW
        num_c_boxes_alt += global_params.NUM_FILTER_STEPS_FOR_Ts

    # Now go through and add in all the interpolation table information
    tables_float = tables_double = 0.0

    tables_float += global_params.NUM_FILTER_STEPS_FOR_Ts  # R_values
    tables_double += 2.0 * global_params.NUM_FILTER_STEPS_FOR_Ts  # zpp_edge, sigma_atR

    if user_params.USE_INTERPOLATION_TABLES:
        tables_float += (
            2.0 * global_params.NUM_FILTER_STEPS_FOR_Ts
        )  # min_densities, max_densities
        tables_float += zpp_interp_points_sfr  # zpp_interp_table

    if flag_options.USE_MASS_DEPENDENT_ZETA:
        # delNL0
        if user_params.MINIMIZE_MEMORY:
            num_c_boxes_initialised += 1.0
        else:
            num_c_boxes_initialised += global_params.NUM_FILTER_STEPS_FOR_Ts

        # del_fcoll_Rct, m_xHII_low_box, inverse_val_box
        num_c_boxes_initialised += (
            3.0  # m_xHII_low_box is an int, but I'll count it as 4 bytes just in case
        )

        # dxheat_dt_box, dxion_source_dt_box, dxlya_dt_box, dstarlya_dt_box
        num_c_boxes_initialised += 2.0 * 4.0  # factor of 2. as these are doubles
        if flag_options.USE_LYA_HEATING:
            num_c_boxes_initialised += (
                2.0 * 2.0
            )  # dstarlya_cont_dt_box, dstarlya_inj_dt_box

        tables_float += global_params.NUM_FILTER_STEPS_FOR_Ts  # SFR_timescale_factor
        tables_double += 2.0 * (ngl_sfr + 1.0)  # xi_SFR_Xray, wi_SFR_Xray

        if user_params.USE_INTERPOLATION_TABLES:
            tables_double += (
                nsfr_low + nsfr_high
            )  # overdense_low_table, overdense_high_table

            tables_float += global_params.NUM_FILTER_STEPS_FOR_Ts * (
                nsfr_low + nsfr_high
            )  # log10_SFRD_z_low_table, SFRD_z_high_table

        if flag_options.USE_MINI_HALOS:
            # del_fcoll_Rct_MINI
            num_c_boxes_initialised += 1.0

            # dstarlyLW_dt_box, dxheat_dt_box_MINI, dxion_source_dt_box_MINI
            # dxlya_dt_box_MINI, dstarlya_dt_box_MINI, dstarlyLW_dt_box_MINI
            num_c_boxes_initialised += (
                2.0 * 6.0
            )  # factor of 2. taking into account that these are doubles
            if flag_options.USE_LYA_HEATING:
                num_c_boxes_initialised += (
                    2.0 * 2.0
                )  # dstarlya_cont_dt_box_MINI, dstarlya_inj_dt_box_MINI

            tables_double += (
                global_params.NUM_FILTER_STEPS_FOR_Ts
            )  # log10_Mcrit_LW_ave_list

            if user_params.USE_INTERPOLATION_TABLES:
                # log10_SFRD_z_low_table_MINI, SFRD_z_high_table_MINI
                tables_float += (
                    global_params.NUM_FILTER_STEPS_FOR_Ts
                    * nmturn
                    * (nsfr_low + nsfr_high)
                )

    else:
        # delNL0_rev
        num_c_boxes_initialised += global_params.NUM_FILTER_STEPS_FOR_Ts

        if user_params.USE_INTERPOLATION_TABLES:
            tables_double += zpp_interp_points_sfr  # Sigma_Tmin_grid

            # fcoll_R_grid, dfcoll_dz_grid
            tables_double += 2.0 * (
                global_params.NUM_FILTER_STEPS_FOR_Ts
                * zpp_interp_points_sfr
                * dens_ninterp
            )

            tables_double += (
                6.0 * global_params.NUM_FILTER_STEPS_FOR_Ts * dens_ninterp
            )  # grid_dens, density_gridpoints, fcoll_interp1, fcoll_interp2, dfcoll_interp1, dfcoll_interp2
            tables_double += zpp_interp_points_sfr  # ST_over_PS_arg_grid

            tables_float += (
                7.0 * global_params.NUM_FILTER_STEPS_FOR_Ts
            )  # delNL0_bw, delNL0_Offset, delNL0_LL, delNL0_UL, delNL0_ibw, log10delNL0_diff, log10delNL0_diff_UL

            # dens_grid_int_vals
            num_c_boxes_initialised += (
                0.5 * global_params.NUM_FILTER_STEPS_FOR_Ts
            )  # 0.5 as it is a short

    # dstarlya_dt_prefactor, fcoll_R_array, sigma_Tmin, ST_over_PS, sum_lyn
    tables_double += 5.0 * global_params.NUM_FILTER_STEPS_FOR_Ts
    if flag_options.USE_LYA_HEATING:
        tables_double += (
            4.0 * global_params.NUM_FILTER_STEPS_FOR_Ts
        )  # dstarlya_cont_dt_prefactor,dstarlya_inj_dt_prefactor, sum_ly2, sum_lynto2

    if flag_options.USE_MINI_HALOS:
        tables_double += (
            7.0 * global_params.NUM_FILTER_STEPS_FOR_Ts
        )  # dstarlya_dt_prefactor_MINI, dstarlyLW_dt_prefactor, dstarlyLW_dt_prefactor_MINI
        # ST_over_PS_MINI, sum_lyn_MINI, sum_lyLWn, sum_lyLWn_MINI,
        if flag_options.USE_LYA_HEATING:
            tables_double += (
                4.0 * global_params.NUM_FILTER_STEPS_FOR_Ts
            )  # dstarlya_cont_dt_prefactor_MINI,dstarlya_inj_dt_prefactor_MINI,sum_ly2_MINI,sum_lynto2_MINI

        tables_float += global_params.NUM_FILTER_STEPS_FOR_Ts  # Mcrit_atom_interp_table

    tables_float += (
        1.5 * global_params.NUM_FILTER_STEPS_FOR_Ts
    )  # zpp_for_evolve_list, SingleVal_int (0.5 as it is a short)

    tables_double += (
        6.0 * x_int_nxhii * global_params.NUM_FILTER_STEPS_FOR_Ts
    )  # freq_int_heat_tbl, freq_int_ion_tbl, freq_int_lya_tbl, freq_int_heat_tbl_diff, freq_int_ion_tbl_diff, freq_int_lya_tbl_diff

    tables_float += (
        x_int_nxhii + global_params.NUM_FILTER_STEPS_FOR_Ts
    )  # inverse_diff, zpp_growth

    if flag_options.USE_MASS_DEPENDENT_ZETA:
        if flag_options.USE_MINI_HALOS:
            tables_double += (
                4.0 * zpp_interp_points_sfr + 2.0 * zpp_interp_points_sfr * nmturn
            )  # z_val, Nion_z_val, Nion_z_val_MINI, z_X_val, SFRD_val, SFRD_val_MINI
        else:
            tables_double += (
                4.0 * zpp_interp_points_sfr
            )  # z_val, Nion_z_val, z_X_val, SFRD_val
    else:
        # This is dependent on the user defined input, that cannot be captured here.
        # However, approximating it as ~ 1000 (almost certainly an overestimate) should be sufficient for almost all uses.
        tables_double += 1000.0  # FgtrM_1DTable_linear

    # These supersede usage in IonisationBox
    if user_params.USE_INTERPOLATION_TABLES:
        tables_float += (
            3.0 * nmass
        )  # Mass_InterpTable, Sigma_InterpTable, dSigmadm_InterpTable

    # These are all fftwf complex arrays (thus 2 * size)
    size_c = (2.0 * (np.float32(1.0).nbytes)) * num_c_boxes * hii_kspace_num_pixels

    size_c += (
        (np.float32(1.0).nbytes)
        * num_c_boxes_alt
        * (user_params.HII_DIM)
        * user_params.NON_CUBIC_FACTOR
        * (user_params.HII_DIM) ** 2
    )

    size_c_init = (
        (np.float32(1.0).nbytes)
        * num_c_boxes_initialised
        * (user_params.HII_DIM)
        * user_params.NON_CUBIC_FACTOR
        * (user_params.HII_DIM) ** 2
    )

    # Now, add all the table data (which are kept throughout the calculation)
    size_c_init += (np.float32(1.0).nbytes) * tables_float + (
        np.float64(1.0).nbytes
    ) * tables_double

    return {"python": size_py, "c_init": size_c_init, "c_per_z": size_c}


def mem_brightness_temperature(
    *,
    user_params=None,
    astro_params=None,
    flag_options=None,
):
    """A function to estimate total memory usage of a brightness_temperature call."""
    # Memory usage of Python BrightnessTemp class.
    ts_box = BrightnessTemp(
        redshift=9.0,
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )
    size_py = 0
    for key in ts_box._array_structure:
        size_py += np.prod(ts_box._array_structure[key])

    # These are all float arrays
    size_py = (np.float32(1.0).nbytes) * size_py

    # Memory usage within BrightnessTemperatureBox.c
    hii_tot_fft_num_pixels = (
        2.0
        * (float(user_params.HII_DIM) * user_params.NON_CUBIC_FACTOR / 2.0 + 1.0)
        * (user_params.HII_DIM) ** 2
    )

    # v, vel_gradient
    num_c_boxes = 2.0

    size_c = (np.float32(1.0).nbytes) * num_c_boxes * hii_tot_fft_num_pixels

    # For x_pos, x_pos_offset, delta_T_RSD_LOS as part of the RSDs
    size_c += (np.float32(1.0).nbytes) * (
        2.0 * global_params.NUM_FILTER_STEPS_FOR_Ts
        + user_params.N_THREADS
        * float(user_params.HII_DIM)
        * user_params.NON_CUBIC_FACTOR
    )

    return {"python": size_py, "c": size_c}


def mem_halo_field(
    *,
    user_params=None,
):
    """A function to estimate total memory usage of a determine_halo_list call."""
    # Memory usage within FindHaloes.c
    kspace_num_pixels = (
        float(user_params.DIM) * user_params.NON_CUBIC_FACTOR / 2.0 + 1.0
    ) * (user_params.DIM) ** 2

    # density_field, density_field_saved
    num_c_boxes = 2.0

    # halo_field, in_halo
    num_c_boxes_alt = (
        1.25  # in_halo is size 1 (char *) so count it as 0.25 of float (4)
    )

    if global_params.OPTIMIZE:
        # forbidden
        num_c_boxes_alt += 0.25

    # These are fftwf complex arrays (thus 2 * size)
    size_c = (2.0 * np.float32(1.0).nbytes) * num_c_boxes * kspace_num_pixels

    size_c += (
        (np.float32(1.0).nbytes)
        * num_c_boxes_alt
        * (user_params.DIM)
        * user_params.NON_CUBIC_FACTOR
        * (user_params.DIM) ** 2
    )

    # We don't know a priori how many haloes that will be found, but we'll estimate the memory usage
    # at 10 per cent of the total number of pixels (HII_DIM, likely an over estimate)
    # Below the factor of 4 corresponds to the mass and three spatial locations. It is defined as an
    # int but I'll leave it as 4 bytes in case
    size_c += (
        0.1
        * 4.0
        * (np.float32(1.0).nbytes)
        * (user_params.HII_DIM)
        * user_params.NON_CUBIC_FACTOR
        * (user_params.HII_DIM) ** 2
    )
    size_py = (
        0.1
        * 4.0
        * (np.float32(1.0).nbytes)
        * (user_params.HII_DIM)
        * user_params.NON_CUBIC_FACTOR
        * (user_params.HII_DIM) ** 2
    )

    return {"python": size_py, "c": size_c}


def mem_perturb_halo(
    *,
    user_params=None,
):
    """A function to estimate total memory usage of a perturb_halo_list call."""
    # Memory usage of Python PerturbHaloField class.

    # We don't know a priori how many haloes that will be found, but we'll estimate the memory usage
    # at 10 per cent of the total number of pixels (HII_DIM, likely an over estimate)
    # Below the factor of 4 corresponds to the mass and three spatial locations. It is defined as an
    # int but I'll leave it as 4 bytes in case
    size_c = (
        0.1
        * 4.0
        * (np.float32(1.0).nbytes)
        * (user_params.HII_DIM)
        * user_params.NON_CUBIC_FACTOR
        * (user_params.HII_DIM) ** 2
    )

    return {"python": 0.0, "c": size_c}


def print_memory_estimate(
    *,
    memory_data=None,
    user_params=None,
    astro_params=None,
    flag_options=None,
):
    """Function to output information in a manageable format."""
    # bytes_in_gb = 1024**3

    # print("")
    # if "python_lc" in memory_data.keys():
    #     print("Memory info for run_lightcone")
    # elif len(memory_data) == 3:
    #     print("Memory info for initial_conditions")
    # elif len(memory_data) == 5:
    #     print("Memory info for perturb_field")
    # else:
    #     print("Memory info for run_coeval")
    # print("")
    # print("%s" % (user_params))
    # if astro_params is not None:
    #     print("%s" % (astro_params))
    # if flag_options is not None:
    #     print("%s" % (flag_options))
    # print("")
    # print("Peak memory usage: %g (GB)" % (memory_data["peak_memory"] / bytes_in_gb))
    # print("")
    # if "python_lc" in memory_data.keys():
    #     print(
    #         "Memory for stored lightcones: %g (GB)"
    #         % (memory_data["python_lc"] / bytes_in_gb)
    #     )
    # """logger.info("Peak memory usage: %g (GB)"%(memory_data['peak_memory']/bytes_in_gb))"""
    # print(
    #     "Memory for ICs: %g (GB; Python) %g (GB; C)"
    #     % (memory_data["ics_python"] / bytes_in_gb, memory_data["ics_c"] / bytes_in_gb)
    # )
    # if "pf_python" in memory_data.keys():
    #     print(
    #         "Memory for single perturbed field: %g (GB; Python) %g (GB; C)"
    #         % (
    #             memory_data["pf_python"] / bytes_in_gb,
    #             memory_data["pf_c"] / bytes_in_gb,
    #         )
    #     )
    # if "hf_python" in memory_data.keys() and flag_options.USE_HALO_FIELD:
    #     print(
    #         "Note these are approximations as we don't know a priori how many haloes there are (assume 10 per cent of volume)"
    #     )
    #     print(
    #         "Memory for generating halo list: %g (GB; Python) %g (GB; C)"
    #         % (
    #             memory_data["hf_python"] / bytes_in_gb,
    #             memory_data["hf_c"] / bytes_in_gb,
    #         )
    #     )
    #     print(
    #         "Memory for perturbing halo list: %g (GB; Python) %g (GB; C)"
    #         % (
    #             memory_data["phf_python"] / bytes_in_gb,
    #             memory_data["phf_c"] / bytes_in_gb,
    #         )
    #     )
    # if "ib_python" in memory_data.keys():
    #     print(
    #         "Memory for single ionized box: %g (GB; Python) %g (GB; C)"
    #         % (
    #             memory_data["ib_python"] / bytes_in_gb,
    #             memory_data["ib_c"] / bytes_in_gb,
    #         )
    #     )

    # if "st_python" in memory_data.keys() and flag_options.USE_TS_FLUCT:
    #     print(
    #         "Memory for single spin temperature box: %g (GB; Python) %g (GB; C per z) %g (GB; C retained)"
    #         % (
    #             memory_data["st_python"] / bytes_in_gb,
    #             memory_data["st_c_per_z"] / bytes_in_gb,
    #             memory_data["st_c_init"] / bytes_in_gb,
    #         )
    #     )
    # if "bt_python" in memory_data.keys():
    #     print(
    #         "Memory for single brightness temperature box: %g (GB; Python) %g (GB; C)"
    #         % (
    #             memory_data["bt_python"] / bytes_in_gb,
    #             memory_data["bt_c"] / bytes_in_gb,
    #         )
    #     )
    # print("")
