"""Function to estimate total memory usage."""

import numpy as np
from copy import deepcopy

from .inputs import AstroParams, CosmoParams, FlagOptions, UserParams, global_params
from .wrapper import _logscroll_redshifts, _setup_lightcone


def estimate_memory_coeval(
    *,
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
):
    """Compute an estimate of the requisite memory needed by the user for a run_coeval call."""
    """
    # Initial conditions
    mem_initial_conditions(user_params=self.user_params)

    # Perturb field

    # Halo field
    if flag_options.USE_HALO_FIELD:

    # Photon non-conservation
    if flag_options.PHOTON_CONS:

    # Note, if any of below are set the require current and
    # previous boxes, thus need to be careful

    # Spin temperature
    if flag_options.USE_TS_FLUCT:

    # Mini-halos
    if flag_options.USE_MINI_HALOS:

    # Inhomogeneous recombinations
    if flag_options.INHOMO_RECO:


    # Output a summary of information in human readable format
    """
    return {}


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
    """Compute an estimate of the requisite memory needed by the user for a run_lightcone call."""
    # First, calculate the memory usage for the initial conditions
    memory_ics = mem_initial_conditions(user_params=user_params)

    memory_data = {"ics_%s" % k: memory_ics["%s" % k] for k in memory_ics.keys()}

    # Maximum memory while running ICs
    peak_memory = memory_ics["c"] + memory_ics["python"]

    # Now the perturb field
    memory_pf = mem_perturb_field(user_params=user_params)

    memory_data.update({"pf_%s" % k: memory_pf["%s" % k] for k in memory_pf.keys()})

    # Stored ICs in python + allocated C and Python memory for perturb_field
    current_memory = memory_ics["python"] + memory_pf["python"] + memory_pf["c"]

    # Check if running perturb_field requires more memory than generating ICs
    peak_memory = peak_memory if peak_memory > current_memory else current_memory

    # If we are using the photon non-conservation correction
    if flag_options.PHOTON_CONS:
        # First need to create new structs for photon the photon-conservation
        astro_params_photoncons = deepcopy(astro_params)
        astro_params_photoncons._R_BUBBLE_MAX = astro_params.R_BUBBLE_MAX

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

    memory_data.update({"hf_%s" % k: memory_hf["%s" % k] for k in memory_hf.keys()})

    # Calculate the memory for a perturb_halo_list call
    memory_phf = mem_perturb_halo(user_params=user_params)

    memory_data.update({"phf_%s" % k: memory_phf["%s" % k] for k in memory_phf.keys()})

    # Calculate the memory for an ionize_box call
    memory_ib = mem_ionize_box(
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )

    memory_data.update({"ib_%s" % k: memory_ib["%s" % k] for k in memory_ib.keys()})

    # Calculate the memory for a spin_temperature call
    memory_st = mem_spin_temperature(
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )

    memory_data.update({"st_%s" % k: memory_st["%s" % k] for k in memory_st.keys()})

    # Calculate the memory for a brightness_temperature call
    memory_bt = mem_brightness_temperature(user_params=user_params)

    memory_data.update({"bt_%s" % k: memory_bt["%s" % k] for k in memory_bt.keys()})

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

    memory_data.update({"peak_memory": peak_memory})

    return memory_data


def mem_initial_conditions(
    *,
    user_params=None,
):
    """A function to estimate total memory usage of an initial_conditions call."""
    """Memory usage of Python InitialConditions class."""
    """All declared HII_DIM boxes"""
    # lowres_density, lowres_vx, lowres_vy, lowres_vz, lowres_vcb
    # lowres_vx_2LPT, lowres_vy_2LPT, lowres_vz_2LPT
    num_py_boxes_HII_DIM = 8.0

    """All declared DIM boxes"""
    # hires_density, hires_vx, hires_vy, hires_vz
    # hires_vx_2LPT, hires_vy_2LPT, hires_vz_2LPT
    num_py_boxes_DIM = 7.0

    size_py = num_py_boxes_DIM * (user_params.DIM) ** 3
    size_py += num_py_boxes_HII_DIM * (user_params.HII_DIM) ** 3

    # These are all float arrays
    size_py = (np.float32(1.0).nbytes) * size_py

    """Memory usage within GenerateICs"""
    kspace_num_pixels = (float(user_params.DIM) / 2.0 + 1.0) * (user_params.DIM) ** 2

    """All declared DIM boxes"""
    # HIRES_box, HIRES_box_saved
    num_c_boxes = 2

    """All declared 2LPT boxes (DIM)"""
    # phi_1 (6 components)
    if global_params.SECOND_ORDER_LPT_CORRECTIONS:
        num_c_boxes += 6

    # These are all fftwf complex arrays (thus 2 * size)
    size_c = (2.0 * (np.float32(1.0).nbytes)) * num_c_boxes * kspace_num_pixels

    return {"python": size_py, "c": size_c}


def mem_perturb_field(
    *,
    user_params=None,
):
    """A function to estimate total memory usage of a perturb_field call."""
    """Memory usage of Python PerturbedField class."""
    """All declared HII_DIM boxes"""
    # density, velocity
    num_py_boxes_HII_DIM = 2.0

    size_py = num_py_boxes_HII_DIM * (user_params.HII_DIM) ** 3

    # These are all float arrays
    size_py = (np.float32(1.0).nbytes) * size_py

    """Memory usage within PerturbField.c"""
    kspace_num_pixels = (float(user_params.DIM) / 2.0 + 1.0) * (user_params.DIM) ** 2
    hii_kspace_num_pixels = (float(user_params.HII_DIM) / 2.0 + 1.0) * (
        user_params.HII_DIM
    ) ** 2

    tot_num_pixels = float(user_params.DIM) ** 3
    hii_tot_num_pixels = float(user_params.HII_DIM) ** 3

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
    # First do check on inhomogeneous recombinations
    astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

    # determine number of filtering scales (for USE_MINI_HALOS)
    if flag_options.USE_MINI_HALOS:
        n_filtering = (
            int(
                np.log(
                    min(astro_params.R_BUBBLE_MAX, 0.620350491 * user_params.BOX_LEN)
                    / max(
                        global_params.R_BUBBLE_MIN,
                        0.620350491
                        * user_params.BOX_LEN
                        / np.float64(user_params.HII_DIM),
                    )
                )
                / np.log(global_params.DELTA_R_HII_FACTOR)
            )
        ) + 1
    else:
        n_filtering = 1

    """Memory usage of Python IonizedBox class."""

    """All declared HII_DIM boxes"""
    # xH_box, Gamma12_box, MFP_box, z_re_box, dNrec_box, temp_kinetic_all_gas
    num_py_boxes = 6.0

    # Fcoll
    num_py_boxes += n_filtering

    # Fcoll_MINI
    if flag_options.USE_MINI_HALOS:
        num_py_boxes += n_filtering

    size_py = num_py_boxes * (user_params.HII_DIM) ** 3

    # These are all float arrays
    size_py = (np.float32(1.0).nbytes) * size_py

    """Memory usage within IonisationBox.c"""
    hii_kspace_num_pixels = (float(user_params.HII_DIM) / 2.0 + 1.0) * (
        user_params.HII_DIM
    ) ** 2

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

    # There are a bunch of 1 and 2D interpolation tables, but ignore those as they are small relative to 3D grids

    if flag_options.USE_MASS_DEPENDENT_ZETA and flag_options.USE_MINI_HALOS:
        # log10_Mturnover_unfiltered, log10_Mturnover_filtered, log10_Mturnover_MINI_unfiltered, log10_Mturnover_MINI_filtered
        num_c_boxes += 4.0

    if flag_options.USE_HALO_FIELD:
        # M_coll_unfiltered, M_coll_filtered
        num_c_boxes += 2.0

    # These are all fftwf complex arrays (thus 2 * size)
    size_c = (2.0 * (np.float32(1.0).nbytes)) * num_c_boxes * hii_kspace_num_pixels

    return {"python": size_py, "c": size_c}


def mem_spin_temperature(
    *,
    user_params=None,
    astro_params=None,
    flag_options=None,
):
    """A function to estimate total memory usage of a spin_temperature call."""
    """Memory usage of Python IonizedBox class."""

    """All declared HII_DIM boxes"""
    # Ts_bx, x_e_box, Tk_box, J_21_LW_box
    num_py_boxes = 4.0

    size_py = num_py_boxes * (user_params.HII_DIM) ** 3

    # These are all float arrays
    size_py = (np.float32(1.0).nbytes) * size_py

    """Memory usage within SpinTemperatureBox.c"""
    hii_kspace_num_pixels = (float(user_params.HII_DIM) / 2.0 + 1.0) * (
        user_params.HII_DIM
    ) ** 2

    # box, unfiltered_box
    num_c_boxes = 2.0
    num_c_boxes_alt = 0.0
    num_c_boxes_initialised = 0.0

    mem_c_interp = 0.0

    if flag_options.USE_MINI_HALOS:
        # log10_Mcrit_LW_unfiltered, log10_Mcrit_LW_filtered
        num_c_boxes += 2.0

        # log10_Mcrit_LW
        num_c_boxes_alt += global_params.NUM_FILTER_STEPS_FOR_Ts

    # There are a bunch of 1 and 2D interpolation tables, but ignore those as they are small relative to 3D grids
    # I will add any that could end up considerably large.

    if flag_options.USE_MASS_DEPENDENT_ZETA:
        # delNL0
        num_c_boxes_initialised += global_params.NUM_FILTER_STEPS_FOR_Ts

        # del_fcoll_Rct, m_xHII_low_box, inverse_val_box
        num_c_boxes_initialised += (
            3.0  # m_xHII_low_box is an int, but I'll count it as 4 bytes just in case
        )

        # dxheat_dt_box, dxion_source_dt_box, dxlya_dt_box, dstarlya_dt_box
        num_c_boxes_initialised += 2.0 * 4.0  # factor of 2. as these are doubles

        if flag_options.USE_MINI_HALOS:
            # del_fcoll_Rct_MINI
            num_c_boxes_initialised += 1.0

            # dstarlyLW_dt_box, dxheat_dt_box_MINI, dxion_source_dt_box_MINI
            # dxlya_dt_box_MINI, dstarlya_dt_box_MINI, dstarlyLW_dt_box_MINI
            num_c_boxes_initialised += (
                2.0 * 6.0
            )  # factor of 2. taking into account that these are doubles

            if user_params.USE_INTERPOLATION_TABLES:
                # log10_SFRD_z_low_table_MINI, SFRD_z_high_table_MINI (factor of 4 as these are float tables)
                mem_c_interp += (
                    4.0 * global_params.NUM_FILTER_STEPS_FOR_Ts * 250.0 * 50.0
                )  # NSFR_low = 250, NMTURN = 50

                mem_c_interp += (
                    4.0 * global_params.NUM_FILTER_STEPS_FOR_Ts * 200.0 * 50.0
                )  # NSFR_high = 200, NMTURN = 50
    else:
        # delNL0_rev
        num_c_boxes_initialised += global_params.NUM_FILTER_STEPS_FOR_Ts

        if user_params.USE_INTERPOLATION_TABLES:
            # fcoll_R_grid, dfcoll_dz_grid (factor of 8. as these are double)
            mem_c_interp += (
                2.0 * 8.0 * (global_params.NUM_FILTER_STEPS_FOR_Ts * 400 * 400)
            )  # zpp_interp_points_SFR = 400, dens_Ninterp = 400

            # dens_grid_int_vals
            num_c_boxes_initialised += 0.5  # 0.5 as it is a short

    # These are all fftwf complex arrays (thus 2 * size)
    size_c = (2.0 * (np.float32(1.0).nbytes)) * num_c_boxes * hii_kspace_num_pixels

    size_c += (np.float32(1.0).nbytes) * num_c_boxes_alt * (user_params.HII_DIM ** 3.0)

    size_c_init = (
        (np.float32(1.0).nbytes)
        * num_c_boxes_initialised
        * (user_params.HII_DIM ** 3.0)
    )

    size_c_init += mem_c_interp

    return {"python": size_py, "c_init": size_c_init, "c_per_z": size_c}


def mem_brightness_temperature(
    *,
    user_params=None,
):
    """A function to estimate total memory usage of a brightness_temperature call."""
    """Memory usage of Python BrightnessTemp class."""

    """All declared HII_DIM boxes"""
    # brightness_temp
    num_py_boxes = 1.0

    size_py = num_py_boxes * (user_params.HII_DIM) ** 3

    # These are all float arrays
    size_py = (np.float32(1.0).nbytes) * size_py

    """Memory usage within BrightnessTemperatureBox.c"""
    hii_tot_fft_num_pixels = (
        2.0 * (float(user_params.HII_DIM) / 2.0 + 1.0) * (user_params.HII_DIM) ** 2
    )

    # box, unfiltered_box
    num_c_boxes = 2.0

    size_c = (np.float32(1.0).nbytes) * num_c_boxes * hii_tot_fft_num_pixels

    return {"python": size_py, "c": size_c}


def mem_halo_field(
    *,
    user_params=None,
):
    """A function to estimate total memory usage of a determine_halo_list call."""
    """Memory usage of Python HaloField class."""

    """All declared DIM boxes"""
    # halo_field
    num_py_boxes = 1.0

    size_py = num_py_boxes * (user_params.DIM) ** 3

    # These are all float arrays
    size_py = (np.float32(1.0).nbytes) * size_py

    """Memory usage within FindHaloes.c"""
    kspace_num_pixels = (float(user_params.DIM) / 2.0 + 1.0) * (user_params.DIM) ** 2

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

    size_c += (np.float32(1.0).nbytes) * num_c_boxes_alt * (user_params.DIM) ** 3

    # We don't know a priori how many haloes that will be found, but we'll estimate the memory usage
    # at 10 per cent of the total number of pixels (HII_DIM, likely an over estimate)
    # Below the factor of 4 corresponds to the mass and three spatial locations. It is defined as an
    # int but I'll leave it as 4 bytes in case
    size_c += 0.1 * 4.0 * (np.float32(1.0).nbytes) * (user_params.HII_DIM) ** 3

    return {"python": size_py, "c": size_c}


def mem_perturb_halo(
    *,
    user_params=None,
):
    """A function to estimate total memory usage of a perturb_halo_list call."""
    """Memory usage of Python PerturbHaloField class."""

    # We don't know a priori how many haloes that will be found, but we'll estimate the memory usage
    # at 10 per cent of the total number of pixels (HII_DIM, likely an over estimate)
    # Below the factor of 4 corresponds to the mass and three spatial locations. It is defined as an
    # int but I'll leave it as 4 bytes in case
    size_c = 0.1 * 4.0 * (np.float32(1.0).nbytes) * (user_params.HII_DIM) ** 3

    return {"python": 0.0, "c": size_c}
