"""Function to estimate total memory usage."""

import numpy as np

from .inputs import AstroParams, CosmoParams, FlagOptions, UserParams, global_params


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
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
):
    """Compute an estimate of the requisite memory needed by the user for a run_lightcone call."""
    memory_ics = mem_initial_conditions(user_params=user_params)

    memory_pf = mem_perturb_field(user_params=user_params)

    memory_ib = mem_ionize_box(
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )

    memory_st = mem_spin_temperature(
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )

    return {
        "ics_python": memory_ics["python"],
        "ics_c": memory_ics["c"],
        "pf_python": memory_pf["python"],
        "pf_c": memory_pf["c"],
        "ib_python": memory_ib["python"],
        "ib_c": memory_ib["c"],
        "st_python": memory_st["python"],
        "st_c": memory_st["c"],
    }


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

    mem_c_interp = 0.0

    if flag_options.USE_MINI_HALOS:
        # log10_Mcrit_LW_unfiltered, log10_Mcrit_LW_filtered
        num_c_boxes += 2.0

        num_c_boxes_alt += global_params.NUM_FILTER_STEPS_FOR_Ts

    # There are a bunch of 1 and 2D interpolation tables, but ignore those as they are small relative to 3D grids
    # I will add any that could end up considerably large.

    if flag_options.USE_MASS_DEPENDENT_ZETA:
        # delNL0
        num_c_boxes_alt += global_params.NUM_FILTER_STEPS_FOR_Ts

        # del_fcoll_Rct, m_xHII_low_box, inverse_val_box
        num_c_boxes_alt += (
            3.0  # m_xHII_low_box is an int, but I'll count it as 4 bytes just in case
        )

        # dxheat_dt_box, dxion_source_dt_box, dxlya_dt_box, dstarlya_dt_box
        num_c_boxes_alt += 2.0 * 4.0  # factor of 2. as these are doubles

        if flag_options.USE_MINI_HALOS:
            # del_fcoll_Rct_MINI
            num_c_boxes_alt += 1.0

            # dstarlyLW_dt_box, dxheat_dt_box_MINI, dxion_source_dt_box_MINI
            # dxlya_dt_box_MINI, dstarlya_dt_box_MINI, dstarlyLW_dt_box_MINI
            num_c_boxes_alt += (
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
        num_c_boxes_alt += global_params.NUM_FILTER_STEPS_FOR_Ts

        if user_params.USE_INTERPOLATION_TABLES:
            # fcoll_R_grid, dfcoll_dz_grid (factor of 8. as these are double)
            mem_c_interp += (
                2.0 * 8.0 * (global_params.NUM_FILTER_STEPS_FOR_Ts * 400 * 400)
            )  # zpp_interp_points_SFR = 400, dens_Ninterp = 400

            # dens_grid_int_vals
            num_c_boxes_alt += 0.5  # 0.5 as it is a short

    # These are all fftwf complex arrays (thus 2 * size)
    size_c = (2.0 * (np.float32(1.0).nbytes)) * num_c_boxes * hii_kspace_num_pixels

    size_c += (np.float32(1.0).nbytes) * num_c_boxes_alt * (user_params.HII_DIM ** 3.0)

    size_c += mem_c_interp

    return {"python": size_py, "c": size_c}
