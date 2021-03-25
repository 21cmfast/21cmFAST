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
    return {}


def mem_initial_conditions(
    *,
    user_params=None,
):
    """Memory usage of Python initial conditions class."""
    # lowres_density, lowres_vx, lowres_vy, lowres_vz, lowres_vcb,
    # lowres_vx_2LPT, lowres_vy_2LPT, lowres_vz_2LPT
    num_py_boxes_HII_DIM = 8.0

    # hires_density, hires_vx, hires_vy, hires_vz, hires_vcb,
    # hires_vx_2LPT, hires_vy_2LPT, hires_vz_2LPT
    num_py_boxes_DIM = 8.0

    size_py = num_py_boxes_DIM * (user_params.DIM) ** 3
    size_py += num_py_boxes_HII_DIM * (user_params.HII_DIM) ** 3

    size_py = (np.float32(1.0).nbytes) * size_py

    """Memory usage of C initial conditions"""
    KSPACE_NUM_PIXELS = (float(user_params.DIM) / 2.0 + 1.0) * (user_params.DIM) ** 2

    # HIRES_box, HIRES_box_saved
    num_c_boxes = 2

    # HIRES_box_vcb_x, HIRES_box_vcb_y, HIRES_box_vcb_z
    if user_params.USE_RELATIVE_VELOCITIES:
        num_c_boxes += 3

    # phi_1 (6 components)
    if global_params.SECOND_ORDER_LPT_CORRECTIONS:
        num_c_boxes += 6

    # These are all fftwf complex arrays (thus 2 * size)
    size_c = (2.0 * (np.float32(1.0).nbytes)) * num_c_boxes * KSPACE_NUM_PIXELS

    return {"python": size_py, "c": size_c}
