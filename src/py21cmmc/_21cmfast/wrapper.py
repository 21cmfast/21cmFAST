"""
A thin python wrapper for the 21cmFAST C-code.
"""
from ._21cmfast import ffi, lib
import numpy as np


class InitialConditions:
    """
    A class containing all initial conditions boxes.
    """
    def __init__(self, box_dim):
        self.lowres_density = np.zeros(box_dim.HII_tot_num_pixels)
        self.lowres_vx = np.zeros(box_dim.HII_tot_num_pixels)
        self.lowres_vy = np.zeros(box_dim.HII_tot_num_pixels)
        self.lowres_vz = np.zeros(box_dim.HII_tot_num_pixels)
        self.lowres_vx_2LPT = np.zeros(box_dim.HII_tot_num_pixels)
        self.lowres_vy_2LPT = np.zeros(box_dim.HII_tot_num_pixels)
        self.lowres_vz_2LPT = np.zeros(box_dim.HII_tot_num_pixels)
        self.hires_density = np.zeros(box_dim.tot_fft_num_pixels)


class PerturbedField:
    """
    A class containing all perturbed field boxes
    """
    def __init__(self, n):
        self.density = np.zeros(n)
        self.velocity = np.zeros(n)


def initial_conditions(box_dim, cosmo_params, regenerate=False, write=True):
    """
    Compute initial conditions.

    Parameters
    ----------
    box_dim
    cosmo_params
    regenerate
    write
    dir

    Returns
    -------

    """
    # First initialize memory for the boxes that will be returned.
    boxes = InitialConditions(box_dim)

    # Run the C code
    lib.ComputeInitialConditions(box_dim.cstruct, cosmo_params.cstruct, boxes)

    # Optionally do stuff with the result (like writing it)
    if write:
        pass

    return boxes


def perturb_field(redshift, init_boxes,  write=True, regenerate=False, read=False):
    """
    Compute a perturbed field at a given redshift.

    Parameters
    ----------
    redshift
    init_boxes
    write
    regenerate
    read
    dir

    Returns
    -------

    """
    # First initialize perturbed boxes.
    fields = PerturbedField(len(init_boxes.lowres_vx))

    # Run the C Code
    lib.ComputePerturbField(redshift, init_boxes, fields)

    # Optionally do stuff with the result (like writing it)
    if write:
        pass

    return fields


def ionize(redshifts, flag_options, astro_params):
    for z in redshifts:
        lib.ComputeIonisationBoxes(z, z+0.2, flag_options, astro_params)

    return something


def run_21cmfast(redshifts, box_dim=None, flag_options=None, astro_params=None, cosmo_params=None,
                 write=True, regenerate=False, run_perturb=True, run_ionize=True, init_boxes=None,
                 free_ps=True, progress_bar=True):

    # Create structures of parameters
    box_dim = box_dim or {}
    flag_options = flag_options or {}
    astro_params = astro_params or {}
    cosmo_params = cosmo_params or {}

    box_dim = BoxDim(**box_dim)
    flag_options = FlagOptions(**flag_options)
    astro_params = AstroParams(**astro_params)
    cosmo_params = CosmoParams(**cosmo_params)

    # Compute initial conditions, but only if they aren't passed in directly by the user.
    if init_boxes is None:
        init_boxes = initial_conditions(box_dim, cosmo_params, regenerate, write)

    output = [init_boxes]

    # Run perturb if desired
    if run_perturb:
        for z in redshifts:
            perturb_fields = perturb_field(z, init_boxes, regenerate=regenerate)

    # Run ionize if desired
    if run_ionize:
        ionized_boxes = ionize(redshifts, flag_options, astro_params)
        output += [ionized_boxes]

    return output

