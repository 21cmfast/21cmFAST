"""
A thin python wrapper for the 21cmFAST C-code.
"""
from ._21cmfast import ffi, lib
import numpy as np
from ._utils import StructWithDefaults
from astropy.cosmology import Planck15


# ======================================================================================================================
# PARAMETER STRUCTURES
# ======================================================================================================================
class CosmoParams(StructWithDefaults):
    """
    Cosmological parameters (with defaults) which translates to a C struct.

    Parameters
    ----------
    RANDOM_SEED : float, optional
        A seed to set the IC generator. If None, chosen from uniform distribution.

    SIGMA_8 : float, optional
        RMS mass variance (power spectrum normalisation).

    hlittle : float, optional
        H_0/100.

    OMm : float, optional
        Omega matter.

    OMb : float, optional
        Omega baryon, the baryon component.

    POWER_INDEX : float, optional
        Spectral index of the power spectrum.
    """
    ffi = ffi

    _defaults_ = dict(
        RANDOM_SEED = None,
        SIGMA_8 = 0.82,
        hlittle = Planck15.h,
        OMm = Planck15.Om0,
        OMb = Planck15.Ob0,
        POWER_INDEX = 0.97
    )

    @property
    def RANDOM_SEED(self):
        return self._RANDOM_SEED or int(np.random.randint(1, 1e12))

    @property
    def OMl(self):
        return 1 - self.OMm

    def cosmology(self):
        return Planck15.clone(h = self.hlittle, Om0 = self.OMm, Ob0 = self.OMb)


class UserParams(StructWithDefaults):
    """
    Structure containing user parameters (with defaults).

    Parameters
    ----------
    HII_DIM : int, optional
        Number of cells for the low-res box.

    DIM : int,optional
        Number of cells for the high-res box (sampling ICs) along a principal axis. To avoid
        sampling issues, DIM should be at least 3 or 4 times HII_DIM, and an integer multiple.
        By default, it is set to 4*HII_DIM.

    BOX_LEN : float, optional
        Length of the box, in Mpc.
    """
    ffi = ffi

    _defaults_ = dict(
        BOX_LEN = 150.0,
        DIM = None,
        HII_DIM = 100,
    )

    @property
    def DIM(self):
        return self._DIM or 4 * self.HII_DIM

    @property
    def tot_fft_num_pixels(self):
        return self.DIM**3


# ======================================================================================================================
# OUTPUT STRUCTURES
# ======================================================================================================================
class InitialConditions:
    """
    A class containing all initial conditions boxes.
    """
    def __init__(self, box_dim):
        # self.lowres_density = np.zeros(box_dim.HII_tot_num_pixels)
        # self.lowres_vx = np.zeros(box_dim.HII_tot_num_pixels)
        # self.lowres_vy = np.zeros(box_dim.HII_tot_num_pixels)
        # self.lowres_vz = np.zeros(box_dim.HII_tot_num_pixels)
        # self.lowres_vx_2LPT = np.zeros(box_dim.HII_tot_num_pixels)
        # self.lowres_vy_2LPT = np.zeros(box_dim.HII_tot_num_pixels)
        # self.lowres_vz_2LPT = np.zeros(box_dim.HII_tot_num_pixels)
        self.hires_density = np.zeros(box_dim.tot_fft_num_pixels, dtype=np.float32)

        # Put everything in the struct
        self.cstruct = ffi.new("struct InitialConditions*")
        self.cstruct.hires_density = ffi.cast("float *", ffi.from_buffer(self.hires_density))


class PerturbedField:
    """
    A class containing all perturbed field boxes
    """
    def __init__(self, n):
        self.density = np.zeros(n)
        self.velocity = np.zeros(n)


def initial_conditions(user_params=UserParams(), cosmo_params=CosmoParams(), regenerate=False, write=True):
    """
    Compute initial conditions.

    Parameters
    ----------
    user_params : `~UserParams` instance, optional
        Defines the overall options and parameters of the run.

    cosmo_params : `~CosmoParams` instance, optional
        Defines the cosmological parameters used to compute initial conditions.

    regenerate : bool, optional
        Whether to force regeneration of the initial conditions, even if a corresponding box is found.

    write : bool, optional
        Whether to write results to file.

    Returns
    -------
    `~InitialConditions`
        The class which contains the various boxes defining the initial conditions.
    """
    # First initialize memory for the boxes that will be returned.
    boxes = InitialConditions(user_params)

    # Run the C code
    lib.ComputeInitialConditions(user_params(), cosmo_params(), boxes.cstruct)

    # Optionally do stuff with the result (like writing it)
    if write:
        pass

    return boxes


# def perturb_field(redshift, init_boxes,  write=True, regenerate=False, read=False):
#     """
#     Compute a perturbed field at a given redshift.
#
#     Parameters
#     ----------
#     redshift
#     init_boxes
#     write
#     regenerate
#     read
#     dir
#
#     Returns
#     -------
#
#     """
#     # First initialize perturbed boxes.
#     fields = PerturbedField(len(init_boxes.lowres_vx))
#
#     # Run the C Code
#     lib.ComputePerturbField(redshift, init_boxes, fields)
#
#     # Optionally do stuff with the result (like writing it)
#     if write:
#         pass
#
#     return fields
#
#
# def ionize(redshifts, flag_options, astro_params):
#     for z in redshifts:
#         lib.ComputeIonisationBoxes(z, z+0.2, flag_options, astro_params)
#
#     return something
#
#
# def run_21cmfast(redshifts, box_dim=None, flag_options=None, astro_params=None, cosmo_params=None,
#                  write=True, regenerate=False, run_perturb=True, run_ionize=True, init_boxes=None,
#                  free_ps=True, progress_bar=True):
#
#     # Create structures of parameters
#     box_dim = box_dim or {}
#     flag_options = flag_options or {}
#     astro_params = astro_params or {}
#     cosmo_params = cosmo_params or {}
#
#     box_dim = BoxDim(**box_dim)
#     flag_options = FlagOptions(**flag_options)
#     astro_params = AstroParams(**astro_params)
#     cosmo_params = CosmoParams(**cosmo_params)
#
#     # Compute initial conditions, but only if they aren't passed in directly by the user.
#     if init_boxes is None:
#         init_boxes = initial_conditions(box_dim, cosmo_params, regenerate, write)
#
#     output = [init_boxes]
#
#     # Run perturb if desired
#     if run_perturb:
#         for z in redshifts:
#             perturb_fields = perturb_field(z, init_boxes, regenerate=regenerate)
#
#     # Run ionize if desired
#     if run_ionize:
#         ionized_boxes = ionize(redshifts, flag_options, astro_params)
#         output += [ionized_boxes]
#
#     return output

