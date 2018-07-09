"""
A thin python wrapper for the 21cmFAST C-code.
"""
from ._21cmfast import ffi, lib
import numpy as np
from ._utils import StructWithDefaults, OutputStruct
from astropy.cosmology import Planck15

from os import path
import yaml


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
        while not self._RANDOM_SEED:
            self._RANDOM_SEED = int(np.random.randint(1, 1e12))
        return self._RANDOM_SEED

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

    @property
    def HII_tot_num_pixels(self):
        return self.HII_DIM**3


# ======================================================================================================================
# OUTPUT STRUCTURES
# ======================================================================================================================
class InitialConditions(OutputStruct):
    """
    A class containing all initial conditions boxes.
    """
    ffi = ffi

    def _init_boxes(self):
        self.hires_density = np.zeros(self.user_params.tot_fft_num_pixels, dtype=np.float32)
        self.lowres_density = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vz = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)        
        self.lowres_vz_2LPT = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)        
        return ['hires_density','lowres_density','lowres_vz','lowres_vz_2LPT']

class PerturbedField(InitialConditions):
    """
    A class containing all perturbed field boxes
    """
    _id = "InitialConditions" # Makes it look at the InitialConditions files for writing.

    def __init__(self, user_params, cosmo_params, redshift):
        super().__init__(user_params, cosmo_params, redshift=float(redshift))

        # Extend its group name to include the redshift, so that
        self._group += "_z%.4f"%self.redshift

    def _init_boxes(self):
        self.density = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.velocity = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        return ['density', 'velocity']


# ======================================================================================================================
# WRAPPING FUNCTIONS
# ======================================================================================================================
def initial_conditions(user_params=UserParams(), cosmo_params=CosmoParams(), regenerate=False, write=True, direc=None,
                       fname=None, match_seed=False):
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

    direc : str, optional
        The directory in which to search for the boxes and write them. By default, this is the centrally-managed
        directory, given by the ``config.yml`` in ``.21CMMC`.

    fname : str, optional
        The filename to search for/write to.

    match_seed : bool, optional
        Whether to force the random seed to also match in order to be considered a match.

    Returns
    -------
    `~InitialConditions`
        The class which contains the various boxes defining the initial conditions.
    """
    # First initialize memory for the boxes that will be returned.
    boxes = InitialConditions(user_params,cosmo_params)
        
    # First check whether the boxes already exist.
    if not regenerate:
        try:
            boxes.read(direc, fname, match_seed)
            print("Existing init_boxes found and read in.")
            return boxes
        except IOError:
            pass

    # Run the C code
    lib.ComputeInitialConditions(user_params(), cosmo_params(), boxes())
    boxes.filled = True

    # Optionally do stuff with the result (like writing it)
    if write:
        boxes.write(direc, fname)

    return boxes


def perturb_field(redshift, init_boxes=None, user_params=None, cosmo_params=None,
                  regenerate=False, write=True, direc=None,
                  fname=None, match_seed=False):
    """
    Compute a perturbed field at a given redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to compute the perturbed field.

    init_boxes : :class:`~InitialConditions` instance, optional
        If given, these initial conditions boxes will be used, otherwise initial conditions will be generated. If given,
        the user and cosmo params will be set from this object.

    user_params : `~UserParams` instance, optional
        Defines the overall options and parameters of the run.

    cosmo_params : `~CosmoParams` instance, optional
        Defines the cosmological parameters used to compute initial conditions.


    regenerate : bool, optional
        Whether to force regeneration of the initial conditions, even if a corresponding box is found.

    write : bool, optional
        Whether to write results to file.

    direc : str, optional
        The directory in which to search for the boxes and write them. By default, this is the centrally-managed
        directory, given by the ``config.yml`` in ``.21CMMC`.

    fname : str, optional
        The filename to search for/write to.

    match_seed : bool, optional
        Whether to force the random seed to also match in order to be considered a match.

    Returns
    -------
    :class:`~PerturbField`
        An object containing the density and velocity fields at the specified redshift.

    Examples
    --------
    The simplest method is just to give a redshift::

    >>> field = perturb_field(7.0)
    >>> print(field.density)

    Doing so will internally call the :func:`~initial_conditions` function. If initial conditions have already been
    calculated, this can be avoided by passing them:

    >>> init_boxes = initial_conditions()
    >>> field7 = perturb_field(7.0, init_boxes)
    >>> field8 = perturb_field(8.0, init_boxes)

    The user and cosmo parameter structures are by default inferred from the ``init_boxes``, so that the following is
    consistent::

    >>> init_boxes = initial_conditions(user_params= UserParams(HII_DIM=1000))
    >>> field7 = perturb_field(7.0, init_boxes)

    If ``init_boxes`` is not passed, then these parameters can be directly passed::

    >>> field7 = perturb_field(7.0, user_params=UserParams(HII_DIM=1000))

    """
    # Try setting the user/cosmo params via the init_boxes
    if init_boxes is not None:
        user_params = init_boxes.user_params
        cosmo_params = init_boxes.cosmo_params

    # Set to defaults if init_boxes wasn't provided and neither were they.
    user_params = user_params or UserParams()
    cosmo_params = cosmo_params or CosmoParams()

    # Make sure we've got computed init boxes.
    if init_boxes is None or not init_boxes.filled:
        init_boxes = initial_conditions(
            user_params, cosmo_params, regenerate=regenerate, write=write,
            direc=direc, fname=fname
        )

    # Initialize perturbed boxes.
    fields = PerturbedField(user_params, cosmo_params, redshift)

    # Check whether the boxes already exist
    if not regenerate:
        try:
            fields.read(direc, fname, match_seed=match_seed)
            print("Existing perturb_field boxes found and read in.")
            return fields
        except IOError:
            pass

    # Run the C Code
    lib.ComputePerturbField(redshift, init_boxes(), fields())
    fields.filled = True

    # Optionally do stuff with the result (like writing it)
    if write:
        fields.write(direc, fname)

    return fields

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

