"""
A thin python wrapper for the 21cmFAST C-code.
"""
from ._21cmfast import ffi, lib
import numpy as np
from ._utils import StructWithDefaults
from astropy.cosmology import Planck15

from os import path
import h5py
import yaml
import re, glob

from hashlib import md5

# Global Options
with open(path.expanduser(path.join("~", '.21CMMC', "config.yml"))) as f:
    opts = yaml.load(f)


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


# ======================================================================================================================
# OUTPUT STRUCTURES
# ======================================================================================================================
class InitialConditions:
    """
    A class containing all initial conditions boxes.
    """
    filled = False

    def __init__(self, user_params, cosmo_params):
        # self.lowres_density = np.zeros(box_dim.HII_tot_num_pixels)
        # self.lowres_vx = np.zeros(box_dim.HII_tot_num_pixels)
        # self.lowres_vy = np.zeros(box_dim.HII_tot_num_pixels)
        # self.lowres_vz = np.zeros(box_dim.HII_tot_num_pixels)
        # self.lowres_vx_2LPT = np.zeros(box_dim.HII_tot_num_pixels)
        # self.lowres_vy_2LPT = np.zeros(box_dim.HII_tot_num_pixels)
        # self.lowres_vz_2LPT = np.zeros(box_dim.HII_tot_num_pixels)
        self.hires_density = np.zeros(user_params.tot_fft_num_pixels, dtype=np.float32)

        self.user_params = user_params
        self.cosmo_params = cosmo_params

        # Put everything in the struct
        self.cstruct = ffi.new("struct InitialConditions*")
        self.cstruct.hires_density = ffi.cast("float *", ffi.from_buffer(self.hires_density))

    @property
    def _md5(self):
        return md5(repr(self).encode()).hexdigest()

    @property
    def _hashname(self):
        return self._md5 + "_r%s" % self.cosmo_params.RANDOM_SEED + ".h5"

    def _get_fname(self, direc=None, fname=None):
        if direc:
            fname = fname or self._hashname
        else:
            fname = self._hashname

        direc = direc or path.expanduser(opts['boxdir'])

        return path.join(direc, fname)

    def write(self, direc=None, fname=None):
        """
        Write the initial conditions boxes in standard HDF5 format.

        Parameters
        ----------
        direc : str, optional
            The directory in which to write the boxes. By default, this is the centrally-managed directory, given
            by the ``config.yml`` in ``.21CMMC`.

        fname : str, optional
            The filename to write to. This is only used if `direc` is not None. By default, the filename is a hash
            which accounts for the various parameters that define the boxes, to ensure uniqueness.
        """
        if not self.filled:
            raise IOError("The boxes have not yet been computed.")

        with h5py.File(self._get_fname(direc,fname), 'w') as f:
            # Save the cosmo and user params to the file
            cosmo = f.create_group("cosmo")
            for k,v in self.cosmo_params.pystruct.items():
                cosmo.attrs[k] = v

            user = f.create_group("user_params")
            for k, v in self.user_params.pystruct.items():
                user.attrs[k] = v

            # Save the boxes to the file
            boxes = f.create_group("init_boxes")
            boxes.create_dataset("hires_density", data = self.hires_density)

    @staticmethod
    def _find_file_without_seed(f):
        f = re.sub("r\d+\.", "r*.", f)
        allfiles = glob.glob(f)
        if allfiles:
            return allfiles[0]
        else:
            return None

    def find_existing(self, direc=None, fname=None, match_seed=False):
        """
        Try to find existing boxes which match the parameters of this instance.

        Parameters
        ----------
        direc : str, optional
            The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
            by the ``config.yml`` in ``.21CMMC`. This central directory will be searched in addition to whatever is
            passed to `direc`.

        fname : str, optional
            The filename to search for. This is used in addition to the filename automatically assigned by the hash
            of this instance.

        match_seed : bool, optional
            Whether to force the random seed to also match in order to be considered a match.

        Returns
        -------
        str
            The filename of an existing set of boxes, or None.
        """
        if direc is not None:
            if fname is not None:
                if path.exists(self._get_fname(direc, fname)):
                    return self._get_fname(direc, fname)

            f = self._get_fname(direc, None)
            if path.exists(f):
                return f
            elif not match_seed:
                f = self._find_file_without_seed(f)
                if f: return f

        f = self._get_fname(None, None)
        if path.exists(f):
            return f
        else:
            f = self._find_file_without_seed(f)
            if f: return f

        return None

    def exists(self, direc=None, fname=None, match_seed=False):
        return self.find_existing(direc, fname, match_seed) is not None

    def read(self, direc=None, fname=None, match_seed=False):
        """
        Try to find and read in existing boxes which match the parameters of this instance.

        Parameters
        ----------
        direc : str, optional
            The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
            by the ``config.yml`` in ``.21CMMC`. This central directory will be searched in addition to whatever is
            passed to `direc`.

        fname : str, optional
            The filename to search for. This is used in addition to the filename automatically assigned by the hash
            of this instance.

        match_seed : bool, optional
            Whether to force the random seed to also match in order to be considered a match.
        """
        pth = self.find_existing(direc, fname, match_seed)
        if pth is None:
            raise IOError("No boxes exist for these cosmo and user parameters.")

        with h5py.File(pth,'r') as f:
            boxes = f['init_boxes']

            # Fill our arrays.
            for k in boxes.keys():
                getattr(self, k)[...] = boxes[k][...]

            # Need to make sure that the seed is set to the one that's read in.
            seed = f['cosmo'].attrs['RANDOM_SEED']
            self.cosmo_params._RANDOM_SEED = seed

        self.filled = True

    def __repr__(self):
        return self.__class__.__name__ + "("+repr(self.user_params) + repr(self.cosmo_params)+")"

    def __hash__(self):
        return hash(repr(self.user_params)+repr(self.cosmo_params))


class PerturbedField:
    """
    A class containing all perturbed field boxes
    """
    def __init__(self, n):
        self.density = np.zeros(n)
        self.velocity = np.zeros(n)


# ======================================================================================================================
# WRAPPING FUNCTIONS
# ======================================================================================================================
def initial_conditions(user_params=UserParams(), cosmo_params=CosmoParams(), regenerate=False, write=True, direc=None,
                       fname=None):
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

    Returns
    -------
    `~InitialConditions`
        The class which contains the various boxes defining the initial conditions.
    """
    # First initialize memory for the boxes that will be returned.
    boxes = InitialConditions(user_params, cosmo_params)

    # First check whether the boxes already exist.
    if not regenerate:
        if boxes.exists(direc, fname):
            print("Existing init_boxes found, reading them in...")
            boxes.read(direc, fname)
            return boxes

    # Run the C code
    lib.ComputeInitialConditions(user_params(), cosmo_params(), boxes.cstruct)
    boxes.filled = True

    # Optionally do stuff with the result (like writing it)
    if write:
        boxes.write(direc, fname)

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

