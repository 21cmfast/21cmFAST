"""
This is the main wrapper for the underlying 21cmFAST C-code, and the module provides a number of:

* Input-parameter classes which wrap various C structs (these are the classes ending with ``*Params`` or ``*Options``)
* Output objects which simplify access to underlying data structures, such as density, velocity and ionization fields
* Low-level functions which simplify calling the background C functions which populate these output objects given the
  input classes.
* High-level functions which provide the most efficient and simplest way to generate the most commonly desired outputs.

Along with these, the module exposes ``global_params``, which is a simple class providing read/write access to a number
of parameters used throughout the computation which are very rarely varied. These parameters can be accessed as
standard instance attributes of ``global_params``, and listed using ``dir(global_params)``. When set, they are used
globally for all proceeding calculations.

**Input Parameter Classes**

There are four input parameter/option classes, not all of which are required for any given function. They are
:class:`UserParams`, :class:`CosmoParams`, :class:`AstroParams` and :class:`FlagOptions`. Each of them defines a number
of variables, and all of these have default values, to minimize the burden on the user. These defaults are accessed via
the ``_defaults_`` class attribute of each class. The available parameters for each are listed in the documentation
for each class below.

**Output Objects**

The remainder of the classes defined in this module are output classes. These exist to simplify access to large datasets
created within C. Fundamentally, ownership of the data belongs to these classes, and the C functions merely accesses
this and fills it. The various boxes and quantities associated with each output are available as instance attributes.
Along with the output data, each output object contains the various input parameter objects necessary to define it.

.. warning:: These should not be instantiated or filled by the user, but always handled as output objects from the
             various functions contained here. Only the data within the objects should be accessed.

**Low-level functions**

The low-level functions provided here ease the production of the aforementioned output objects. Functions exist for
each low-level C routine, which have been decoupled as far as possible. So, functions exist to create
:func:`initial_conditions`, :func:`perturb_field`, :class:`ionize_box` and so on. Creating a brightness temperature
box (often the desired final output) would generally require calling each of these in turn, as each depends on the
result of a previous function. Nevertheless, each function has the capability of generating the required previous
outputs on-the-fly, so one can instantly call :func:`ionize_box` and get a self-consistent result. Doing so, while
convenient, is sometimes not *efficient*, especially when using inhomogeneous recombinations or the spin temperature
field, which intrinsically require consistent evolution of the ionization field through redshift. In these cases, for
best efficiency it is recommended to either use a customised manual approach to calling these low-level functions, or
to call a higher-level function which optimizes this process.

Finally, note that ``21CMMC`` attempts to optimize the production of the large amount of data via on-disk caching.
By default, if a previous set of data has been computed using the current input parameters, it will be read-in from
a caching repository and returned directly. This behaviour can be tuned in any of the low-level (or high-level)
functions by setting the `write`, `direc`, `regenerate` and `match_seed` parameters (see docs for
:func:`initial_conditions` for details). The function :func:`~query_cache` can be used to search the cache, and return
empty datasets corresponding to each (these can the be filled with the data merely by calling ``.read()`` on any
data set). Conversely, a specific data set can be read and returned as a proper output object by calling the
:func:`readbox` function.


**High-level functions**

As previously mentioned, calling the low-level functions in some cases is non-optimal, especially when full evolution
of the field is required, and thus iteration through a series of redshift. In addition, while
:class:`InitialConditions` and :class:`PerturbedField` are necessary intermediate data, it is *usually* the resulting
brightness temperature which is of most interest, and it is easier to not have to worry about the intermediate steps
explicitly. For these typical use-cases, two high-level functions are available: :func:`run_coeval` and
:func:`run_lightcone`, whose purpose should be self-explanatory. These will optimally run all necessary intermediate
steps (using cached results by default if possible) and return all datasets of interest.


Examples
--------
A typical example of using this module would be the following.

>>> import py21cmmc as p21

Get coeval cubes at redshift 7,8 and 9, without spin temperature or inhomogeneous recombinations:

>>> init, perturb, xHI, Tb = p21.run_coeval(
>>>                              redshift=[7,8,9],
>>>                              cosmo_params=p21.CosmoParams(hlittle=0.7),
>>>                              user_params=p21.UserParams(HII_DIM=100)
>>>                          )

Get coeval cubes at the same redshift, with both spin temperature and inhomogeneous recombinations, pulled from the
natural evolution of the fields:

>>> all_boxes = p21.run_coeval(
>>>                 redshift=[7,8,9],
>>>                 user_params=p21.UserParams(HII_DIM=100),
>>>                 flag_options=p21.FlagOptions(INHOMO_RECO=True),
>>>                 do_spin_temp=True
>>>             )

Get a self-consistent lightcone defined between z1 and z2 (`z_step_factor` changes the logarithmic steps between
redshift that are actually evaluated, which are then interpolated onto the lightcone cells):

>>> lightcone = p21.run_lightcone(redshift=z2, max_redshift=z2, z_step_factor=1.03)
"""
import copy
import numbers
import warnings
from os import path
import yaml

import h5py
import glob
import numpy as np
from astropy.cosmology import Planck15, z_at_value
from astropy import units

from ._21cmfast import ffi, lib
from ._utils import StructWithDefaults, OutputStruct as _OS, _StructWrapper

# Global Options
with open(path.expanduser(path.join("~", '.21CMMC', "config.yml"))) as f:
    config = yaml.load(f)

global_params = _StructWrapper(lib.global_params, ffi)
EXTERNALTABLES = ffi.new("char[]", path.join(path.expanduser("~"), ".21CMMC").encode())
global_params.external_table_path = EXTERNALTABLES


# ======================================================================================================================
# PARAMETER STRUCTURES
# ======================================================================================================================
class CosmoParams(StructWithDefaults):
    """
    Cosmological parameters (with defaults) which translates to a C struct.

    To see default values for each parameter, use ``CosmoParams._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

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
    _ffi = ffi

    _defaults_ = dict(
        RANDOM_SEED=None,
        SIGMA_8=0.82,
        hlittle=Planck15.h,
        OMm=Planck15.Om0,
        OMb=Planck15.Ob0,
        POWER_INDEX=0.97
    )

    @property
    def RANDOM_SEED(self):
        """
        The current random seed for the cosmology, which determines the initial conditions.
        """
        while not self._RANDOM_SEED:
            self._RANDOM_SEED = int(np.random.randint(1, 1e12))
        return self._RANDOM_SEED

    @property
    def OMl(self):
        """
        Omega lambda, dark energy density.
        """
        return 1 - self.OMm

    @property
    def cosmo(self):
        """
        Return an astropy cosmology object for this cosmology.
        """
        return Planck15.clone(H0=self.hlittle*100, Om0=self.OMm, Ob0=self.OMb)


class UserParams(StructWithDefaults):
    """
    Structure containing user parameters (with defaults).

    To see default values for each parameter, use ``UserParams._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

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
    _ffi = ffi

    _defaults_ = dict(
        BOX_LEN=150.0,
        DIM=None,
        HII_DIM=50,
        USE_FFTW_WISDOM=False,
    )

    @property
    def DIM(self):
        """
        Number of cells for the high-res box (sampling ICs) along a principal axis. Dynamically generated.
        """
        return self._DIM or 4 * self.HII_DIM

    @property
    def tot_fft_num_pixels(self):
        "Number of pixels in the high-res box."
        return self.DIM ** 3

    @property
    def HII_tot_num_pixels(self):
        "Number of pixels in the low-res box."
        return self.HII_DIM ** 3


class AstroParams(StructWithDefaults):
    """
    Astrophysical parameters.

    To see default values for each parameter, use ``AstroParams._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

    Parameters
    ----------
    INHOMO_RECO : bool, optional
        Whether inhomogeneous recombinations are being calculated. This is not a part of the astro parameters structure,
        but is required by this class to set some default behaviour.
    EFF_FACTOR_PL_INDEX : float, optional
    HII_EFF_FACTOR : float, optional
    R_BUBBLE_MAX : float, optional
        Default is 50 if `INHOMO_RECO` is True, or 15.0 if not.
    ION_Tvir_MIN : float, optional
    L_X : float, optional
    NU_X_THRESH : float, optional
    X_RAY_SPEC_INDEX : float, optional
    X_RAY_Tvir_MIN : float, optional
        Default is `ION_Tvir_MIN`.
    F_STAR : float, optional
    t_STAR : float, optional
    N_RSD_STEPS : float, optional
    """

    _ffi = ffi

    _defaults_ = dict(
        EFF_FACTOR_PL_INDEX=0.0,
        HII_EFF_FACTOR=30.0,
        R_BUBBLE_MAX=None,
        ION_Tvir_MIN=4.69897,
        L_X=40.0,
        NU_X_THRESH=500.0,
        X_RAY_SPEC_INDEX=1.0,
        X_RAY_Tvir_MIN=None,
        F_STAR=0.05,
        t_STAR=0.5,
        N_RSD_STEPS=20,
    )

    def __init__(self, *args, INHOMO_RECO=False, **kwargs):
        # TODO: should try to get inhomo_reco out of here... just needed for default of R_BUBBLE_MAX.
        self.INHOMO_RECO = INHOMO_RECO
        super().__init__(*args, **kwargs)

    def convert(self, key, val):
        if key in ['ION_Tvir_MIN', "L_X", "X_RAY_Tvir_MIN"]:
            return 10 ** val
        else:
            return val

    @property
    def R_BUBBLE_MAX(self):
        "Maximum radius of bubbles to be searched. Set dynamically."
        if not self._R_BUBBLE_MAX:
            return 50.0 if self.INHOMO_RECO else 15.0
        else:
            return self._R_BUBBLE_MAX

    # @property
    # def ION_Tvir_MIN(self):
    #     "Minimum virial temperature of ionization (unlogged)."
    #     return 10 ** self._ION_Tvir_MIN
    #
    # @property
    # def L_X(self):
    #     "X-ray luminosity (unlogged)"
    #     return 10 ** self._L_X

    @property
    def X_RAY_Tvir_MIN(self):
        "Minimum virial temperature of X-ray emitting sources (unlogged and set dynamically)."
        return self._X_RAY_Tvir_MIN if self._X_RAY_Tvir_MIN else self.ION_Tvir_MIN


class FlagOptions(StructWithDefaults):
    """
    Flag-style options for the ionization routines.

    To see default values for each parameter, use ``FlagOptions._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

    Parameters
    ----------
    USE_MASS_DEPENDENT_ZETA : bool, optional
        Set to True if using new parameterization.

    SUBCELL_RSDS : bool, optional
        Add sub-cell RSDs (currently doesn't work if Ts is not used)

    INHOMO_RECO : bool, optional
        Whether to perform inhomogeneous recombinations

    USE_TS_FLUCT : bool, optional
        Whether to perform IGM spin temperature fluctuations (i.e. X-ray heating)
    """

    _ffi = ffi

    _defaults_ = dict(
        USE_MASS_DEPENDENT_ZETA=False,
        SUBCELL_RSD=False,
        INHOMO_RECO=False,
        USE_TS_FLUCT=False,        
    )


# ======================================================================================================================
# OUTPUT STRUCTURES
# ======================================================================================================================
class _OutputStruct(_OS):
    _global_params = global_params

    def __init__(self, user_params=UserParams(), cosmo_params=CosmoParams(), **kwargs):
        super().__init__(user_params=user_params, cosmo_params=cosmo_params, **kwargs)

    _ffi = ffi


class _OutputStructZ(_OutputStruct):
    _inputs = ['redshift', 'user_params', 'cosmo_params']

    # def __init__(self, redshift, user_params=UserParams(), cosmo_params=CosmoParams(), **kwargs):
    #     super().__init__(user_params=user_params, cosmo_params=cosmo_params, redshift=float(redshift), **kwargs)
    #     self._name += "_z%.4f" % self.redshift


class InitialConditions(_OutputStruct):
    """
    A class containing all initial conditions boxes.
    """

    def _init_arrays(self):
        self.lowres_density = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vx = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vy = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vz = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vx_2LPT = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vy_2LPT = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vz_2LPT = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.hires_density = np.zeros(self.user_params.tot_fft_num_pixels, dtype=np.float32)

        shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)
        self.lowres_density.shape = shape
        self.lowres_vx.shape = shape
        self.lowres_vy.shape = shape
        self.lowres_vz.shape = shape
        self.lowres_vx_2LPT.shape = shape
        self.lowres_vy_2LPT.shape = shape
        self.lowres_vz_2LPT.shape = shape
        self.hires_density.shape = (self.user_params.DIM, self.user_params.DIM, self.user_params.DIM)


class PerturbedField(_OutputStructZ):
    """
    A class containing all perturbed field boxes
    """

    def _init_arrays(self):
        self.density = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.velocity = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)

        self.density.shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)
        self.velocity.shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)


class IonizedBox(_OutputStructZ):
    "A class containing all ionized boxes"
    _inputs = ['redshift', 'user_params', 'cosmo_params', 'flag_options', 'astro_params']

    def __init__(self, astro_params=None, flag_options=FlagOptions(), first_box=False, **kwargs):
        if astro_params is None:
            astro_params = AstroParams(INHOMO_RECO=flag_options.INHOMO_RECO)
        self.first_box = first_box

        super().__init__(astro_params=astro_params, flag_options=flag_options, **kwargs)

    def _init_arrays(self):
        # ionized_box is always initialised to be neutral for excursion set algorithm. Hence np.ones instead of np.zeros
        self.xH_box = np.ones(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.Gamma12_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.z_re_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.dNrec_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)

        shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)
        self.xH_box.shape = shape
        self.Gamma12_box.shape = shape
        self.z_re_box.shape = shape
        self.dNrec_box.shape = shape


class TsBox(IonizedBox):
    "A class containing all spin temperature boxes"

    def _init_arrays(self):
        self.Ts_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.x_e_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.Tk_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)

        self.Ts_box.shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)
        self.x_e_box.shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)
        self.Tk_box.shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)


class BrightnessTemp(IonizedBox):
    "A class containin the brightness temperature box."

    def _init_arrays(self):
        self.brightness_temp = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)

        self.brightness_temp.shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)


# ======================================================================================================================
# HELPER FUNCTIONS
# ======================================================================================================================
def _check_compatible_inputs(*datasets, ignore_redshift=False):
    done = []
    for i, d in enumerate(datasets):
        if d is None:
            continue

        for inp in d._inputs:
            if ignore_redshift and inp == "redshift":
                continue

            if inp not in done:
                for j, d2 in enumerate(datasets[(i + 1):]):
                    if d2 is None:
                        continue

                    if inp in d2._inputs and getattr(d, inp) != getattr(d2, inp):
                        raise ValueError("%s and %s are incompatible" % (d.__class__.__name__, d2.__class__.__name__))
                done += [inp]


def _get_inputs(defaults, *structs):
    for i in range(len(defaults)):
        k = ''.join('_' + c.lower() if c.isupper() else c for c in defaults[i].__class__.__name__).strip('_')

        for s in structs:
            if s is None:
                continue

            if hasattr(s, k):
                defaults[i] = getattr(s, k)
                break

    return defaults


def _get_redshift(redshift, *structs):
    if redshift is None and all([s is None for s in structs]):
        raise ValueError("Either redshift must be provided, or a data set containing it.")

    for s in structs:
        if hasattr(s, "redshift"):
            return s.redshift

    return redshift


# ======================================================================================================================
# WRAPPING FUNCTIONS
# ======================================================================================================================
def initial_conditions(user_params=None, cosmo_params=None, regenerate=False, write=True, direc=None,
                       match_seed=False):
    """
    Compute initial conditions.

    Parameters
    ----------
    user_params : :class:`~UserParams` instance, optional
        Defines the overall options and parameters of the run.

    cosmo_params : :class:`~CosmoParams` instance, optional
        Defines the cosmological parameters used to compute initial conditions.

    regenerate : bool, optional
        Whether to force regeneration of data, even if matching cached data is found. This is applied recursively to
        any potential sub-calculations. It is ignored in the case of dependent data only if that data is explicitly
        passed to the function.

    write : bool, optional
        Whether to write results to file (i.e. cache). This is recursively applied to any potential sub-calculations.

    direc : str, optional
        The directory in which to search for the boxes and write them. By default, this is the directory given by
        ``boxdir`` in the configuration file, ``~/.21CMMC/config.yml``. Note that for *reading* data, while the
        specified `direc` is searched first, the default directory will *also* be searched if no appropriate data is
        found in `direc`. This is recursively applied to any potential sub-calculations.

    match_seed : bool, optional
        If `False`, then the caching mechanism will consider any initial conditions boxes with the same defining parameters
        to be a match, without considering the random seed. Otherwise, the random seed must also be matched to be read
        in from cache. Any other kinds of boxes will always require matching the seed to be self-consistent.

    Returns
    -------
    :class:`~InitialConditions`
    """
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)

    # Initialize memory for the boxes that will be returned.
    boxes = InitialConditions(user_params, cosmo_params)

    # First check whether the boxes already exist.
    if not regenerate:
        try:
            boxes.read(direc, match_seed)
            print("Existing init_boxes found and read in.")
            return boxes
        except IOError:
            pass

    # Run the C code
    lib.ComputeInitialConditions(user_params(), cosmo_params(), boxes())
    boxes.filled = True
    boxes._expose()

    # Optionally do stuff with the result (like writing it)
    if write:
        boxes.write(direc)

    return boxes


def perturb_field(redshift, init_boxes=None, user_params=None, cosmo_params=None,
                  regenerate=False, write=True, direc=None,
                  match_seed=False):
    """
    Compute a perturbed field at a given redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to compute the perturbed field.

    init_boxes : :class:`~InitialConditions`, optional
        If given, these initial conditions boxes will be used, otherwise initial conditions will be generated. If given,
        the user and cosmo params will be set from this object.

    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.

    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.

    Returns
    -------
    :class:`~PerturbedField`

    Other Parameters
    ----------------
    regenerate, write, direc, match_seed:
        See docs of :func:`initial_conditions` for more information.

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
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)

    # Try setting the user/cosmo params via the init_boxes
    if init_boxes is not None:
        user_params, cosmo_params = _get_inputs([user_params, cosmo_params], init_boxes)
        match_seed = True  # Need to match seed if matching an init box.

    # Initialize perturbed boxes.
    fields = PerturbedField(redshift=redshift, user_params=user_params, cosmo_params=cosmo_params)

    # Check whether the boxes already exist
    if not regenerate:
        try:
            fields.read(direc, match_seed=match_seed)
            print("Existing z=%s perturb_field boxes found and read in." % redshift)
            return fields
        except IOError:
            pass

    # Make sure we've got computed init boxes.
    if init_boxes is None or not init_boxes.filled:
        init_boxes = initial_conditions(
            user_params, cosmo_params, regenerate=regenerate, write=write,
            direc=direc, match_seed=match_seed
        )

    # Run the C Code
    lib.ComputePerturbField(redshift, user_params(), cosmo_params(), init_boxes(), fields())
    fields.filled = True
    fields._expose()

    # Optionally do stuff with the result (like writing it)
    if write:
        fields.write(direc)

    return fields


def ionize_box(astro_params=None, flag_options=None,
               redshift=None, perturbed_field=None,
               previous_ionize_box=None, z_step_factor=1.02, z_heat_max=None,
               do_spin_temp=False, spin_temp=None,
               init_boxes=None, cosmo_params=None, user_params=None,
               regenerate=False, write=True, direc=None,
               match_seed=False):
    """
    Compute an ionized box at a given redshift.

    This function has various options for how the evolution of the ionization is computed (if at all). See the Notes
    below for details.

    Parameters
    ----------
    astro_params: :class:`~AstroParams` instance, optional
        The astrophysical parameters defining the course of reionization.

    flag_options: :class:`~FlagOptions` instance, optional
        Some options passed to the reionization routine.

    redshift : float, optional
        The redshift at which to compute the ionized box. If `perturbed_field` is given, its inherent redshift
        will take precedence over this argument. If not, this argument is mandatory.

    perturbed_field : :class:`~PerturbField`, optional
        If given, this field will be used, otherwise it will be generated. To be generated, either `init_boxes` and
        `redshift` must be given, or `user_params`, `cosmo_params` and `redshift`.

    init_boxes : :class:`~InitialConditions` , optional
        If given, and `perturbed_field` *not* given, these initial conditions boxes will be used to generate the
        perturbed field, otherwise initial conditions will be generated on the fly. If given,
        the user and cosmo params will be set from this object.

    previous_ionize_box: :class:`IonizedBox` or float, optional
        An ionized box at higher redshift. This is only used if `INHOMO_RECO` and/or `do_spin_temp` are true. If either
        of these are true, and this is not given, then it will be assumed that this is the "first box", i.e. that it
        can be populated accurately without knowing source statistics.

    z_step_factor: float, optional
        A factor greater than unity, which specifies the logarithmic steps in redshift with which the spin temperature
        box is evolved.

    z_heat_max: float, optional
        The maximum redshift at which to search for heating sources. Practically, this defines the limit in redshift
        at which the spin temperature can be defined purely from the background perturbed field rather than by evolving
        from a previous spin temperature field. Default is the global parameter `Z_HEAT_MAX`.

    do_spin_temp: bool, optional
        Whether to use the spin temperature.

    spin_temp: :class:`TsBox` or None, optional
        A spin-temperature box, only required if `do_spin_temp` is True.
        If None, will try to read in a spin temp box at the current redshift, and failing that will try to
        automatically create one, using the previous ionized box redshift as the previous spin temperature redshift.

    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.

    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.

    Returns
    -------
    :class:`~IonizedBox`
        An object containing the ionized box data.

    Other Parameters
    ----------------
    regenerate, write, direc, match_seed:
        See docs of :func:`initial_conditions` for more information.

    Notes
    -----

    Typically, the ionization field at any redshift is dependent on the evolution of xHI up until
    that redshift, which necessitates providing a previous ionization field to define the current one. This
    function provides several options for doing so. First, if neither the spin temperature field, nor inhomogeneous
    recombinations (specified in flag options) are used, no evolution needs to be done. Otherwise, either (in order of
    precedence) (i) a specific previous :class`~IonizedBox` object is provided, which will be used directly,
    (ii) a previous redshift is provided, for which a cached field on disk will be sought, (iii) a step factor is
    provided which recursively steps through redshift, calculating previous fields up until Z_HEAT_MAX, and returning
    just the final field at the current redshift, or (iv) the function is instructed to treat the current field as
    being an initial "high-redshift" field such that specific sources need not be found and evolved.

    .. note:: If a previous specific redshift is given, but no cached field is found at that redshift, the previous
              ionization field will be evaluated based on `z_step_factor`.

    Examples
    --------
    By default, no spin temperature is used, and neither are inhomogeneous recombinations, so that no evolution is
    required, thus the following will compute a coeval ionization box:

    >>> xHI = ionize_box(redshift=7.0)

    However, if either of those options are true, then a full evolution will be required:

    >>> xHI = ionize_box(redshift=7.0, do_spin_temp=True, flag_options=FlagOptions(INHOMO_RECO=True))

    This will by default evolve the field from a redshift of *at least* `Z_HEAT_MAX` (a global parameter), in logarithmic
    steps of `z_step_factor`. Thus to change these:

    >>> xHI = ionize_box(redshift=7.0, z_step_factor=1.2, z_heat_max=15.0, do_spin_temp=True)

    Alternatively, one can pass an exact previous redshift, which will be sought in the disk cache, or evaluated:

    >>> ts_box = ionize_box(redshift=7.0, previous_ionize_box=8.0, do_spin_temp=True)

    Beware that doing this, if the previous box is not found on disk, will continue to evaluate prior boxes based on the
    `z_step_factor`. Alternatively, one can pass a previous :class:`~IonizedBox`:

    >>> xHI_0 = ionize_box(redshift=8.0, do_spin_temp=True)
    >>> xHI = ionize_box(redshift=7.0, previous_ionize_box=xHI_0, do_spin_temp=True)

    Again, the first line here will implicitly use `z_step_factor` to evolve the field from ~`Z_HEAT_MAX`. Note that
    in the second line, all of the input parameters are taken directly from `xHI_0` so that they are consistent.
    Finally, one can force the function to evaluate the current redshift as if it was beyond Z_HEAT_MAX so that it
    depends only on itself:

    >>> xHI = ionize_box(redshift=7.0, z_step_factor=None, do_spin_temp=True)

    This is usually a bad idea, and will give a warning, but it is possible.

    As the function recursively evaluates previous redshift, the previous spin temperature fields will also be
    consistently recursively evaluated. Only the final ionized box will actually be returned and kept in memory, however
    intervening results will by default be cached on disk. One can also pass an explicit spin temperature obj:

    >>> ts = spin_temperature(redshift=7.0)
    >>> xHI = ionize_box(redshift=7.0, spin_temp=ts)

    If automatic recursion is used, then it is done in such a way that no large boxes are kept around in memory for
    longer than they need to be (only two at a time are required).
    """
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    flag_options = FlagOptions(flag_options)
    astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

    if spin_temp is not None or perturbed_field is not None or init_boxes is not None:
        match_seed = True
        _check_compatible_inputs(spin_temp, perturbed_field, init_boxes)
        _check_compatible_inputs(spin_temp, previous_ionize_box, ignore_redshift=True)

    cosmo_params, user_params, astro_params, flag_options = _get_inputs(
        [cosmo_params, user_params, astro_params, flag_options],
        spin_temp, previous_ionize_box, perturbed_field, init_boxes
    )

    if spin_temp is not None:
        do_spin_temp = True

    # Set the upper limit on redshift at which we require a previous spin temp box.
    if z_heat_max is not None:
        global_params.Z_HEAT_MAX = z_heat_max

    if spin_temp is not None and not isinstance(spin_temp, TsBox):
        raise ValueError("spin_temp must be a TsBox instance")

    redshift = _get_redshift(redshift, perturbed_field, spin_temp)

    # Set the default astro params, using the INHOMO_RECO flag.
    if astro_params is None:
        astro_params = AstroParams(flag_options.INHOMO_RECO)

    box = IonizedBox(
        first_box=((1 + redshift) * z_step_factor - 1) > global_params.Z_HEAT_MAX and (
                    not isinstance(previous_ionize_box, IonizedBox) or not previous_ionize_box.filled),
        user_params=user_params, cosmo_params=cosmo_params,
        redshift=redshift, astro_params=astro_params, flag_options=flag_options
    )

    # Check whether the boxes already exist
    if not regenerate:
        try:
            box.read(direc, match_seed=match_seed)
            print("Existing z=%s ionized boxes found and read in." % redshift)
            return box
        except IOError:
            pass

    # EVERYTHING PAST THIS POINT ONLY HAPPENS IF THE BOX DOESN'T ALREADY EXIST
    # ------------------------------------------------------------------------
    # Get the previous redshift
    if flag_options.INHOMO_RECO or do_spin_temp:

        if previous_ionize_box is not None:
            if hasattr(previous_ionize_box, "redshift"):
                prev_z = previous_ionize_box.redshift
            elif isinstance(previous_ionize_box, numbers.Number):
                prev_z = previous_ionize_box
            else:
                raise ValueError("previous_ionize_box must be an IonizedBox or a float")
        elif z_step_factor is not None:
            prev_z = (1 + redshift) * z_step_factor - 1
        else:
            prev_z = None
            if redshift < global_params.Z_HEAT_MAX:
                warnings.warn(
                    "Attempting to evaluate ionization field at z=%s as if it was beyond Z_HEAT_MAX=%s" % (
                        redshift, global_params.Z_HEAT_MAX))

        # Ensure the previous spin temperature has a higher redshift than this one.
        if prev_z and prev_z <= redshift:
            raise ValueError("Previous ionized box must have a higher redshift than that being evaluated.")
    else:
        prev_z = None

    # Dynamically produce the initial conditions.
    if init_boxes is None or not init_boxes.filled:
        init_boxes = initial_conditions(
            user_params=user_params, cosmo_params=cosmo_params,
            regenerate=regenerate, write=write, direc=direc,
            match_seed=match_seed
        )
        match_seed = True

    # Get appropriate previous ionization box
    if not isinstance(previous_ionize_box, IonizedBox):
        # If we are beyond Z_HEAT_MAX, just make an empty box
        if prev_z is None or prev_z > global_params.Z_HEAT_MAX:
            previous_ionize_box = IonizedBox(redshift=0)

        # Otherwise recursively create new previous box.
        else:
            previous_ionize_box = ionize_box(
                astro_params=astro_params, flag_options=flag_options, redshift=prev_z,
                z_step_factor=z_step_factor, z_heat_max=z_heat_max,
                do_spin_temp=do_spin_temp,
                init_boxes=init_boxes, regenerate=regenerate, write=write, direc=direc,
                match_seed=match_seed
            )
            match_seed = True

    # Dynamically produce the perturbed field.
    if perturbed_field is None or not perturbed_field.filled:
        perturbed_field = perturb_field(
            redshift=redshift, init_boxes=init_boxes, # Definition have init_boxes here.
            regenerate=regenerate, write=write, direc=direc,
            match_seed=match_seed
        )
        match_seed = True

    # Set empty spin temp box if necessary.
    if not do_spin_temp:
        spin_temp = TsBox(redshift=0)
    elif spin_temp is None:
        spin_temp = spin_temperature(
            perturbed_field=perturbed_field, previous_spin_temp=prev_z,
            flag_options=flag_options, user_params=user_params,
            match_seed=match_seed, direc=direc, write=write, regenerate=regenerate
        )

    # Run the C Code
    lib.ComputeIonizedBox(redshift, previous_ionize_box.redshift, user_params(),
                          cosmo_params(), astro_params(), flag_options(), perturbed_field(),
                          previous_ionize_box(), do_spin_temp, spin_temp(), box())

    box.filled = True
    box._expose()

    # Optionally do stuff with the result (like writing it)
    if write:
        box.write(direc)

    return box


def spin_temperature(astro_params=None, flag_options=FlagOptions(), redshift=None, perturbed_field=None,
                     previous_spin_temp=None, z_step_factor=1.02, z_heat_max=None,
                     init_boxes=None, cosmo_params=CosmoParams(), user_params=UserParams(), regenerate=False,
                     write=True, direc=None,
                     match_seed=False):
    """
    Compute spin temperature boxes at a given redshift.

    See the notes below for how the spin temperature field is evolved through redshift.

    Parameters
    ----------
    astro_params: :class:`~AstroParams`, optional
        The astrophysical parameters defining the course of reionization.

    flag_options: :class:`~FlagOptions`, optional
        Some options passed to the reionization routine.

    redshift : float, optional
        The redshift at which to compute the ionized box. If not given, the redshift from `perturbed_field` will be used.
        Either `redshift`, `perturbed_field` or both must be given.

    perturbed_field : :class:`~PerturbField`, optional
        If given, this field will be used, otherwise it will be generated. To be generated, either `init_boxes` and
        `redshift` must be given, or `user_params`, `cosmo_params` and `redshift`. By default, this will be generated
        at the same redshift as the spin temperature box. However, it does not need to be defined at the same redshift.
        If at a different redshift, it will be linearly evolved to the redshift of the spin temperature box.

    previous_spin_temp : :class:`TsBox` or float, optional
        The previous spin temperature box, or its redshift. This redshift must be greater than `redshift`. If a
        redshift, then this will try to read in the previous spin temp box at this redshift or generate it.

    z_step_factor: float, optional
        A factor greater than unity, which specifies the logarithmic steps in redshift with which the spin temperature
        box is evolved. If None, the code will assume that this is the first box in the evolution process, and generate
        the spin temp directly from the perturbed field.

    z_heat_max: float, optional
        The maximum redshift at which to search for heating sources. Practically, this defines the limit in redshift
        at which the spin temperature can be defined purely from the background perturbed field rather than by evolving
        from a previous spin temperature field. Default is the global parameter `Z_HEAT_MAX`.

    init_boxes : :class:`~InitialConditions`, optional
        If given, and `perturbed_field` *not* given, these initial conditions boxes will be used to generate the
        perturbed field, otherwise initial conditions will be generated on the fly. If given,
        the user and cosmo params will be set from this object.

    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.

    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.

    Returns
    -------
    :class:`~TsBox`
        An object containing the spin temperature box data.

    Other Parameters
    ----------------
    regenerate, write, direc, match_seed:
        See docs of :func:`initial_conditions` for more information.

    Notes
    -----

    Typically, the spin temperature field at any redshift is dependent on the evolution of spin temperature up until
    that redshift, which necessitates providing a previous spin temperature field to define the current one. This
    function provides several options for doing so. Either (in order of precedence) (i) a specific previous spin
    temperature object is provided, which will be used directly, (ii) a previous redshift is provided, for which a
    cached field on disk will be sought, (iii) a step factor is provided which recursively steps through redshift,
    calculating previous fields up until Z_HEAT_MAX, and returning just the final field at the current redshift, or
    (iv) the function is instructed to treat the current field as being an initial "high-redshift" field such that
    specific sources need not be found and evolved.

    .. note:: If a previous specific redshift is given, but no cached field is found at that redshift, the previous
              spin temperature field will be evaluated based on `z_step_factor`.

    Examples
    --------
    To calculate and return a fully evolved spin temperature field at a given redshift (with default input parameters),
    simply use:

    >>> ts_box = spin_temperature(redshift=7.0)

    This will by default evolve the field from a redshift of *at least* `Z_HEAT_MAX` (a global parameter), in logarithmic
    steps of `z_step_factor`. Thus to change these:

    >>> ts_box = spin_temperature(redshift=7.0, z_step_factor=1.2, z_heat_max=15.0)

    Alternatively, one can pass an exact previous redshift, which will be sought in the disk cache, or evaluated:

    >>> ts_box = spin_temperature(redshift=7.0, previous_spin_temp=8.0)

    Beware that doing this, if the previous box is not found on disk, will continue to evaluate prior boxes based on the
    `z_step_factor`. Alternatively, one can pass a previous spin temperature box:

    >>> ts_box1 = spin_temperature(redshift=8.0)
    >>> ts_box = spin_temperature(redshift=7.0, previous_spin_temp=ts_box1)

    Again, the first line here will implicitly use `z_step_factor` to evolve the field from ~`Z_HEAT_MAX`. Note that
    in the second line, all of the input parameters are taken directly from `ts_box1` so that they are consistent.
    Finally, one can force the function to evaluate the current redshift as if it was beyond Z_HEAT_MAX so that it
    depends only on itself:

    >>> ts_box = spin_temperature(redshift=7.0, z_step_factor=None)

    This is usually a bad idea, and will give a warning, but it is possible.
    """
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    flag_options = FlagOptions(flag_options)
    astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

    if perturbed_field is not None or previous_spin_temp is not None or init_boxes is not None:
        match_seed = True
        if previous_spin_temp is None or isinstance(previous_spin_temp, _OutputStruct):
            # This switch accounts for the case where previous_spin_temp is a float.
            _check_compatible_inputs(perturbed_field, init_boxes, previous_spin_temp, ignore_redshift=True)
        else:
            _check_compatible_inputs(perturbed_field, init_boxes, ignore_redshift=True)

    cosmo_params, user_params, astro_params, flag_options = _get_inputs(
        [cosmo_params, user_params, astro_params, flag_options],
        previous_spin_temp, perturbed_field, init_boxes
    )

    # Set the upper limit on redshift at which we require a previous spin temp box.
    if z_heat_max is not None:
        global_params.Z_HEAT_MAX = z_heat_max

    redshift = _get_redshift(redshift, perturbed_field)

    # Set the default astro params, using the INHOMO_RECO flag.
    if astro_params is None:
        astro_params = AstroParams(flag_options.INHOMO_RECO)

    box = TsBox(
        first_box=((1 + redshift) * z_step_factor - 1) > global_params.Z_HEAT_MAX and (
                    not isinstance(previous_spin_temp, IonizedBox) or not previous_spin_temp.filled),
        user_params=user_params, cosmo_params=cosmo_params,
        redshift=redshift, astro_params=astro_params, flag_options=flag_options
    )

    # Check whether the boxes already exist on disk.
    if not regenerate:
        try:
            box.read(direc, match_seed=match_seed)
            print("Existing z=%s spin_temp boxes found and read in." % redshift)
            return box
        except IOError:
            pass

    # EVERYTHING PAST THIS POINT ONLY HAPPENS IF THE BOX DOESN'T ALREADY EXIST
    # ------------------------------------------------------------------------
    # Get the previous redshift
    if previous_spin_temp is not None:
        if hasattr(previous_spin_temp, "redshift"):
            prev_z = previous_spin_temp.redshift
        elif isinstance(previous_spin_temp, numbers.Number):
            prev_z = previous_spin_temp
    elif z_step_factor is not None:
        prev_z = (1 + redshift) * z_step_factor - 1
    else:
        prev_z = None
        if redshift < global_params.Z_HEAT_MAX:
            warnings.warn("Attempting to evaluate spin temperature field at z=%s as if it was beyond Z_HEAT_MAX=%s" % (
            redshift, global_params.Z_HEAT_MAX))

    # Ensure the previous spin temperature has a higher redshift than this one.
    if prev_z and prev_z <= redshift:
        raise ValueError("Previous spin temperature box must have a higher redshift than that being evaluated.")

    # Dynamically produce the initial conditions.
    if init_boxes is None or not init_boxes.filled:
        init_boxes = initial_conditions(
            user_params=user_params, cosmo_params=cosmo_params,
            regenerate=regenerate, write=write, direc=direc,
            match_seed=match_seed
        )
        match_seed = True

    # Create appropriate previous_spin_temp
    if not isinstance(previous_spin_temp, TsBox):
        if prev_z > global_params.Z_HEAT_MAX or prev_z is None:
            previous_spin_temp = TsBox(redshift=0)
        else:
            previous_spin_temp = spin_temperature(
                init_boxes=init_boxes,
                astro_params=astro_params, flag_options=flag_options, redshift=prev_z,
                z_step_factor=z_step_factor, z_heat_max=z_heat_max,
                regenerate=regenerate, write=write, direc=direc,
                match_seed=match_seed
            )

    # Dynamically produce the perturbed field.
    if perturbed_field is None or not perturbed_field.filled:
        perturbed_field = perturb_field(
            redshift=redshift,
            init_boxes=init_boxes, user_params=user_params, cosmo_params=cosmo_params,
            regenerate=regenerate, write=write, direc=direc,
            match_seed=match_seed
        )
        match_seed = True

    # Run the C Code
    lib.ComputeTsBox(redshift, previous_spin_temp.redshift, perturbed_field.user_params(),
                     perturbed_field.cosmo_params(), astro_params(), perturbed_field.redshift, perturbed_field(),
                     previous_spin_temp(), box())
    box.filled = True
    box._expose()

    # Optionally do stuff with the result (like writing it)
    if write:
        box.write(direc)

    return box


def brightness_temperature(ionized_box, perturb_field, spin_temp=None):
    """
    Compute a coeval brightness temperature box.

    Parameters
    ----------
    ionized_box: :class:`IonizedBox`
        A pre-computed ionized box.

    perturb_field: :class:`PerturbedField`
        A pre-computed perturbed field at the same redshift as `ionized_box`.

    spin_temp: :class:`TsBox`, optional
        A pre-computed spin temperature, at the same redshift as the other boxes.

    Returns
    -------
    :class:`BrightnessTemp` instance.
    """
    _check_compatible_inputs(ionized_box, perturb_field, spin_temp)

    if spin_temp is None:
        saturated_limit = True
        #        spin_temp = ffi.new("struct TsBox*")
        spin_temp = TsBox(redshift=0)

    else:
        saturated_limit = False

    #    if spin_temp.redshift != ionized_box.redshift != perturb_field.redshift:
    #        raise ValueError("all box redshift must be the same.")

    #    if spin_temp.user_params != ionized_box.user_params != perturb_field.user_params:
    #        raise ValueError("all box user_params must be the same")

    #    if spin_temp.cosmo_params != ionized_box.cosmo_params != perturb_field.cosmo_params:
    #        raise ValueError("all box cosmo_params must be the same")

    #    if spin_temp.astro_params != ionized_box.astro_params:
    #        raise ValueError("all box astro_params must be the same")

    box = BrightnessTemp(user_params=ionized_box.user_params, cosmo_params=ionized_box.cosmo_params,
                         astro_params=ionized_box.astro_params, flag_options=ionized_box.flag_options,
                         redshift=ionized_box.redshift)

    lib.ComputeBrightnessTemp(ionized_box.redshift, saturated_limit,
                              ionized_box.user_params(), ionized_box.cosmo_params(), ionized_box.astro_params(),
                              ionized_box.flag_options(),
                              spin_temp(), ionized_box(), perturb_field(), box())
    box.filled = True
    box._expose()

    return box


def _logscroll_redshifts(min_redshift, z_step_factor, zmax):
    redshifts = [min_redshift]  # mult by 1.001 is probably bad...
    while redshifts[-1] < zmax:
        redshifts.append((redshifts[-1] + 1.) * z_step_factor - 1.)
    return redshifts[::-1]


def run_coeval(redshift=None, user_params=UserParams(), cosmo_params=CosmoParams(), astro_params=None,
               flag_options=FlagOptions(), do_spin_temp=False, regenerate=False, write=True, direc=None,
               match_seed=False, z_step_factor=1.02, z_heat_max=None, init_box=None, perturb=None):
    """
    Evaluates a coeval ionized box at a given redshift, or multiple redshift.

    This is generally the easiest and most efficient way to generate a set of coeval cubes at a given set of redshift.
    It self-consistently deals with situations in which the field needs to be evolved, and does this with the highest
    memory-efficiency, only returning the desired redshift. All other calculations are by default stored in the
    on-disk cache so they can be re-used at a later time.

    .. note:: User-supplied redshift are *not* used as previous redshift in any scrolling, so that pristine
              log-sampling can be maintained.

    Parameters
    ----------
    redshift: array_like
        A single redshift, or multiple redshift, at which to return results. The minimum of these
        will define the log-scrolling behaviour (if necessary).
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    astro_params: :class:`~AstroParams`, optional
        The astrophysical parameters defining the course of reionization.
    flag_options: :class:`~FlagOptions`, optional
        Some options passed to the reionization routine.
    do_spin_temp: bool, optional
        Whether to use spin temperature in the calculation, or assume the saturated limit.
    z_step_factor: float, optional
        How large the logarithmic steps between redshift are (if required).
    z_heat_max: float, optional
        Controls the global `Z_HEAT_MAX` parameter, which specifies the maximum redshift up to which heating sources
        are required to specify the ionization field. Beyond this, the ionization field is specified directly from
        the perturbed density field.

    Returns
    -------
    init_box: :class:`~InitialConditions`
        The initial conditions data.

    perturb: :class:`~PerturbedField` or list thereof
        The perturbed field at the given redshift(s)

    xHI: :class:`~IonizedBox` or list thereof
        The ionization field(s).

    bt: :class:`~BrightnessTemp` or list thereof
        The brightness temperature box(es)

    Other Parameters
    ----------------
    regenerate, write, direc, match_seed:
        See docs of :func:`initial_conditions` for more information.
    """
    if z_heat_max:
        global_params.Z_HEAT_MAX = z_heat_max

    if init_box is None:  # no need to get cosmo, user params out of it.
        init_box = initial_conditions(user_params, cosmo_params, write=write, regenerate=regenerate, direc=direc,
                                      match_seed=match_seed)

    if perturb is not None:
        _check_compatible_inputs(init_box, *perturb, ignore_redshift=True)

    if redshift is None and perturb is None:
        raise ValueError("Either redshift or perturb must be given")
    if perturb is not None:
        if not hasattr(perturb, "__len__"):
            perturb = [perturb]
        redshift = [p.redshift for p in perturb]

    singleton = False
    if not hasattr(redshift, "__len__"):
        singleton = True
        redshift = [redshift]

    if perturb is None:
        perturb = []
        for z in redshift:
            perturb += [perturb_field(redshift=z, init_boxes=init_box, regenerate=regenerate,
                                      direc=direc, match_seed=True)]

    minarg = np.argmin(redshift)

    # Get the list of redshift we need to scroll through.
    if flag_options.INHOMO_RECO or do_spin_temp:
        redshifts = _logscroll_redshifts(min(redshift), z_step_factor, global_params.Z_HEAT_MAX)
    else:
        redshifts = [min(redshift)]

    # Add in the redshift defined by the user, and sort in order, omitting the minimum,
    # because it won't be exactly reproduced. Turn into a set so that exact matching user-set redshift
    # don't double-up with scrolling ones.
    redshifts += redshift
    redshifts = sorted(list(set(redshifts)), reverse=True)

    ib_tracker = []
    bt = []
    st, ib = None, None  # At first we don't have any "previous" st or ib.
    # Iterate through redshift from top to bottom
    for z in redshifts:
        if do_spin_temp:
            st2 = spin_temperature(
                redshift=z,
                previous_spin_temp=st,
                astro_params=astro_params, flag_options=flag_options,
                perturbed_field=perturb[minarg], regenerate=regenerate,
                write=write, direc=direc, match_seed=True
            )

            if z not in redshift:
                st = st2

        ib2 = ionize_box(
            redshift=z, previous_ionize_box=ib,
            init_boxes=init_box,
            perturbed_field=perturb[redshift.index(z)] if z in redshift else None,
            astro_params=astro_params, flag_options=flag_options,
            spin_temp=st2 if do_spin_temp else None,
            regenerate=regenerate,
            write=write, direc=direc, match_seed=True
        )

        if z not in redshift:
            ib = ib2
        else:
            ib_tracker.append(ib2)
            bt += [brightness_temperature(ib2, perturb[redshift.index(z)], st2 if do_spin_temp else None)]
    #
    # # The last one won't get in because of the dodgy redshift thing
    # ib_tracker += [ib]
    # bt += [brightness_temperature(ib, perturb[minarg], st if do_spin_temp else None)]

    # If a single redshift was passed, then pass back singletons.
    if singleton:
        ib_tracker = ib_tracker[0]
        bt = bt[0]
        perturb = perturb[0]

    return init_box, perturb, ib_tracker, bt


class LightCone:
    def __init__(self, redshift, user_params, cosmo_params, astro_params, flag_options, brightness_temp,
                 node_redshifts=None, global_xHI=None, global_brightness_temp=None):
        self.redshift = redshift
        self.user_params = user_params
        self.cosmo_params = cosmo_params
        self.astro_params = astro_params
        self.flag_options= flag_options
        self.brightness_temp = brightness_temp

        self.node_redshifts = node_redshifts
        self.global_xHI = global_xHI
        self.global_brightness_temp = global_brightness_temp

    @property
    def cell_size(self):
        return self.user_params.BOX_LEN/self.user_params.HII_DIM

    @property
    def lightcone_dimensions(self):
        return (self.user_params.BOX_LEN, self.user_params.BOX_LEN,
                self.n_slices*self.cell_size)

    @property
    def shape(self):
        return self.brightness_temp.shape

    @property
    def n_slices(self):
        return self.shape[-1]

    @property
    def lightcone_coords(self):
        return np.linspace(0, self.lightcone_dimensions[-1], self.n_slices)

    @property
    def lightcone_distances(self):
        return self.cosmo_params.cosmo.comoving_distance(self.redshift).value + self.lightcone_coords

    @property
    def lightcone_redshifts(self):
        return np.array([z_at_value(self.cosmo_params.cosmo.comoving_distance, d *units.Mpc) for d in self.lightcone_distances])


def run_lightcone(redshift, max_redshift=None, user_params=UserParams(), cosmo_params=CosmoParams(), astro_params=None,
                  flag_options=FlagOptions(), do_spin_temp=False, regenerate=False, write=True, direc=None,
                  match_seed=False, z_step_factor=1.02, z_heat_max=None, init_box=None, perturb=None,
                  ):
    """
    Evaluates a full lightcone ending at a given redshift.

    This is generally the easiest and most efficient way to generate a lightcone, though it can be done manually by
    using the lower-level functions which are called by this function.

    Parameters
    ----------
    redshift: float
        The minimum redshift of the lightcone.
    max_redshift: float, optional
        The maximum redshift at which to keep lightcone information. By default, this is equal to `z_heat_max`.
        Note that this is not *exact*, but will be typically slightly exceeded.
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    flag_options: :class:`~FlagOptions`, optional
        Some options passed to the reionization routine.
    do_spin_temp: bool, optional
        Whether to use spin temperature in the calculation, or assume the saturated limit.
    z_step_factor: float, optional
        How large the logarithmic steps between redshift are (if required).
    z_heat_max: float, optional
        Controls the global `Z_HEAT_MAX` parameter, which specifies the maximum redshift up to which heating sources
        are required to specify the ionization field. Beyond this, the ionization field is specified directly from
        the perturbed density field.

    Returns
    -------
    lightcone: :class:`~LightCone`
        The lightcone object.

    Other Parameters
    ----------------
    regenerate, write, direc, match_seed:
        See docs of :func:`initial_conditions` for more information.
    """
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    flag_options = FlagOptions(flag_options)
    astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

    cosmo_params, user_params, astro_params, flag_options = _get_inputs(
        [cosmo_params, user_params, astro_params, flag_options],
        init_box, perturb
    )

    if z_heat_max:
        global_params.Z_HEAT_MAX = z_heat_max

    if init_box is None:  # no need to get cosmo, user params out of it.
        init_box = initial_conditions(user_params, cosmo_params, write=write, regenerate=regenerate, direc=direc,
                                      match_seed=match_seed)

    if perturb is not None:
        _check_compatible_inputs(init_box, perturb, ignore_redshift=True)

    if redshift is None and perturb is None:
        raise ValueError("Either redshift or perturb must be given")

    if perturb is not None:
        redshift = perturb.redshift
    else:
        # The perturb field that we get here is at the *final* redshift, and can be used in TsBox.
        perturb = perturb_field(redshift=redshift, init_boxes=init_box, regenerate=regenerate,
                                direc=direc, match_seed=True)

    max_redshift = global_params.Z_HEAT_MAX if (flag_options.INHOMO_RECO or do_spin_temp or max_redshift is None) else max_redshift

    # Get the redshift through which we scroll and evaluate the ionization field.
    scrollz = _logscroll_redshifts(redshift, z_step_factor, max_redshift)

    # Here set up the lightcone box.
    # Get a length of the lightcone (bigger than it needs to be at first).
    d_at_redshift = cosmo_params.cosmo.comoving_distance(redshift).value
    Ltotal = cosmo_params.cosmo.comoving_distance(scrollz[0] * z_step_factor).value - d_at_redshift
    lc_distances = np.arange(0, Ltotal, user_params.BOX_LEN / user_params.HII_DIM)

    # Use max_redshift to get the actual distances we require.
    Lmax = cosmo_params.cosmo.comoving_distance(max_redshift).value - d_at_redshift
    first_greater = np.argwhere(lc_distances > Lmax)[0][0]

    # Get *at least* as far as max_redshift
    lc_distances = lc_distances[:(first_greater + 1)]
    # lc_redshifts = z_at_value(cosmo_params.cosmo.comoving_distance, lc_distances*units.Mpc)
    n_lightcone = len(lc_distances)
    lc = np.zeros((user_params.HII_DIM, user_params.HII_DIM, n_lightcone))

    scroll_distances = cosmo_params.cosmo.comoving_distance(scrollz).value - d_at_redshift

    # Iterate through redshift from top to bottom
    st, ib, bt = None, None, None
    lc_index = 0
    box_index = 0
    neutral_fraction = np.zeros(len(scrollz))
    global_signal = np.zeros(len(scrollz))

    for iz, z in enumerate(scrollz):
        if do_spin_temp:
            st2 = spin_temperature(
                redshift=z,
                previous_spin_temp=st,
                astro_params=astro_params, flag_options=flag_options,
                perturbed_field=perturb, regenerate=regenerate,
                write=write, direc=direc, match_seed=True
            )

        # Best to get a perturb for this redshift, to pass to brightness_temperature
        this_perturb = perturb_field(redshift=z, init_boxes=init_box, regenerate=regenerate,
                                     direc=direc, match_seed=True)

        ib2 = ionize_box(
            redshift=z, previous_ionize_box=ib,
            init_boxes=init_box,
            perturbed_field=this_perturb,
            astro_params=astro_params, flag_options=flag_options,
            spin_temp=st2 if do_spin_temp else None,
            regenerate=regenerate,
            write=write, direc=direc, match_seed=True
        )

        # FIXME: Need perturb for this redshift, OR get it dynamically in brightness_temperature
        bt2 = brightness_temperature(ib2, this_perturb, st2 if do_spin_temp else None)

        # Save mean/global quantities
        neutral_fraction[iz] = np.mean(ib2.xH_box)
        global_signal[iz] = np.mean(bt2.brightness_temp)

        # HERE IS WHERE WE NEED TO DO THE INTERPOLATION ONTO THE LIGHTCONE!
        if z < max_redshift: # i.e. now redshift is in the bit where the user wants to save the lightcone:
            # Do linear interpolation only.
            prev_d = scroll_distances[iz - 1]
            this_d = scroll_distances[iz]

            # Get the cells that need to be filled on this iteration.
            these_distances = lc_distances[np.logical_and(lc_distances < prev_d, lc_distances >= this_d)]

            n = len(these_distances)
            ind = np.arange(-(box_index+n), -box_index)
            lc[:, :, -(lc_index+n):n_lightcone-lc_index] = (np.abs(prev_d - these_distances)*bt.brightness_temp.take(ind, axis=2, mode='wrap') +
                                               np.abs(this_d - these_distances)*bt2.brightness_temp.take(ind, axis=2, mode='wrap'))/\
                                              (np.abs(prev_d - this_d))

            lc_index += n
            box_index += n

        # Save current ones as old ones.
        if do_spin_temp: st = st2
        ib = ib2
        bt = bt2
    
    return LightCone(
        redshift, user_params, cosmo_params, astro_params, flag_options, lc,
        node_redshifts=scrollz, global_xHI = neutral_fraction, global_brightness_temp=global_signal
    )


def readbox(direc=None, fname=None, hash=None, kind=None, seed=None, load_data=True):
    """
    A function to read in a data set and return an appropriate object for it.

    Parameters
    ----------
    direc : str, optional
        The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
        by the ``config.yml`` in ``.21CMMC``.
    fname: str, optional
        The filename (without directory) of the data set. If given, this will be preferentially used, and must exist.
    hash: str, optional
        The md5 hash of the object desired to be read. Required if `fname` not given.
    kind: str, optional
        The kind of dataset, eg. "InitialConditions". Will be the name of a class defined in :mod:`~wrapper`. Required
        if `fname` not given.
    seed: str or int, optional
        The random seed of the data set to be read. If not given, and filename not given, then a box will be read if
        it matches the kind and hash, with an arbitrary seed.
    load_data: bool, optional
        Whether to read in the data in the data set. Otherwise, only its defining parameters are read.

    Returns
    -------
    dataset:
        An output object, whose type depends on the kind of data set being read.
    """
    direc = direc or path.expanduser(config['boxdir'])

    # We either need fname, or hash and kind.
    if not fname and not (hash and kind):
        raise ValueError("Either fname must be supplied, or kind and hash")

    if fname:
        kind, hash, seed = _parse_fname(fname)

    if not seed:
        fname = kind + "_" + hash + "_r*.h5"
        files = glob.glob(path.join(direc, fname))
        if files:
            fname = files[0]
        else:
            raise IOError("No files exist with that kind and hash.")
    else:
        fname = kind + "_" + hash + "_r" + str(seed) + ".h5"

    # Now, open the file and read in the parameters
    with h5py.File(path.join(direc, fname), 'r') as f:
        # First get items out of attrs.
        top_level = {}
        for k, v in f.attrs.items():
            top_level[k] = v

        # Now descend into each group of parameters
        params = {}
        for grp_nm, grp in f.items():
            if grp_nm != kind:  # is a parameter
                params[grp_nm] = {}
                for k, v in grp.attrs.items():
                    params[grp_nm][k] = v

    # Need to map the parameters to input parameters.
    passed_parameters = {}
    for k, v in params.items():
        if "global_params" in k:
            for kk, vv in v.items():
                setattr(global_params, kk, vv)

        else:
            # The following line takes something like "cosmo_params", turns it into "CosmoParams", and instantiates
            # that particular class with the dictionary parameters.
            passed_parameters[k] = globals()[k.title().replace("_", "")](**v)

    for k, v in top_level.items():
        passed_parameters[k] = v

    # Make an instance of the object.
    inst = globals()[kind](**passed_parameters)

    # Read in the actual data (this avoids duplication of reading data).
    if load_data:
        inst.read(direc=direc, match_seed=True)

    return inst


def _parse_fname(fname):
    try:
        kind = fname.split("_")[0]
        hash = fname.split("_")[1]
        seed = fname.split("_")[-1].split(".")[0][1:]
    except IndexError:
        raise ValueError("fname does not have correct format")

    if kind + "_" + hash + "_r" + seed + ".h5" != fname:
        raise ValueError("fname does not have correct format")

    return kind, hash, seed


def list_datasets(direc=None, kind=None, hash=None, seed=None):
    """
    Yield all datasets which match a given set of filters.

    Can be used to determine parameters of all cached datasets, in conjunction with readbox.

    Parameters
    ----------
    direc : str, optional
        The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
        by the ``config.yml`` in ``.21CMMC``.
    kind: str, optional
        Filter by this kind. Must be one of "InitialConditions", "PerturbedField", "IonizedBox", "TsBox" or "BrightnessTemp".
    hash: str, optional
        Filter by this hash.
    seed: str, optional
        Filter by this seed.

    Yields
    ------
    fname: str
        The filename of the dataset (without directory).
    parts: tuple of strings
        The (kind, hash, seed) of the data set.
    """
    direc = direc or path.expanduser(config['boxdir'])

    kind = kind or "*"
    hash = hash or "*"
    seed = seed or "*"

    fname = path.join(direc, kind + "_" + hash + "_r" + seed + ".h5")

    files = [path.basename(file) for file in glob.glob(fname)]

    for file in files:
        yield file, _parse_fname(file)


def query_cache(direc=None, kind=None, hash=None, seed=None, show=True):
    """
    Walk through the cache, with given filters, and return all un-initialised dataset objects, optionally printing
    their representation to screen.

    Usefor for querying which kinds of datasets are available within the cache, and choosing one to read and use.

    Parameters
    ----------
    direc : str, optional
        The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
        by the ``config.yml`` in ``.21CMMC``.
    kind: str, optional
        Filter by this kind. Must be one of "InitialConditions", "PerturbedField", "IonizedBox", "TsBox" or "BrightnessTemp".
    hash: str, optional
        Filter by this hash.
    seed: str, optional
        Filter by this seed.
    show: bool, optional
        Whether to print out a repr of each object that exists.

    Yields
    ------
    obj:
       Output objects, un-initialized.
    """
    for file, parts in list_datasets(direc, kind, hash, seed):
        cls = readbox(direc, fname=file, load_data=False)
        if show:
            print(file + ": " + str(cls))
        yield file, cls
