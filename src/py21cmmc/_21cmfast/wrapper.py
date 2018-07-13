"""
A thin python wrapper for the 21cmFAST C-code.
"""
from ._21cmfast import ffi, lib
import numpy as np
from ._utils import StructWithDefaults, OutputStruct as _OS
from astropy.cosmology import Planck15
import numbers
import warnings

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
        RANDOM_SEED=None,
        SIGMA_8=0.82,
        hlittle=Planck15.h,
        OMm=Planck15.Om0,
        OMb=Planck15.Ob0,
        POWER_INDEX=0.97
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
        return Planck15.clone(h=self.hlittle, Om0=self.OMm, Ob0=self.OMb)


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
        BOX_LEN=150.0,
        DIM=None,
        HII_DIM=50,
    )

    @property
    def DIM(self):
        return self._DIM or 4 * self.HII_DIM

    @property
    def tot_fft_num_pixels(self):
        return self.DIM ** 3

    @property
    def HII_tot_num_pixels(self):
        return self.HII_DIM ** 3


class AstroParams(StructWithDefaults):
    """
    Astrophysical parameters.

    Parameters
    ----------
    INHOMO_RECO : bool, optional
    EFF_FACTOR_PL_INDEX : float, optional
    HII_EFF_FACTOR : float, optional
    R_BUBBLE_MAX : float, optional
    ION_Tvir_MIN : float, optional
    L_X : float, optional
    NU_X_THRESH : float, optional
    X_RAY_SPEC_INDEX : float, optional
    X_RAY_Tvir_MIN : float, optional
    F_STAR : float, optional
    t_STAR : float, optional
    N_RSD_STEPS : float, optional
    """

    ffi = ffi

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

    def __init__(self, INHOMO_RECO, **kwargs):
        # TODO: should try to get inhomo_reco out of here... just needed for default of R_BUBBLE_MAX.
        self.INHOMO_RECO = INHOMO_RECO
        super().__init__(**kwargs)

    @property
    def R_BUBBLE_MAX(self):
        if not self._R_BUBBLE_MAX:
            return 50.0 if self.INHOMO_RECO else 15.0
        else:
            return self._R_BUBBLE_MAX

    @property
    def ION_Tvir_MIN(self):
        return 10 ** self._ION_Tvir_MIN

    @property
    def L_X(self):
        return 10 ** self._L_X

    @property
    def X_RAY_Tvir_MIN(self):
        return 10 ** self._X_RAY_Tvir_MIN if self._X_RAY_Tvir_MIN else self.ION_Tvir_MIN


class FlagOptions(StructWithDefaults):
    """
    Flag-style options for the ionization routines.

    Parameters
    ----------
    INCLUDE_ZETA_PL : bool, optional
        Should always be zero (have yet to include this option)

    SUBCELL_RSDS : bool, optional
        Add sub-cell RSDs (currently doesn't work if Ts is not used)

    INHOMO_RECO : bool, optional
        Whether to perform inhomogeneous recombinations

    USE_TS_FLUCT : bool, optional
        Whether to perform IGM spin temperature fluctuations (i.e. X-ray heating)
    """

    ffi = ffi

    _defaults_ = dict(
        INCLUDE_ZETA_PL=False,
        SUBCELL_RSD=False,
        INHOMO_RECO=False,
        USE_TS_FLUCT=False,
    )

# ======================================================================================================================
# OUTPUT STRUCTURES
# ======================================================================================================================
class OutputStruct(_OS):
    def __init__(self, user_params=UserParams(), cosmo_params=CosmoParams(), **kwargs):
        super().__init__(user_params=user_params, cosmo_params=cosmo_params, **kwargs)

    ffi = ffi


class OutputStructZ(OutputStruct):
    def __init__(self, redshift, user_params=UserParams(), cosmo_params=CosmoParams(), **kwargs):
        super().__init__(user_params=user_params, cosmo_params=cosmo_params, redshift=float(redshift), **kwargs)
        self._name += "_z%.4f" % self.redshift


class InitialConditions(OutputStruct):
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


class PerturbedField(OutputStructZ):
    """
    A class containing all perturbed field boxes
    """

    def _init_arrays(self):
        self.density = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.velocity = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)


class IonizedBox(OutputStructZ):
    "A class containing all ionized boxes"

    def __init__(self, astro_params=None, flag_options=FlagOptions(), first_box=False, **kwargs):
        if astro_params is None:
            astro_params = AstroParams(flag_options.INHOMO_RECO)
        super().__init__(astro_params=astro_params, flag_options=flag_options, first_box=first_box, **kwargs)

    def _init_arrays(self):
        # ionized_box is always initialised to be neutral, for excursion set algorithm. Hence np.ones instead of np.zeros
        self.xH_box = np.ones(self.user_params.HII_tot_num_pixels, dtype=np.float32) 
        self.Gamma12_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)


class TsBox(IonizedBox):
    "A class containing all spin temperature boxes"

    def _init_arrays(self):
        self.Ts_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.x_e_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)


class BrightnessTemp(IonizedBox):
    "A class containin the brightness temperature box."

    def _init_arrays(self):
        self.brightness_temp = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)


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
    boxes = InitialConditions(user_params, cosmo_params)

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
    boxes._expose()

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
            direc=direc, fname=None
        )

    # Initialize perturbed boxes.
    fields = PerturbedField(redshift, user_params, cosmo_params)

    # Check whether the boxes already exist
    if not regenerate:
        try:
            fields.read(direc, fname, match_seed=match_seed)
            print("Existing perturb_field boxes found and read in.")
            return fields
        except IOError:
            pass

    # Run the C Code
    lib.ComputePerturbField(redshift, user_params(), cosmo_params(), init_boxes(), fields())
    fields.filled = True
    fields._expose()

    # Optionally do stuff with the result (like writing it)
    if write:
        fields.write(direc, fname)

    return fields


def ionize_box(astro_params=None, flag_options=FlagOptions(),
               redshift=None, perturbed_field=None,
               previous_ionize_box=None, z_step_factor = 1.02, z_heat_max = None,
               do_spin_temp=False, spin_temp=None,
               init_boxes=None, cosmo_params=CosmoParams(), user_params=UserParams(),
               regenerate=False, write=True, direc=None,
               fname=None, match_seed=False):
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

    perturbed_field : :class:`~PerturbField` instance, optional
        If given, this field will be used, otherwise it will be generated. To be generated, either `init_boxes` and
        `redshift` must be given, or `user_params`, `cosmo_params` and `redshift`.

    init_boxes : :class:`~InitialConditions` instance, optional
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
    :class:`~IonizedBox`
        An object containing the ionized box data.


    Notes
    -----

    Typically, the ionization field at any redshift is dependent on the evolution of xHI up until
    that redshift, which necessitates providing a previous ionization field to define the current one. This
    function provides several options for doing so. First, if neither the spin temperature field, nor inhomogeneous
    recombinations (specified in flag options) are used, no evolution needs to be done. Otherwise, either (in order of
    precedence) (i) a specific previous :class`~IonizedBox` object is provided, which will be used directly,
    (ii) a previous redshift is provided, for which a cached field on disk will be sought, (iii) a step factor is
    provided which recursively steps through redshifts, calculating previous fields up until Z_HEAT_MAX, and returning
    just the final field at the current redshift, or (iv) the function is instructed to treat the current field as
    being an initial "high-redshift" field such that specific sources need not be found and evolved.

    .. note:: if a previous specific redshift is given, but no cached field is found at that redshift, the previous
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

    As the function recursively evaluates previous redshifts, the previous spin temperature fields will also be
    consistently recursively evaluated. Only the final ionized box will actually be returned and kept in memory, however
    intervening results will by default be cached on disk. One can also pass an explicit spin temperature obj:

    >>> ts = spin_temperature(redshift=7.0)
    >>> xHI = ionize_box(redshift=7.0, spin_temp=ts)

    If automatic recursion is used, then it is done in such a way that no large boxes are kept around in memory for
    longer than they need to be (only two at a time are required).
    """
    if spin_temp is not None:
        do_spin_temp = True

    # Set the upper limit on redshift at which we require a previous spin temp box.
    if z_heat_max is not None:
        global_params.Z_HEAT_MAX = z_heat_max

    if spin_temp is not None and not isinstance(spin_temp, TsBox):
        raise ValueError("spin_temp must be a TsBox instance")

    if isinstance(previous_ionize_box, IonizedBox):
        cosmo_params = previous_ionize_box.cosmo_params
        astro_params = previous_ionize_box.astro_params
        flag_options = previous_ionize_box.flag_options
        user_params = previous_ionize_box.user_params
    elif spin_temp is not None:
        cosmo_params = spin_temp.cosmo_params
        astro_params = spin_temp.astro_params
        flag_options = spin_temp.flag_options
        user_params = spin_temp.user_params
    elif perturbed_field is not None:
        cosmo_params = perturbed_field.cosmo_params
        user_params = perturbed_field.user_params

    if spin_temp is not None and isinstance(previous_ionize_box, IonizedBox):
        if (
            spin_temp.cosmo_params != previous_ionize_box.cosmo_params or
            spin_temp.user_params != previous_ionize_box.user_params or
            spin_temp.astro_params != previous_ionize_box.astro_params or
            spin_temp.flag_options != previous_ionize_box.flag_options
        ):
            raise ValueError("spin_temp and previous_ionize_box must have the same input parameters.")

    if perturbed_field is None and redshift is None and spin_temp is None:
        raise ValueError("Either perturbed_field, spin_temp, or redshift must be provided.")
    elif spin_temp is not None:
        redshift = spin_temp.redshift
    elif perturbed_field is not None:
        redshift = perturbed_field.redshift

    # Set the default astro params, using the INHOMO_RECO flag.
    if astro_params is None:
        astro_params = AstroParams(flag_options.INHOMO_RECO)

    if spin_temp is not None and spin_temp.redshift != redshift:
        raise ValueError("The redshift of the spin_temp field needs to be the current redshift")

    if perturbed_field is not None and perturbed_field.redshift != redshift:
        raise ValueError("The provided perturbed_field must have the same redshift as the provided spin_temp")

    box = IonizedBox(
        first_box=redshift > global_params.Z_HEAT_MAX and (not isinstance(previous_ionize_box, IonizedBox) or not previous_ionize_box.filled),
        user_params=user_params, cosmo_params=cosmo_params,
        redshift=redshift, astro_params=astro_params, flag_options=flag_options
    )

    # Check whether the boxes already exist
    if not regenerate:
        try:
            box.read(direc, fname, match_seed=match_seed)
            print("Existing ionized boxes found and read in.")
            return box
        except IOError:
            pass

    # EVERYTHING PAST THIS POINT ONLY HAPPENS IF THE BOX DOESN'T ALREADY EXIST
    # ------------------------------------------------------------------------
    # Get the previous redshift
    if flag_options.INHOMO_RECO or do_spin_temp:

        if previous_ionize_box is not None:
            print('1')
            if hasattr(previous_ionize_box, "redshift"):
                prev_z = previous_ionize_box.redshift
            elif isinstance(previous_ionize_box, numbers.Number):
                prev_z = previous_ionize_box
            else:
                raise ValueError("previous_ionize_box must be an IonizedBox or a float")
        elif z_step_factor is not None:

            prev_z = (1 + redshift) * z_step_factor - 1
            print(redshift, z_step_factor, prev_z)
        else:
            print('3')
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

    # Get appropriate previous ionization box
    if not isinstance(previous_ionize_box, IonizedBox):
        # If we are beyond Z_HEAT_MAX, just make an empty box
        if redshift > global_params.Z_HEAT_MAX or prev_z is None:
            previous_ionize_box = IonizedBox(redshift=0)

        # Otherwise recursively create new previous box.
        else:
            previous_ionize_box = ionize_box(
                astro_params=astro_params, flag_options=flag_options, redshift=prev_z,
                z_step_factor=z_step_factor, z_heat_max=z_heat_max,
                do_spin_temp=do_spin_temp,
                init_boxes=init_boxes, regenerate=regenerate, write=write, direc=direc,
                match_seed=True
            )

    # Dynamically produce the perturbed field.
    if perturbed_field is None or not perturbed_field.filled:
        perturbed_field = perturb_field(
            redshift=redshift, init_boxes=init_boxes, user_params=user_params, cosmo_params=cosmo_params,
            regenerate=regenerate, write=write, direc=direc,
            fname=None, match_seed=match_seed
        )

    # Set empty spin temp box if necessary.
    if not do_spin_temp:
        spin_temp = TsBox(redshift=0)
    elif spin_temp is None:
        # The following will raise an error (rightly) if the previous spin temperature does not exist.
        spin_temp = spin_temperature(
            redshift=redshift, perturbed_field=perturbed_field,  previous_spin_temp=prev_z,
            astro_params=astro_params, cosmo_params=cosmo_params, flag_options=flag_options, user_params=user_params
        )

    # Run the C Code
    lib.ComputeIonizedBox(redshift, previous_ionize_box.redshift, perturbed_field.user_params(),
                          perturbed_field.cosmo_params(), astro_params(), flag_options(), perturbed_field(),
                          previous_ionize_box(), do_spin_temp, spin_temp(), box())

    box.filled = True
    box._expose()

    # Optionally do stuff with the result (like writing it)
    if write:
        box.write(direc, fname)

    return box


def spin_temperature(astro_params=None, flag_options=FlagOptions(), redshift=None, perturbed_field=None,
                     previous_spin_temp=None, z_step_factor = 1.02, z_heat_max = None,
                     init_boxes=None, cosmo_params=CosmoParams(), user_params=UserParams(), regenerate=False, write=True, direc=None,
                     fname=None, match_seed=False):
    """
    Compute spin temperature boxes at a given redshift.

    See the notes below for how the spin temperature field is evolved through redshift.

    Parameters
    ----------
    astro_params: :class:`~AstroParams` instance, optional
        The astrophysical parameters defining the course of reionization.

    flag_options: :class:`~FlagOptions` instance, optional
        Some options passed to the reionization routine.

    redshift : float, optional
        The redshift at which to compute the ionized box. If not given, the redshift from `perturbed_field` will be used.
        Either `redshift`, `perturbed_field` or both must be given.

    perturbed_field : :class:`~PerturbField` instance, optional
        If given, this field will be used, otherwise it will be generated. To be generated, either `init_boxes` and
        `redshift` must be given, or `user_params`, `cosmo_params` and `redshift`. By default, this will be generated
        at the same redshift as the spin temperature box. However, it does not need to be defined at the same redshift.
        If at a different redshift, it will be linearly evolved to the redshift of the spin temperature box.

    previous_spin_temp : :class:`TsBox` or float, optional
        The previous spin temperature box, or its redshift. This redshift must be greater than `redshift`. If not given,
        will assume that this is the initial, high redshift box. If a redshift, then this will try to read in the
        previous spin temp box at this redshift, and if it doesn't exist, will raise an exception.

    z_step_factor: float, optional
        A factor greater than unity, which specifies the logarithmic steps in redshift with which the spin temperature
        box is evolved.

    z_heat_max: float, optional
        The maximum redshift at which to search for heating sources. Practically, this defines the limit in redshift
        at which the spin temperature can be defined purely from the background perturbed field rather than by evolving
        from a previous spin temperature field. Default is the global parameter `Z_HEAT_MAX`.

    init_boxes : :class:`~InitialConditions` instance, optional
        If given, and `perturbed_field` *not* given, these initial conditions boxes will be used to generate the
        perturbed field, otherwise initial conditions will be generated on the fly. If given,
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
        Whether to force the random seed to also match in order to be considered a match. This will always be true
        when looking for previous spin temperature boxes, but does not need to be true when identifying the primary
        density field.

    Returns
    -------
    :class:`~TsBox`
        An object containing the spin temperature box data.


    Notes
    -----

    Typically, the spin temperature field at any redshift is dependent on the evolution of spin temperature up until
    that redshift, which necessitates providing a previous spin temperature field to define the current one. This
    function provides several options for doing so. Either (in order of precedence) (i) a specific previous spin
    temperature object is provided, which will be used directly, (ii) a previous redshift is provided, for which a
    cached field on disk will be sought, (iii) a step factor is provided which recursively steps through redshifts,
    calculating previous fields up until Z_HEAT_MAX, and returning just the final field at the current redshift, or
    (iv) the function is instructed to treat the current field as being an initial "high-redshift" field such that
    specific sources need not be found and evolved.

    .. note:: if a previous specific redshift is given, but no cached field is found at that redshift, the previous
              spin temperature field will be evaluated based on `z_step_factor`.

    .. warning:: though it is convenient to have this function recursively determine the history of re-ionization, it
                 is by no means efficient. It requires to keep all intermediate boxes
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
    # Set the upper limit on redshift at which we require a previous spin temp box.
    if z_heat_max is not None:
        global_params.Z_HEAT_MAX = z_heat_max

    if isinstance(previous_spin_temp, TsBox):
        cosmo_params = previous_spin_temp.cosmo_params
        astro_params = previous_spin_temp.astro_params
        flag_options = previous_spin_temp.flag_options
        user_params = previous_spin_temp.user_params
    elif perturbed_field is not None:
        cosmo_params = perturbed_field.cosmo_params
        user_params = perturbed_field.user_params

    if perturbed_field is None and redshift is None:
        raise ValueError("Either perturbed_field or redshift must be provided.")
    elif redshift is None:
        redshift = perturbed_field.redshift

    # Set the default astro params, using the INHOMO_RECO flag.
    if astro_params is None:
        astro_params = AstroParams(flag_options.INHOMO_RECO)

    box = TsBox(
        first_box= redshift > global_params.Z_HEAT_MAX and (not isinstance(previous_spin_temp, IonizedBox) or not previous_spin_temp.filled),
        user_params=user_params, cosmo_params=cosmo_params,
        redshift=redshift, astro_params=astro_params, flag_options=flag_options
    )

    # Check whether the boxes already exist on disk.
    if not regenerate:
        try:
            box.read(direc, fname, match_seed=match_seed)
            print("Existing spin_temp boxes found and read in.")
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
        prev_z = (1+redshift)*z_step_factor - 1
    else:
        prev_z = None
        if redshift < global_params.Z_HEAT_MAX:
            warnings.warn("Attempting to evaluate spin temperature field at z=%s as if it was beyond Z_HEAT_MAX=%s"%(redshift, global_params.Z_HEAT_MAX))

    # Ensure the previous spin temperature has a higher redshift than this one.
    if prev_z and prev_z <= redshift:
        raise ValueError("Previous spin temperature box must have a higher redshift than that being evaluated.")

    # Dynamically produce the perturbed field.
    if perturbed_field is None or not perturbed_field.filled:
        perturbed_field = perturb_field(
            redshift=redshift,
            init_boxes=init_boxes, user_params=user_params, cosmo_params=cosmo_params,
            regenerate=regenerate, write=write, direc=direc,
            fname=None, match_seed=match_seed
        )

    # Create appropriate previous_spin_temp
    if not isinstance(previous_spin_temp, TsBox):
        if redshift > global_params.Z_HEAT_MAX or prev_z is None:
            previous_spin_temp = TsBox(redshift=0)
        else:
            previous_spin_temp = spin_temperature(
                astro_params=astro_params, flag_options=flag_options, redshift=prev_z, perturbed_field=perturbed_field,
                z_step_factor = z_step_factor, z_heat_max = z_heat_max,
                init_boxes=init_boxes, regenerate=regenerate, write=write, direc=direc,
                match_seed=True
            )

    # Run the C Code
    lib.ComputeTsBox(redshift, previous_spin_temp.redshift, perturbed_field.user_params(),
                     perturbed_field.cosmo_params(), astro_params(), perturbed_field.redshift, perturbed_field(),
                     previous_spin_temp(), box())
    box.filled = True
    box._expose()

    # Optionally do stuff with the result (like writing it)
    if write:
        box.write(direc, fname)

    return box


def brightness_temperature(ionized_box, perturb_field, spin_temp=None):

    if spin_temp is None:
        saturated_limit = True
#        spin_temp = ffi.new("struct TsBox*")
        spin_temp = TsBox(redshift=0)
        
    else:
        saturated_limit = False
        spin_temp = spin_temp()

#    if spin_temp.redshift != ionized_box.redshift != perturb_field.redshift:
#        raise ValueError("all box redshifts must be the same.")

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
        ionized_box.user_params(), ionized_box.cosmo_params(), ionized_box.astro_params(), ionized_box.flag_options(),
        spin_temp(), ionized_box(), perturb_field(), box())
    box.filled= True
    box._expose()

    return box


# The global parameter struct which can be modified.
global_params = lib.global_params


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
