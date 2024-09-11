"""
Input parameter classes.

There are four input parameter/option classes, not all of which are required for any
given function. They are :class:`UserParams`, :class:`CosmoParams`, :class:`AstroParams`
and :class:`FlagOptions`. Each of them defines a number of variables, and all of these
have default values, to minimize the burden on the user. These defaults are accessed via
the ``_defaults_`` class attribute of each class. The available parameters for each are
listed in the documentation for each class below.

Along with these, the module exposes ``global_params``, a singleton object of type
:class:`GlobalParams`, which is a simple class providing read/write access to a number of parameters
used throughout the computation which are very rarely varied.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from astropy import units as un
from astropy.cosmology import FLRW, Planck15
from os import path
from pathlib import Path

from ._cfg import config
from ._data import DATA_PATH
from ._utils import StructInstanceWrapper, StructWithDefaults
from .c_21cmfast import ffi, lib

logger = logging.getLogger("21cmFAST")

# Cosmology is from https://arxiv.org/pdf/1807.06209.pdf
# Table 2, last column. [TT,TE,EE+lowE+lensing+BAO]
Planck18 = Planck15.clone(
    Om0=(0.02242 + 0.11933) / 0.6766**2,
    Ob0=0.02242 / 0.6766**2,
    H0=67.66,
    name="Planck18",
)


class GlobalParams(StructInstanceWrapper):
    """
    Global parameters for 21cmFAST.

    This is a thin wrapper over an allocated C struct, containing parameter values
    which are used throughout various computations within 21cmFAST. It is a singleton;
    that is, a single python (and C) object exists, and no others should be created.
    This object is not "passed around", rather its values are accessed throughout the
    code.

    Parameters in this struct are considered to be options that should usually not have
    to be modified, and if so, typically once in any given script or session.

    Values can be set in the normal way, eg.:

    >>> global_params.ALPHA_UVB = 5.5

    The class also provides a context manager for setting parameters for a well-defined
    portion of the code. For example, if you would like to set ``Z_HEAT_MAX`` for a given
    run:

    >>> with global_params.use(Z_HEAT_MAX=25):
    >>>     p21c.run_lightcone(...)  # uses Z_HEAT_MAX=25 for the entire run.
    >>> print(global_params.Z_HEAT_MAX)
    35.0

    Attributes
    ----------
    ALPHA_UVB : float
        Power law index of the UVB during the EoR.  This is only used if `INHOMO_RECO` is
        True (in :class:`FlagOptions`), in order to compute the local mean free path
        inside the cosmic HII regions.
    EVOLVE_DENSITY_LINEARLY : bool
        Whether to evolve the density field with linear theory (instead of 1LPT or Zel'Dovich).
        If choosing this option, make sure that your cell size is
        in the linear regime at the redshift of interest. Otherwise, make sure you resolve
        small enough scales, roughly we find BOX_LEN/DIM should be < 1Mpc
    SMOOTH_EVOLVED_DENSITY_FIELD : bool
        If True, the zeldovich-approximation density field is additionally smoothed
        (aside from the implicit boxcar smoothing performed when re-binning the ICs from
        DIM to HII_DIM) with a Gaussian filter of width ``R_smooth_density*BOX_LEN/HII_DIM``.
        The implicit boxcar smoothing in ``perturb_field()`` bins the density field on
        scale DIM/HII_DIM, similar to what Lagrangian codes do when constructing Eulerian
        grids. In other words, the density field is quantized into ``(DIM/HII_DIM)^3`` values.
        If your usage requires smooth density fields, it is recommended to set this to True.
        This also decreases the shot noise present in all grid based codes, though it
        overcompensates by an effective loss in resolution. **Added in 1.1.0**.
    R_smooth_density : float
        Determines the smoothing length to use if `SMOOTH_EVOLVED_DENSITY_FIELD` is True.
    HII_ROUND_ERR : float
        Rounding error on the ionization fraction. If the mean xHI is greater than
        ``1 - HII_ROUND_ERR``, then finding HII bubbles is skipped, and a homogeneous
        xHI field of ones is returned. Added in  v1.1.0.
    FIND_BUBBLE_ALGORITHM : int, {1,2}
        Choose which algorithm used to find HII bubbles. Options are: (1) Mesinger & Furlanetto 2007
        method of overlapping spheres: paint an ionized sphere with radius R, centered on pixel
        where R is filter radius. This method, while somewhat more accurate, is slower than (2),
        especially in mostly ionized universes, so only use for lower resolution boxes
        (HII_DIM<~400). (2) Center pixel only method (Zahn et al. 2007). This is faster.
    N_POISSON : int
        If not using the halo field to generate HII regions, we provide the option of
        including Poisson scatter in the number of sources obtained through the conditional
        collapse fraction (which only gives the *mean* collapse fraction on a particular
        scale. If the predicted mean collapse fraction is less than  `N_POISSON * M_MIN`,
        then Poisson scatter is added to mimic discrete halos on the subgrid scale (see
        Zahn+2010).Use a negative number to turn it off.

        .. note:: If you are interested in snapshots of the same realization at several
                  redshifts,it is recommended to turn off this feature, as halos can
                  stochastically "pop in and out of" existence from one redshift to the next.
    R_OVERLAP_FACTOR : float
        When using USE_HALO_FIELD, it is used as a factor the halo's radius, R, so that the
        effective radius is R_eff = R_OVERLAP_FACTOR * R.  Halos whose centers are less than
        R_eff away from another halo are not allowed. R_OVERLAP_FACTOR = 1 is fully disjoint
        R_OVERLAP_FACTOR = 0 means that centers are allowed to lay on the edges of
        neighboring halos.
    DELTA_CRIT_MODE : int
        The delta_crit to be used for determining whether a halo exists in a cell
            0: delta_crit is constant (i.e. 1.686)
            1: delta_crit is the sheth tormen ellipsoidal collapse correction to delta_crit
    HALO_FILTER : int
        Filter for the density field used to generate the halo field with EPS
            0: real space top hat filter
            1: sharp k-space filter
            2: gaussian filter
    OPTIMIZE : bool
        Finding halos can be made more efficient if the filter size is sufficiently large that
        we can switch to the collapse fraction at a later stage.
    OPTIMIZE_MIN_MASS : float
        Minimum mass on which the optimization for the halo finder will be used.
    T_USE_VELOCITIES : bool
        Whether to use velocity corrections in 21-cm fields

        .. note:: The approximation used to include peculiar velocity effects works
                  only in the linear regime, so be careful using this (see Mesinger+2010)

    MAX_DVDR : float
        Maximum velocity gradient along the line of sight in units of the hubble parameter at z.
        This is only used in computing the 21cm fields.

        .. note:: Setting this too high can add spurious 21cm power in the early stages,
                  due to the 1-e^-tau ~ tau approximation (see Mesinger's 21cm intro paper and mao+2011).
                  However, this is still a good approximation at the <~10% level.

    VELOCITY_COMPONENT : int
        Component of the velocity to be used in 21-cm temperature maps (1=x, 2=y, 3=z)
    DELTA_R_FACTOR : float
        Factor by which to scroll through filter radius for halos
    DELTA_R_HII_FACTOR : float
        Factor by which to scroll through filter radius for bubbles
    HII_FILTER : int, {0, 1, 2}
        Filter for the Halo or density field used to generate ionization field:
        0. real space top hat filter
        1. k-space top hat filter
        2. gaussian filter
    INITIAL_REDSHIFT : float
        Used to perturb field
    CRIT_DENS_TRANSITION : float
        A transition value for the interpolation tables for calculating the number of ionising
        photons produced given the input parameters. Log sampling is desired, however the numerical
        accuracy near the critical density for collapse (i.e. 1.69) broke down. Therefore, below the
        value for `CRIT_DENS_TRANSITION` log sampling of the density values is used, whereas above
        this value linear sampling is used.
    MIN_DENSITY_LOW_LIMIT : float
        Required for using the interpolation tables for the number of ionising photons. This is a
        lower limit for the density values that is slightly larger than -1. Defined as a density
        contrast.
    RecombPhotonCons : int
        Whether or not to use the recombination term when calculating the filling factor for
        performing the photon non-conservation correction.
    PhotonConsStart : float
        A starting value for the neutral fraction where the photon non-conservation correction is
        performed exactly. Any value larger than this the photon non-conservation correction is not
        performed (i.e. the algorithm is perfectly photon conserving).
    PhotonConsEnd : float
        An end-point for where the photon non-conservation correction is performed exactly. This is
        required to remove undesired numerical artifacts in the resultant neutral fraction histories.
    PhotonConsAsymptoteTo : float
        Beyond `PhotonConsEnd` the photon non-conservation correction is extrapolated to yield
        smooth reionisation histories. This sets the lowest neutral fraction value that the photon
        non-conservation correction will be applied to.
    HEAT_FILTER : int
        Filter used for smoothing the linear density field to obtain the collapsed fraction:
            0: real space top hat filter
            1: sharp k-space filter
            2: gaussian filter
    CLUMPING_FACTOR : float
        Sub grid scale. If you want to run-down from a very high redshift (>50), you should
        set this to one.
    Z_HEAT_MAX : float
        Maximum redshift used in the Tk and x_e evolution equations.
        Temperature and x_e are assumed to be homogeneous at higher redshifts.
        Lower values will increase performance.
    R_XLy_MAX : float
        Maximum radius of influence for computing X-ray and Lya pumping in cMpc. This
        should be larger than the mean free path of the relevant photons.
    NUM_FILTER_STEPS_FOR_Ts : int
        Number of spherical annuli used to compute df_coll/dz' in the simulation box.
        The spherical annuli are evenly spaced in logR, ranging from the cell size to the box
        size. :func:`~wrapper.spin_temp` will create this many boxes of size `HII_DIM`,
        so be wary of memory usage if values are high.
    ZPRIME_STEP_FACTOR : float
        Logarithmic redshift step-size used in the z' integral.  Logarithmic dz.
        Decreasing (closer to unity) increases total simulation time for lightcones,
        and for Ts calculations.
    TK_at_Z_HEAT_MAX : float
        If positive, then overwrite default boundary conditions for the evolution
        equations with this value. The default is to use the value obtained from RECFAST.
        See also `XION_at_Z_HEAT_MAX`.
    XION_at_Z_HEAT_MAX : float
        If positive, then overwrite default boundary conditions for the evolution
        equations with this value. The default is to use the value obtained from RECFAST.
        See also `TK_at_Z_HEAT_MAX`.
    Pop : int
        Stellar Population responsible for early heating (2 or 3)
    Pop2_ion : float
        Number of ionizing photons per baryon for population 2 stellar species.
    Pop3_ion : float
        Number of ionizing photons per baryon for population 3 stellar species.
    NU_X_BAND_MAX : float
        This is the upper limit of the soft X-ray band (0.5 - 2 keV) used for normalising
        the X-ray SED to observational limits set by the X-ray luminosity. Used for performing
        the heating rate integrals.
    NU_X_MAX : float
        An upper limit (must be set beyond `NU_X_BAND_MAX`) for performing the rate integrals.
        Given the X-ray SED is modelled as a power-law, this removes the potential of divergent
        behaviour for the heating rates. Chosen purely for numerical convenience though it is
        motivated by the fact that observed X-ray SEDs apprear to turn-over around 10-100 keV
        (Lehmer et al. 2013, 2015)
    NBINS_LF : int
        Number of bins for the luminosity function calculation.
    P_CUTOFF : bool
        Turn on Warm-Dark-matter power suppression.
    M_WDM : float
        Mass of WDM particle in keV. Ignored if `P_CUTOFF` is False.
    g_x : float
        Degrees of freedom of WDM particles; 1.5 for fermions.
    OMn : float
        Relative density of neutrinos in the universe.
    OMk : float
        Relative density of curvature.
    OMr : float
        Relative density of radiation.
    OMtot : float
        Fractional density of the universe with respect to critical density. Set to
        unity for a flat universe.
    Y_He : float
        Helium fraction.
    wl : float
        Dark energy equation of state parameter (wl = -1 for vacuum )
    SHETH_b : float
        Sheth-Tormen parameter for ellipsoidal collapse (for HMF).

        .. note:: The best fit b and c ST params for these 3D realisations have a redshift,
                  and a ``DELTA_R_FACTOR`` dependence, as shown
                  in Mesinger+. For converged mass functions at z~5-10, set `DELTA_R_FACTOR=1.1`
                  and `SHETH_b=0.15` and `SHETH_c~0.05`.

                  For most purposes, a larger step size is quite sufficient and provides an
                  excellent match to N-body and smoother mass functions, though the b and c
                  parameters should be changed to make up for some "stepping-over" massive
                  collapsed halos (see Mesinger, Perna, Haiman (2005) and Mesinger et al.,
                  in preparation).

                  For example, at z~7-10, one can set `DELTA_R_FACTOR=1.3` and `SHETH_b=0.15`
                   and `SHETH_c=0.25`, to increase the speed of the halo finder.
    SHETH_c : float
        Sheth-Tormen parameter for ellipsoidal collapse (for HMF). See notes for `SHETH_b`.
    Zreion_HeII : float
        Redshift of helium reionization, currently only used for tau_e
    FILTER : int, {0, 1}
        Filter to use for smoothing.
        0. tophat
        1. gaussian
    external_table_path : str
        The system path to find external tables for calculation speedups. DO NOT MODIFY.
    R_BUBBLE_MIN : float
        Minimum radius of bubbles to be searched in cMpc. One can set this to 0, but should
        be careful with shot noise if running on a fine, non-linear density grid. Default
        is set to L_FACTOR which is (4PI/3)^(-1/3) = 0.620350491.
    M_MIN_INTEGRAL:
        Minimum mass when performing integral on halo mass function.
    M_MAX_INTEGRAL:
        Maximum mass when performing integral on halo mass function.
    T_RE:
        The peak gas temperatures behind the supersonic ionization fronts during reionization.
    VAVG:
        Avg value of the DM-b relative velocity [im km/s], ~0.9*SIGMAVCB (=25.86 km/s) normally.
    """

    def __init__(self, wrapped, ffi):
        super().__init__(wrapped, ffi)

        self.external_table_path = ffi.new("char[]", str(DATA_PATH).encode())
        self._wisdoms_path = Path(config["direc"]) / "wisdoms"
        self.wisdoms_path = ffi.new("char[]", str(self._wisdoms_path).encode())

    @property
    def external_table_path(self):
        """An ffi char pointer to the path to which external tables are kept."""
        return self._external_table_path

    @external_table_path.setter
    def external_table_path(self, val):
        self._external_table_path = val

    @property
    def wisdoms_path(self):
        """An ffi char pointer to the path to which external tables are kept."""
        if not self._wisdoms_path.exists():
            self._wisdoms_path.mkdir(parents=True)

        return self._wisdom_path

    @wisdoms_path.setter
    def wisdoms_path(self, val):
        self._wisdom_path = val

    @contextlib.contextmanager
    def use(self, **kwargs):
        """Set given parameters for a certain context.

        .. note:: Keywords are *not* case-sensitive.

        Examples
        --------
        >>> from py21cmfast import global_params, run_lightcone
        >>> with global_params.use(zprime_step_factor=1.1, Sheth_c=0.06):
        >>>     run_lightcone(redshift=7)
        """
        prev = {}
        this_attr_upper = {k.upper(): k for k in self.keys()}

        for k, val in kwargs.items():
            if k.upper() not in this_attr_upper:
                warnings.warn(
                    f"{k} is not a valid parameter of global_params, and will be ignored",
                    UserWarning,
                )
                continue
            key = this_attr_upper[k.upper()]
            prev[key] = getattr(self, key)
            setattr(self, key, val)

        yield

        # Restore everything back to the way it was.
        for k, v in prev.items():
            setattr(self, k, v)


global_params = GlobalParams(lib.global_params, ffi)


class CosmoParams(StructWithDefaults):
    """
    Cosmological parameters (with defaults) which translates to a C struct.

    To see default values for each parameter, use ``CosmoParams._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

    Default parameters are based on Plank18, https://arxiv.org/pdf/1807.06209.pdf,
    Table 2, last column. [TT,TE,EE+lowE+lensing+BAO]

    Parameters
    ----------
    SIGMA_8 : float, optional
        RMS mass variance (power spectrum normalisation).
    hlittle : float, optional
        The hubble parameter, H_0/100.
    OMm : float, optional
        Omega matter.
    OMb : float, optional
        Omega baryon, the baryon component.
    POWER_INDEX : float, optional
        Spectral index of the power spectrum.
    """

    def __init__(self, *args, _base_cosmo=Planck18, **kwargs):
        self._base_cosmo = _base_cosmo
        super().__init__(*args, **kwargs)

    _ffi = ffi

    _defaults_ = {
        "SIGMA_8": 0.8102,
        "hlittle": Planck18.h,
        "OMm": Planck18.Om0,
        "OMb": Planck18.Ob0,
        "POWER_INDEX": 0.9665,
    }

    @property
    def OMl(self):
        """Omega lambda, dark energy density."""
        return 1 - self.OMm

    @property
    def cosmo(self):
        """Return an astropy cosmology object for this cosmology."""
        return self._base_cosmo.clone(
            name=self._base_cosmo.name,
            H0=self.hlittle * 100,
            Om0=self.OMm,
            Ob0=self.OMb,
        )

    @classmethod
    def from_astropy(cls, cosmo: FLRW, **kwargs):
        """Create a CosmoParams object from an astropy cosmology object.

        Pass SIGMA_8 and POWER_INDEX as kwargs if you want to override the default
        values.
        """
        return cls(
            hlittle=cosmo.h, OMm=cosmo.Om0, OMb=cosmo.Ob0, _base_cosmo=cosmo, **kwargs
        )


class UserParams(StructWithDefaults):
    """
    Structure containing user parameters (with defaults).

    To see default values for each parameter, use ``UserParams._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

    Parameters
    ----------
    HII_DIM : int, optional
        Number of cells for the low-res box. Default 200.
    DIM : int,optional
        Number of cells for the high-res box (sampling ICs) along a principal axis. To avoid
        sampling issues, DIM should be at least 3 or 4 times HII_DIM, and an integer multiple.
        By default, it is set to 3*HII_DIM.
    NON_CUBIC_FACTOR : float, optional
        Factor which allows the creation of non-cubic boxes. It will shorten/lengthen the line
        of sight dimension of all boxes. NON_CUBIC_FACTOR * DIM/HII_DIM must result in an integer
    BOX_LEN : float, optional
        Length of the box, in Mpc. Default 300 Mpc.
    HMF: int or str, optional
        Determines which halo mass function to be used for the normalisation of the
        collapsed fraction (default Sheth-Tormen). If string should be one of the
        following codes:
        0: PS (Press-Schechter)
        1: ST (Sheth-Tormen)
        2: Watson (Watson FOF)
        3: Watson-z (Watson FOF-z)
    USE_RELATIVE_VELOCITIES: int, optional
        Flag to decide whether to use relative velocities.
        If True, POWER_SPECTRUM is automatically set to 5. Default False.
    POWER_SPECTRUM: int or str, optional
        Determines which power spectrum to use, default EH (unless `USE_RELATIVE_VELOCITIES`
        is True). If string, use the following codes:
        0: EH
        1: BBKS
        2: EFSTATHIOU
        3: PEEBLES
        4: WHITE
        5: CLASS (single cosmology)
    N_THREADS : int, optional
        Sets the number of processors (threads) to be used for performing 21cmFAST.
        Default 1.
    PERTURB_ON_HIGH_RES : bool, optional
        Whether to perform the Zel'Dovich or 2LPT perturbation on the low or high
        resolution grid.
    NO_RNG : bool, optional
        Ability to turn off random number generation for initial conditions. Can be
        useful for debugging and adding in new features
    USE_FFTW_WISDOM : bool, optional
        Whether or not to use stored FFTW_WISDOMs for improving performance of FFTs
    USE_INTERPOLATION_TABLES: bool, optional
        If True, calculates and evaluates quantites using interpolation tables, which
        is considerably faster than when performing integrals explicitly.
    FAST_FCOLL_TABLES: bool, optional
        Whether to use fast Fcoll tables, as described in Appendix of Muñoz+21 (2110.13919). Significant speedup for minihaloes.
    USE_2LPT: bool, optional
        Whether to use second-order Lagrangian perturbation theory (2LPT).
        Set this to True if the density field or the halo positions are extrapolated to
        low redshifts. The current implementation is very naive and adds a factor ~6 to
        the memory requirements. Reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118
        Appendix D.
    MINIMIZE_MEMORY: bool, optional
        If set, the code will run in a mode that minimizes memory usage, at the expense
        of some CPU/disk-IO. Good for large boxes / small computers.
    """

    _ffi = ffi

    _defaults_ = {
        "BOX_LEN": 300.0,
        "DIM": None,
        "HII_DIM": 200,
        "NON_CUBIC_FACTOR": 1.0,
        "USE_FFTW_WISDOM": False,
        "HMF": 1,
        "USE_RELATIVE_VELOCITIES": False,
        "POWER_SPECTRUM": 0,
        "N_THREADS": 1,
        "PERTURB_ON_HIGH_RES": False,
        "NO_RNG": False,
        "USE_INTERPOLATION_TABLES": None,
        "FAST_FCOLL_TABLES": False,
        "USE_2LPT": True,
        "MINIMIZE_MEMORY": False,
        "KEEP_3D_VELOCITIES": False,
    }

    _hmf_models = ["PS", "ST", "WATSON", "WATSON-Z"]
    _power_models = ["EH", "BBKS", "EFSTATHIOU", "PEEBLES", "WHITE", "CLASS"]

    @property
    def USE_INTERPOLATION_TABLES(self):
        """Whether to use interpolation tables for integrals, speeding things up."""
        if self._USE_INTERPOLATION_TABLES is None:
            warnings.warn(
                "The USE_INTERPOLATION_TABLES setting has changed in v3.1.2 to be "
                "default True. You can likely ignore this warning, but if you relied on"
                "having USE_INTERPOLATION_TABLES=False by *default*, please set it "
                "explicitly. To silence this warning, set it explicitly to True. This"
                "warning will be removed in v4."
            )
            self._USE_INTERPOLATION_TABLES = True

        return self._USE_INTERPOLATION_TABLES

    @property
    def DIM(self):
        """Number of cells for the high-res box (sampling ICs) along a principal axis."""
        return self._DIM or 3 * self.HII_DIM

    @property
    def NON_CUBIC_FACTOR(self):
        """Factor to shorten/lengthen the line-of-sight dimension (non-cubic boxes)."""
        dcf = self.DIM * self._NON_CUBIC_FACTOR
        hdcf = self.HII_DIM * self._NON_CUBIC_FACTOR
        if dcf % int(dcf) or hdcf % int(hdcf):
            raise ValueError(
                "NON_CUBIC_FACTOR * DIM and NON_CUBIC_FACTOR * HII_DIM must be integers"
            )
        else:
            return self._NON_CUBIC_FACTOR

    @property
    def tot_fft_num_pixels(self):
        """Total number of pixels in the high-res box."""
        return self.NON_CUBIC_FACTOR * self.DIM**3

    @property
    def HII_tot_num_pixels(self):
        """Total number of pixels in the low-res box."""
        return self.NON_CUBIC_FACTOR * self.HII_DIM**3

    @property
    def POWER_SPECTRUM(self):
        """
        The power spectrum generator to use, as an integer.

        See :func:`power_spectrum_model` for a string representation.
        """
        if self.USE_RELATIVE_VELOCITIES:
            if self._POWER_SPECTRUM != 5 or (
                isinstance(self._POWER_SPECTRUM, str)
                and self._POWER_SPECTRUM.upper() != "CLASS"
            ):
                logger.warn(
                    "Automatically setting POWER_SPECTRUM to 5 (CLASS) as you are using "
                    "relative velocities"
                )
                self._POWER_SPECTRUM = 5
            return 5
        else:
            if isinstance(self._POWER_SPECTRUM, str):
                val = self._power_models.index(self._POWER_SPECTRUM.upper())
            else:
                val = self._POWER_SPECTRUM

            if not 0 <= val < len(self._power_models):
                raise ValueError(
                    f"Power spectrum must be between 0 and {len(self._power_models) - 1}"
                )

            return val

    @property
    def HMF(self):
        """The HMF to use (an int, mapping to a given form).

        See hmf_model for a string representation.
        """
        if isinstance(self._HMF, str):
            val = self._hmf_models.index(self._HMF.upper())
        else:
            val = self._HMF

        try:
            val = int(val)
        except (ValueError, TypeError) as e:
            raise ValueError("Invalid value for HMF") from e

        if not 0 <= val < len(self._hmf_models):
            raise ValueError(
                f"HMF must be an int between 0 and {len(self._hmf_models) - 1}"
            )

        return val

    @property
    def hmf_model(self):
        """String representation of the HMF model used."""
        return self._hmf_models[self.HMF]

    @property
    def power_spectrum_model(self):
        """String representation of the power spectrum model used."""
        return self._power_models[self.POWER_SPECTRUM]

    @property
    def FAST_FCOLL_TABLES(self):
        """Check that USE_INTERPOLATION_TABLES is True."""
        if not self._FAST_FCOLL_TABLES or self.USE_INTERPOLATION_TABLES:
            return self._FAST_FCOLL_TABLES
        logger.warn(
            "You cannot turn on FAST_FCOLL_TABLES without USE_INTERPOLATION_TABLES."
        )
        return False

    @property
    def cell_size(self) -> un.Quantity[un.Mpc]:
        """The resolution of a low-res cell."""
        return (self.BOX_LEN / self.HII_DIM) * un.Mpc

    @property
    def cell_size_hires(self) -> un.Quantity[un.Mpc]:
        """The resolution of a hi-res cell."""
        return (self.BOX_LEN / self.DIM) * un.Mpc


class FlagOptions(StructWithDefaults):
    """
    Flag-style options for the ionization routines.

    To see default values for each parameter, use ``FlagOptions._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes
    which should be considered read-only. This is true of all input-parameter classes.

    Note that all flags are set to False by default, giving the simplest "vanilla"
    version of 21cmFAST.

    Parameters
    ----------
    USE_HALO_FIELD : bool, optional
        Set to True if intending to find and use the halo field. If False, uses
        the mean collapse fraction (which is considerably faster).
    USE_MINI_HALOS : bool, optional
        Set to True if using mini-halos parameterization.
        If True, USE_MASS_DEPENDENT_ZETA and INHOMO_RECO must be True.
    USE_CMB_HEATING : bool, optional
        Whether to include CMB Heating. (cf Eq.4 of Meiksin 2021, arxiv.org/abs/2105.14516)
    USE_LYA_HEATING : bool, optional
        Whether to use Lyman-alpha heating. (cf Sec. 3 of Reis+2021, doi.org/10.1093/mnras/stab2089)
    USE_MASS_DEPENDENT_ZETA : bool, optional
        Set to True if using new parameterization. Setting to True will automatically
        set `M_MIN_in_Mass` to True.
    SUBCELL_RSDS : bool, optional
        Add sub-cell redshift-space-distortions (cf Sec 2.2 of Greig+2018).
        Will only be effective if `USE_TS_FLUCT` is True.
    INHOMO_RECO : bool, optional
        Whether to perform inhomogeneous recombinations. Increases the computation
        time.
    USE_TS_FLUCT : bool, optional
        Whether to perform IGM spin temperature fluctuations (i.e. X-ray heating).
        Dramatically increases the computation time.
    M_MIN_in_Mass : bool, optional
        Whether the minimum halo mass (for ionization) is defined by
        mass or virial temperature. Automatically True if `USE_MASS_DEPENDENT_ZETA`
        is True.
    PHOTON_CONS : bool, optional
        Whether to perform a small correction to account for the inherent
        photon non-conservation.
    FIX_VCB_AVG: bool, optional
        Determines whether to use a fixed vcb=VAVG (*regardless* of USE_RELATIVE_VELOCITIES). It includes the average effect of velocities but not its fluctuations. See Muñoz+21 (2110.13919).
    USE_VELS_AUX: bool, optional
        Auxiliary variable (not input) to check if minihaloes are being used without relative velocities and complain
    """

    _ffi = ffi

    _defaults_ = {
        "USE_HALO_FIELD": False,
        "USE_MINI_HALOS": False,
        "USE_CMB_HEATING": True,
        "USE_LYA_HEATING": True,
        "USE_MASS_DEPENDENT_ZETA": False,
        "SUBCELL_RSD": False,
        "APPLY_RSDS": True,
        "INHOMO_RECO": False,
        "USE_TS_FLUCT": False,
        "M_MIN_in_Mass": False,
        "PHOTON_CONS": False,
        "FIX_VCB_AVG": False,
    }

    @property
    def SUBCELL_RSD(self):
        """The SUBCELL_RSD flag is only effective if APPLY_RSDS is True."""
        return self._SUBCELL_RSD and self.APPLY_RSDS

    @property
    def USE_HALO_FIELD(self):
        """Automatically setting USE_HALO_FIELD to False if USE_MINI_HALOS."""
        if self._USE_HALO_FIELD and self.USE_MINI_HALOS:
            logger.warn(
                "You have set USE_MINI_HALOS to True but USE_HALO_FIELD is also True! "
                "Automatically setting USE_HALO_FIELD to False."
            )
            self._USE_HALO_FIELD = False

        return self._USE_HALO_FIELD

    @property
    def M_MIN_in_Mass(self):
        """Whether minimum halo mass is defined in mass or virial temperature."""
        return True if self.USE_MASS_DEPENDENT_ZETA else self._M_MIN_in_Mass

    @property
    def USE_MASS_DEPENDENT_ZETA(self):
        """Automatically setting USE_MASS_DEPENDENT_ZETA to True if USE_MINI_HALOS."""
        if self.USE_MINI_HALOS and not self._USE_MASS_DEPENDENT_ZETA:
            logger.warn(
                "You have set USE_MINI_HALOS to True but USE_MASS_DEPENDENT_ZETA is False! "
                "Automatically setting USE_MASS_DEPENDENT_ZETA to True."
            )
            self._USE_MASS_DEPENDENT_ZETA = True
        return self._USE_MASS_DEPENDENT_ZETA

    @property
    def INHOMO_RECO(self):
        """Automatically setting INHOMO_RECO to True if USE_MINI_HALOS."""
        if self.USE_MINI_HALOS and not self._INHOMO_RECO:
            warnings.warn(
                "You have set USE_MINI_HALOS to True but INHOMO_RECO to False! "
                "Automatically setting INHOMO_RECO to True."
            )
            self._INHOMO_RECO = True
        return self._INHOMO_RECO

    @property
    def USE_TS_FLUCT(self):
        """Automatically setting USE_TS_FLUCT to True if USE_MINI_HALOS."""
        if self.USE_MINI_HALOS and not self._USE_TS_FLUCT:
            logger.warn(
                "You have set USE_MINI_HALOS to True but USE_TS_FLUCT to False! "
                "Automatically setting USE_TS_FLUCT to True."
            )
            self._USE_TS_FLUCT = True
        return self._USE_TS_FLUCT

    @property
    def PHOTON_CONS(self):
        """Automatically setting PHOTON_CONS to False if USE_MINI_HALOS."""
        if self.USE_MINI_HALOS and self._PHOTON_CONS:
            logger.warn(
                "USE_MINI_HALOS is not compatible with PHOTON_CONS! "
                "Automatically setting PHOTON_CONS to False."
            )
            self._PHOTON_CONS = False
        return self._PHOTON_CONS


class AstroParams(StructWithDefaults):
    """
    Astrophysical parameters.

    To see default values for each parameter, use ``AstroParams._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

    Parameters
    ----------
    INHOMO_RECO : bool, optional
        Whether inhomogeneous recombinations are being calculated. This is not a part of the
        astro parameters structure, but is required by this class to set some default behaviour.
    HII_EFF_FACTOR : float, optional
        The ionizing efficiency of high-z galaxies (zeta, from Eq. 2 of Greig+2015).
        Higher values tend to speed up reionization.
    F_STAR10 : float, optional
        The fraction of galactic gas in stars for 10^10 solar mass haloes.
        Only used in the "new" parameterization,
        i.e. when `USE_MASS_DEPENDENT_ZETA` is set to True (in :class:`FlagOptions`).
        If so, this is used along with `F_ESC10` to determine `HII_EFF_FACTOR` (which
        is then unused). See Eq. 11 of Greig+2018 and Sec 2.1 of Park+2018.
        Given in log10 units.
    F_STAR7_MINI : float, optional
        The fraction of galactic gas in stars for 10^7 solar mass minihaloes.
        Only used in the "minihalo" parameterization,
        i.e. when `USE_MINI_HALOS` is set to True (in :class:`FlagOptions`).
        If so, this is used along with `F_ESC7_MINI` to determine `HII_EFF_FACTOR_MINI` (which
        is then unused). See Eq. 8 of Qin+2020.
        Given in log10 units.
    ALPHA_STAR : float, optional
        Power-law index of fraction of galactic gas in stars as a function of halo mass.
        See Sec 2.1 of Park+2018.
    ALPHA_STAR_MINI : float, optional
        Power-law index of fraction of galactic gas in stars as a function of halo mass, for MCGs.
        See Sec 2 of Muñoz+21 (2110.13919).
    F_ESC10 : float, optional
        The "escape fraction", i.e. the fraction of ionizing photons escaping into the
        IGM, for 10^10 solar mass haloes. Only used in the "new" parameterization,
        i.e. when `USE_MASS_DEPENDENT_ZETA` is set to True (in :class:`FlagOptions`).
        If so, this is used along with `F_STAR10` to determine `HII_EFF_FACTOR` (which
        is then unused). See Eq. 11 of Greig+2018 and Sec 2.1 of Park+2018.
    F_ESC7_MINI: float, optional
        The "escape fraction for minihalos", i.e. the fraction of ionizing photons escaping
        into the IGM, for 10^7 solar mass minihaloes. Only used in the "minihalo" parameterization,
        i.e. when `USE_MINI_HALOS` is set to True (in :class:`FlagOptions`).
        If so, this is used along with `F_ESC7_MINI` to determine `HII_EFF_FACTOR_MINI` (which
        is then unused). See Eq. 17 of Qin+2020.
        Given in log10 units.
    ALPHA_ESC : float, optional
        Power-law index of escape fraction as a function of halo mass. See Sec 2.1 of
        Park+2018.
    M_TURN : float, optional
        Turnover mass (in log10 solar mass units) for quenching of star formation in
        halos, due to SNe or photo-heating feedback, or inefficient gas accretion. Only
        used if `USE_MASS_DEPENDENT_ZETA` is set to True in :class:`FlagOptions`.
        See Sec 2.1 of Park+2018.
    R_BUBBLE_MAX : float, optional
        Mean free path in Mpc of ionizing photons within ionizing regions (Sec. 2.1.2 of
        Greig+2015). Default is 50 if `INHOMO_RECO` is True, or 15.0 if not.
    ION_Tvir_MIN : float, optional
        Minimum virial temperature of star-forming haloes (Sec 2.1.3 of Greig+2015).
        Given in log10 units.
    L_X : float, optional
        The specific X-ray luminosity per unit star formation escaping host galaxies.
        Cf. Eq. 6 of Greig+2018. Given in log10 units.
    L_X_MINI: float, optional
        The specific X-ray luminosity per unit star formation escaping host galaxies for
        minihalos. Cf. Eq. 23 of Qin+2020. Given in log10 units.
    NU_X_THRESH : float, optional
        X-ray energy threshold for self-absorption by host galaxies (in eV). Also called
        E_0 (cf. Sec 4.1 of Greig+2018). Typical range is (100, 1500).
    X_RAY_SPEC_INDEX : float, optional
        X-ray spectral energy index (cf. Sec 4.1 of Greig+2018). Typical range is
        (-1, 3).
    X_RAY_Tvir_MIN : float, optional
        Minimum halo virial temperature in which X-rays are produced. Given in log10
        units. Default is `ION_Tvir_MIN`.
    F_H2_SHIELD: float, optional
        Self-shielding factor of molecular hydrogen when experiencing LW suppression.
        Cf. Eq. 12 of Qin+2020. Consistently included in A_LW fit from sims.
        If used we recommend going back to Macachek+01 A_LW=22.86.
    t_STAR : float, optional
        Fractional characteristic time-scale (fraction of hubble time) defining the
        star-formation rate of galaxies. Only used if `USE_MASS_DEPENDENT_ZETA` is set
        to True in :class:`FlagOptions`. See Sec 2.1, Eq. 3 of Park+2018.
    N_RSD_STEPS : int, optional
        Number of steps used in redshift-space-distortion algorithm. NOT A PHYSICAL
        PARAMETER.
    A_LW, BETA_LW: float, optional
        Impact of the LW feedback on Mturn for minihaloes. Default is 22.8685 and 0.47 following Machacek+01, respectively. Latest simulations suggest 2.0 and 0.6. See Sec 2 of Muñoz+21 (2110.13919).
    A_VCB, BETA_VCB: float, optional
        Impact of the DM-baryon relative velocities on Mturn for minihaloes. Default is 1.0 and 1.8, and agrees between different sims. See Sec 2 of Muñoz+21 (2110.13919).
    """

    _ffi = ffi

    _defaults_ = {
        "HII_EFF_FACTOR": 30.0,
        "F_STAR10": -1.3,
        "F_STAR7_MINI": -2.0,
        "ALPHA_STAR": 0.5,
        "ALPHA_STAR_MINI": 0.5,
        "F_ESC10": -1.0,
        "F_ESC7_MINI": -2.0,
        "ALPHA_ESC": -0.5,
        "M_TURN": 8.7,
        "R_BUBBLE_MAX": None,
        "ION_Tvir_MIN": 4.69897,
        "L_X": 40.0,
        "L_X_MINI": 40.0,
        "NU_X_THRESH": 500.0,
        "X_RAY_SPEC_INDEX": 1.0,
        "X_RAY_Tvir_MIN": None,
        "F_H2_SHIELD": 0.0,
        "t_STAR": 0.5,
        "N_RSD_STEPS": 20,
        "A_LW": 2.00,
        "BETA_LW": 0.6,
        "A_VCB": 1.0,
        "BETA_VCB": 1.8,
    }

    def __init__(
        self, *args, INHOMO_RECO=FlagOptions._defaults_["INHOMO_RECO"], **kwargs
    ):
        # TODO: should try to get inhomo_reco out of here... just needed for default of
        #  R_BUBBLE_MAX.
        self.INHOMO_RECO = INHOMO_RECO
        super().__init__(*args, **kwargs)

    def convert(self, key, val):
        """Convert a given attribute before saving it the instance."""
        if key in [
            "F_STAR10",
            "F_ESC10",
            "F_STAR7_MINI",
            "F_ESC7_MINI",
            "M_TURN",
            "ION_Tvir_MIN",
            "L_X",
            "L_X_MINI",
            "X_RAY_Tvir_MIN",
        ]:
            return 10**val
        else:
            return val

    @property
    def R_BUBBLE_MAX(self):
        """Maximum radius of bubbles to be searched. Set dynamically."""
        if not self._R_BUBBLE_MAX:
            return 50.0 if self.INHOMO_RECO else 15.0
        if self.INHOMO_RECO and self._R_BUBBLE_MAX != 50:
            logger.warn(
                "You are setting R_BUBBLE_MAX != 50 when INHOMO_RECO=True. "
                "This is non-standard (but allowed), and usually occurs upon manual "
                "update of INHOMO_RECO"
            )
        return self._R_BUBBLE_MAX

    @property
    def X_RAY_Tvir_MIN(self):
        """Minimum virial temperature of X-ray emitting sources (unlogged and set dynamically)."""
        return self._X_RAY_Tvir_MIN or self.ION_Tvir_MIN

    @property
    def NU_X_THRESH(self):
        """Check if the choice of NU_X_THRESH is sensible."""
        if self._NU_X_THRESH < 100.0:
            raise ValueError(
                "Chosen NU_X_THRESH is < 100 eV. NU_X_THRESH must be above 100 eV as it describes X-ray photons"
            )
        elif self._NU_X_THRESH >= global_params.NU_X_BAND_MAX:
            raise ValueError(
                """
                Chosen NU_X_THRESH > {}, which is the upper limit of the adopted X-ray band
                (fiducially the soft band 0.5 - 2.0 keV). If you know what you are doing with this
                choice, please modify the global parameter: NU_X_BAND_MAX""".format(
                    global_params.NU_X_BAND_MAX
                )
            )
        else:
            if global_params.NU_X_BAND_MAX > global_params.NU_X_MAX:
                raise ValueError(
                    """
                    Chosen NU_X_BAND_MAX > {}, which is the upper limit of X-ray integrals (fiducially 10 keV)
                    If you know what you are doing, please modify the global parameter:
                    NU_X_MAX""".format(
                        global_params.NU_X_MAX
                    )
                )
            else:
                return self._NU_X_THRESH

    @property
    def t_STAR(self):
        """Check if the choice of NU_X_THRESH is sensible."""
        if self._t_STAR <= 0.0 or self._t_STAR > 1.0:
            raise ValueError("t_STAR must be above zero and less than or equal to one")
        else:
            return self._t_STAR


class InputCrossValidationError(ValueError):
    """Error when two parameters from different structs aren't consistent."""

    pass


def convert_input_dicts(
    user_params: dict | UserParams | None = None,
    cosmo_params: dict | CosmoParams | None = None,
    astro_params: dict | UserParams | None = None,
    flag_options: dict | FlagOptions | None = None,
    *,
    defaults: bool = False,
):
    """Convert a full set of input params structs/dicts into their actual classes.

    The 4 parameters can be provided as either positional or keyword arguments. They
    can be passed as a dict (which will be converted to the appropriate class), a
    StructWithDefaults instance (in which case it will be left alone), or None (in
    which case *either* the default struct will be returned, or None).

    Returns
    -------
    user_params, cosmo_params, astro_params, flag_options : UserParams, CosmoParams, AstroParams, FlagOptions
        The validated and converted input parameters.
    """
    if defaults:
        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)
        flag_options = FlagOptions(flag_options)
        astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)
    else:
        user_params = UserParams(user_params) if user_params else None
        cosmo_params = CosmoParams(cosmo_params) if cosmo_params else None
        flag_options = FlagOptions(flag_options) if flag_options else None
        inhomo_reco = (
            flag_options.INHOMO_RECO
            if flag_options is not None
            else FlagOptions().INHOMO_RECO
        )
        astro_params = (
            AstroParams(astro_params, INHOMO_RECO=inhomo_reco) if astro_params else None
        )

    return user_params, cosmo_params, astro_params, flag_options


def validate_all_inputs(
    user_params: UserParams,
    cosmo_params: CosmoParams,
    astro_params: AstroParams | None = None,
    flag_options: FlagOptions | None = None,
):
    """Cross-validate input parameters from different structs.

    The input params may be modified in-place in this function, but if so, a warning
    should be emitted.
    """
    if astro_params is not None:
        if astro_params.R_BUBBLE_MAX > user_params.BOX_LEN:
            astro_params.update(R_BUBBLE_MAX=user_params.BOX_LEN)
            warnings.warn(
                f"Setting R_BUBBLE_MAX to BOX_LEN (={user_params.BOX_LEN} as it doesn't make sense for it to be larger."
            )

        if (
            global_params.HII_FILTER == 1
            and astro_params.R_BUBBLE_MAX > user_params.BOX_LEN / 3
        ):
            msg = (
                "Your R_BUBBLE_MAX is > BOX_LEN/3 "
                f"({astro_params.R_BUBBLE_MAX} > {user_params.BOX_LEN / 3})."
            )

            if config["ignore_R_BUBBLE_MAX_error"]:
                logger.warn(msg)
            else:
                raise ValueError(msg)

    if flag_options is not None and (
        flag_options.USE_MINI_HALOS
        and not user_params.USE_RELATIVE_VELOCITIES
        and not flag_options.FIX_VCB_AVG
    ):
        logger.warn(
            "USE_MINI_HALOS needs USE_RELATIVE_VELOCITIES to get the right evolution!"
        )
