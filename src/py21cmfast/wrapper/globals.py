"""A module holding a GlobalParams singleton instance."""

import contextlib
import warnings
from pathlib import Path

from .._cfg import config
from .._data import DATA_PATH
from ..c_21cmfast import ffi, lib
from .structs import StructInstanceWrapper


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
        JD: This has been removed in the C code, RECFAST is always called
    XION_at_Z_HEAT_MAX : float
        If positive, then overwrite default boundary conditions for the evolution
        equations with this value. The default is to use the value obtained from RECFAST.
        See also `TK_at_Z_HEAT_MAX`.
        JD: This has been removed in the C code, RECFAST is always called
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
