Changelog
=========

dev-version
-----------

v3.1.4 [10 Feb 2022]
----------------------

Fixed
-----

* error in FFT normalization in FindHaloes
* docs not compiling on RTD due to missing ``scipy.integrate`` mock module
* Updated matplotlib removed support for setting vmin/vmax and norm. Now passes vmin/vmax
  to the norm() constructor.

v3.1.3 [27 Oct 2021]
----------------------

* Fixed ``FAST_FCOLL_TABLES`` so it only affects MCGs and not ACGs. Added tests of this
  flag for high and low z separately.

v3.1.2 [14 Jul 2021]
----------------------

Internals
~~~~~~~~~
* ``MINIMIZE_MEMORY`` flag significantly reduces memory without affecting performance much,
  by changing the way some arrays are allocated and accessed in C. (#224)

Change
~~~~~~

* Updated ``USE_INTERPOLATION_TABLES`` to be default True. This makes much more sense as
  a default value. Until v4, a warning will be raised if it is not set explicitly.


v3.1.1 [13 Jun 2021]
----------------------

Fixed
~~~~~

* Bug in deployment to PyPI.

v3.1.0 [13 Jun 2021]
----------------------

Added
~~~~~
* Ability to access all evolutionary Coeval components, either from the end Coeval
  class, or the Lightcone.
* Ability to gather all evolutionary antecedents from a Coeval/Lightcone into the one
  file.
* ``FAST_FCOLL_TABLES`` in ``UserParams`` which improves speeds quite significantly for
  ~<10% accuracy decrease.
* Fast and low-memory generation of relative-velocity (vcb) initial conditions. Eliminated hi-res vcb boxes, as they are never needed.
* Also output the mean free path (i.e. MFP_box in IonizedBox).
* Added the effect of DM-baryon relative velocities on PopIII-forming minihaloes. This now provides the correct background evolution jointly with LW feedback. It gives rise to velocity-induced acoustic oscillations (VAOs) from the relative-velocity fluctuations. We also follow a more flexible parametrization for LW feedback in minihaloes, following new simulation results, and add a new index ALPHA_STAR_MINI for minihaloes, now independent of regular ACGs.
* New ``hooks`` keyword to high-level functions, that are run on the completion of each computational step, and can
  be used to more generically write parts of the data to file.
* Ability to pass a function to ``write=`` to write more specific aspects of the data (internally, this will be put into the ``hooks`` dictionary).
* ``run_lightcone`` and ``run_coeval`` use significantly less memory by offloading initial conditions and perturb_field instances to disk if possible.

Fixed
~~~~~
* Bug in 2LPT when ``USE_RELATIVE_VELOCITIES=True`` [Issue #191, PR #192]
* Error raised when redshifts are not in ascending order [Issue #176, PR #177]
* Errors when ``USE_FFTW_WISDOM`` is used on some systems [Issue #174, PR #199]
* Bug in ComputeIonizedBox causing negative recombination rate and ring structure in ``Gamma12_box`` [Issue #194, PR #210]
* Error in determining the wisdom file name [Issue #209, PR#210]
* Bug in which cached C-based memory would be read in and free'd twice.

Internals
~~~~~~~~~

* Added ``dft.c``, which makes doing all the cubic FFTs a lot easier and more consistent. [PR #199]
* More generic way of keeping track of arrays to be passed between C and Python, and their shape in Python, using ``_get_box_structures``.
  This also means that the various boxes can be queried before they are initialized and computed.
* More stringent integration tests that test each array, not just the final brightness temperature.
* Ability to plot the integration test data to more easily identify where things have gone wrong (use ``--plots`` in the ``pytest`` invocation).
* Nicer CLI interface for ``produce_integration_test_data.py``. New options to ``clean`` the ``test_data/`` directory,
  and also test data is saved by user-defined key rather than massive string of variables.
* Nicer debug statements before calls to C, for easily comparing between versions.
* Much nicer methods of keeping track of array state (in memory, on disk, c-controlled, etc.)
* Ability to free C-based pointers in a more granular way.

v3.0.3
------

Added
~~~~~
* ``coeval_callback`` and ``coeval_callback_redshifts`` flags to the ``run_lightcone``.
  Gives the ability to run arbitrary code on ``Coeval`` boxes.
* JOSS paper!
* ``get_fields`` classmethod on all output classes, so that one can easily figure out
  what fields are computed (and available) for that class.

Fixed
~~~~~
* Only raise error on non-available ``external_table_path`` when actually going to use it.

v3.0.2
------

Fixed
-----
* Added prototype functions to enable compilation for some standard compilers on MacOS.

v3.0.1
------
Modifications to the internal code structure of 21cmFAST

Added
~~~~~
* Refactor FFTW wisdom creation to be a python callable function


v3.0.0
------
Complete overhaul of 21cmFAST, including a robust python-wrapper and interface,
caching mechanisms, and public repository with continuous integration. Changes
and equations for minihalo features in this version are found in
https://arxiv.org/abs/2003.04442

All functionality of the original 21cmFAST v2 C-code has been implemented in this
version, including ``USE_HALO_FIELD`` and performing full integration instead of using
the interpolation tables (which are faster).

Added
~~~~~
* Updated the radiation source model: (i) all radiation fields including X-rays, UV
  ionizing, Lyman Werner and Lyman alpha are considered from two seperated population
  namely atomic-cooling (ACGs) and minihalo-hosted molecular-cooling galaxies (MCGs);
  (ii) the turn-over masses of ACGs and MCGs are estimated with cooling efficiency and
  feedback from reionization and lyman werner suppression (Qin et al. 2020). This can
  be switched on using new ``flag_options`` ``USE_MINI_HALOS``.
* Updated kinetic temperature of the IGM with fully ionized cells following equation 6
  of McQuinn (2015) and partially ionized cells having the volume-weightied temperature
  between the ionized (volume: 1-xHI; temperature T_RE ) and neutral components (volume:
  xHI; temperature: temperature of HI). This is stored in IonizedBox as
  temp_kinetic_all_gas. Note that Tk in TsBox remains to be the kinetic temperature of HI.
* Tests: many unit tests, and also some regression tests.
* CLI: run 21cmFAST boxes from the command line, query the cache database, and produce
  plots for standard comparison runs.
* Documentation: Jupyter notebook demos and tutorials, FAQs, installation instructions.
* Plotting routines: a number of general plotting routines designed to plot coeval
  and lightcone slices.
* New power spectrum option (``POWER_SPECTRUM=5``) that uses a CLASS-based transfer
  function. WARNING: If POWER_SPECTRUM==5 the cosmo parameters cannot be altered, they
  are set to the Planck2018 best-fit values for now (until CLASS is added):
  (omegab=0.02237, omegac= 0.120, hubble=0.6736 (the rest are irrelevant for the
  transfer functions, but in case:  A_s=2.100e-9, n_s=0.9649, z_reio = 11.357)
* New ``user_params`` option ``USE_RELATIVE_VELOCITIES``, which produces initial relative
  velocity cubes (option implemented, but not the actual computation yet).
* Configuration management.
* global params now has a context manager for changing parameters temporarily.
* Vastly improved error handling: exceptions can be caught in C code and propagated to
  Python to inform the user of what's going wrong.
* Ability to write high-level data (``Coeval`` and ``Lightcone`` objects) directly to
  file in a simple portable format.

Changed
~~~~~~~
* ``POWER_SPECTRUM`` option moved from ``global_params`` to ``user_params``.
* Default cosmology updated to Planck18.

v2.0.0
------
All changes and equations for this version are found in https://arxiv.org/abs/1809.08995.

Changed
~~~~~~~

* Updated the ionizing source model: (i) the star formation rates and ionizing escape
  fraction are scaled with the masses of dark matter halos and (ii) the abundance of
  active star forming galaxies is exponentially suppressed below the turn-over halo
  mass, M_{turn}, according to a duty cycle of exp(−M_{turn}/M_{h}), where M_{h} is a
  halo mass.
* Removed the mean free path parameter, R_{mfp}. Instead, directly computes
  inhomogeneous, sub-grid recombinations in the intergalactic medium following the
  approach of Sobacchi & Mesinger (2014)




v1.2.0
------
Added
~~~~~
* Support for a halo mass dependent ionizing efficiency: zeta = zeta_0 (M/Mmin)^alpha,
  where zeta_0 corresponds to  HII_EFF_FACTOR, Mmin --> ION_M_MIN,
  alpha --> EFF_FACTOR_PL_INDEX in ANAL_PARAMS.H


v1.12.0
-------
Added
~~~~~
- Code 'redshift_interpolate_boxes.c' to interpolate between comoving cubes,
  creating comoving light cone boxes.
- Enabled openMP threading  for SMP machines.  You can specify the number of threads
  (for best performace, do not exceed the number of processors) in INIT_PARAMS.H. You do
  not need to have an SMP machine to run the code. NOTE: YOU SHOULD RE-INSTALL FFTW to
  use openMP (see INSTALL file)
- Included a threaded driver file 'drive_zscroll_reion_param.c' set-up to perform
  astrophysical parameter studies of reionization
- Included explicit support for WDM cosmologies; see COSMOLOGY.H.  The prescription is
  similar to that discussed in Barkana+2001; Mesinger+2005, madifying the (i) transfer
  function (according to the Bode+2001 formula; and (ii) including the effective
  pressure term of WDM using a Jeans mass analogy.  (ii) is approximated with a sharp
  cuttoff in the EPS barrier, using 60* M_J found in Barkana+2001 (the 60 is an
  adjustment factor found by fitting to the WDM collapsed fraction).
- A Gaussian filtering step of the PT fields to perturb_field.c, in addition to the
  implicit boxcar smoothing.  This avoids having"empty" density cells, i.e. \delta=-1,
  with some small loss in resolution.  Although for most uses \delta=-1 is ok, some Lya
  forest statistics do not like it.
- Added treatment of the risidual electron fraction from X-ray heating when computing
  the ionization field.  Relatedly, modified Ts.c to output all intermediate evolution
  boxes, Tk and x_e.
- Added a missing factor of Omega_b in Ts.c corresponding to eq. 18 in MFC11.  Users who
  used a previous version should note that their results just effecively correspond to a
  higher effective X-ray efficiency, scaled by 1/Omega_baryon.
- Normalization optimization to Ts.c, increasing performace on arge resolution boxes


Fixed
~~~~~
- GSL interpolation error in kappa_elec_pH for GSL versions > 1.15
- Typo in macro definition, which impacted the Lya background calculation in v1.11 (not applicable to earlier releases)
- Outdated filename sytax when calling gen_size_distr in drive_xHIscroll
- Redshift scrolling so that drive_logZscroll_Ts.c and Ts.c are in sync.

Changed
~~~~~~~
- Output format to avoid FFT padding for all boxes
- Filename conventions to be more explicit.
- Small changes to organization and structure


v1.1.0
------
Added
~~~~~
- Wrapper functions mod_fwrite() and mod_fread() in Cosmo_c_progs/misc.c, which
  should fix problems with the library fwrite() and fread() for large files (>4GB) on
  certain operating systems.
- Included print_power_spectrum_ICs.c program which reads in high resolution initial
  conditions and prints out an ASCII file with the associated power spectrum.
- Parameter in Ts.c for the maximum allowed kinetic temperature, which increases
  stability of the code when the redshift step size and the X-ray efficiencies are large.

Fixed
~~~~~
- Oversight adding support for a Gaussian filter for the lower resolution field.
