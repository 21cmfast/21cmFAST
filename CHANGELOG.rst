Changelog
=========

v3.0.0.dev
----------
Complete overhaul of 21cmFAST, including a robust python-wrapper and interface,
caching mechanisms, and public repository with continuous integration.

Added
~~~~~
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
