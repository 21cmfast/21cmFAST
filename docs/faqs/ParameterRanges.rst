Allowed parameter ranges
========================
The version of 21cmFAST that comes with ``21CMMC`` uses numerous interpolation tables. As such, several parameter 
combinations/choices may have restricted ranges. This, in addition to limits set by observations restricts the ranges
for several of the astrophysical parameters that are available within ``21CMMC``. 

Cosmology
---------

At the present only flat cosmologies are allowed as some in-built cosmological functions do not allow for
non-flat cosmologies. This will likely be remedied in the near future. Thus, only the dark matter
energy density (:math:`\Omega_m`) is available to be varied as :math:`\Omega_\Lambda = 1 - \Omega_m` is enforced.
In all cases, sensible ranges are restricted by Planck and other cosmological probes. For example, see `Kern et al., 2017
<https://ui.adsabs.harvard.edu/abs/2017ApJ...848...23K/abstract>`_ for varying cosmology with ``21CMMC``.

- :math:`\Omega_m` - should be allowed to vary across (0, 1]
- :math:`\Omega_b` - should be allowed to take any non-zero value
- :math:`h` - should be allowed to take any non-zero value
- :math:`\sigma_8` - should be allowed to take any non-zero value. Extreme values may cause issues in combination with the astrophysical parameters governing halo mass (either :math:`M_{min}` or :math:`T_{min,vir}`)
- :math:`n_s` - should be allowed to take any non-zero value

Ionisation astrophysical parameters
-----------------------------------

Allowed astrophysical parameter ranges are determined by which parameter set is being used. Dictated by the choice of
``USE_MASS_DEPENDENT_ZETA`` in the ``FlagOptions`` struct.

If ``USE_MASS_DEPENDENT_ZETA = False`` then the user is selecting the older astrophysical parameterisation for the 
ionising sources. These include any of: (i) :math:`\zeta` - the ionising efficiency (ii) :math:`R_{mfp}` - minimum photon horizon for ionising
photons or (iii) :math:`T_{min,vir}` - minimum halo mass for the ionising sources, :math:`T_{min,vir}`.

- :math:`\zeta` - Largest range used thus far was [5,200] in `Greig & Mesinger, 2015 <https://ui.adsabs.harvard.edu/abs/2017MNRAS.465.4838G/abstract>`_. In principle it can be less than 200 or even extended beyond 200. Going below 5 is also plausible, but starts to cause numerical issues.
- :math:`R_{mfp}` - Only if ``INHOMO_RECO=False``. Typically [5,20] is adopted. Again, going below 5 Mpc cause numerical problems, but any upper limit is allowed. However, beyond 20 Mpc or so, :math:`R_{mfp}` has very little impact on the 21cm power spectrum.
- :math:`T_{min,vir}` - Typically [4,6] is selected, corresponding to :math:`10^4` - :math:`10^6` K corresponding to atomically cooled haloes. It becomes numerically unstable to go beyond :math:`10^6` K owing to interpolation tables with respect to halo mass. It is possible to consider :math:`<10^4` K, however, in doing so it will generate a discontinuity. Internally, :math:`T_{min,vir}` is converted to :math:`M_{min}` using Equation 26 of `Barkana and Loeb, 2001 <https://arxiv.org/pdf/astro-ph/0010468.pdf>`_ whereby the mean molecular weight, :math:`\mu` differs according to the IGM state (:math:`\mu = 1.22` for :math:`<10^4` K and :math:`\mu = 0.59` for :math:`>10^4` K)

If ``USE_MASS_DEPENDENT_ZETA = True`` then the user is selecting the newest astrophysical parameterisation allowing for mass dependent ionisation efficiency as well as constructing luminosity functions (e.g. `Park et al., 2019 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.484..933P/abstract>`_). This allows an expanded set of astrophysical parameter including: (i) :math:`f_{\ast,10}` - star formation efficiency normalised at :math:`10^{10} M_{\odot}` (ii) :math:`\alpha_{\ast}` - power-law scaling of star formation efficiency with halo mass (iii) :math:`f_{esc,10}` - escape fraction normalised at :math:`10^{10} M_{\odot}` (iv) :math:`\alpha_{esc}` - power-law scaling of escape fraction with halo mass (v) :math:`M_{turn}` - Turn-over scale for the minimum halo mass (vi) :math:`t_{star}` - Star-formation time scale and (vii) :math:`R_{mfp}` - minimum photon horizon for ionising photons.

- :math:`f_{\ast,10}` - Typically [-3., 0.] corresponding to a range of :math:`10^{-3} - 1`. In principle :math:`f_{\ast,10}` can exceed unity, as it is the normalisation at :math:`10^{10} M_{\odot}` and would depend on the power-law index, :math:`\alpha_{\ast}` as to whether or not the star-formation efficiency exceeds unity. However, probably no need to consider this scenario.
- :math:`\alpha_{\ast}` - Typically [-0.5,1]. Could be modified for stronger scalings with halo mass.
- :math:`f_{esc,10}` - Typically [-3,0.] corresponding to a range of :math:`10^{-3} - 1`. In principle :math:`f_{esc,10}` can exceed unity, as it is the normalisation at :math:`10^{10} M_{\odot}` and would depend on the power-law index, :math:`\alpha_{esc}` as to whether or not the escape fraction exceeds unity. However, probably no need to consider this scenario.
- :math:`\alpha_{esc}` - Typically [-1.0,0.5]. Could be modified for stronger scalings with halo mass.
- :math:`M_{turn}` - Typically [8,10] corresponding to a range of :math:`10^8` - :math:`10^{10} M_{\odot}` . In principle it could be extended, though less physical. To have :math:`M_{turn} > 10^{10} M_{\odot}` would begin to be inconsistent with existing observed luminosity functions. Could go lower than :math:`M_{turn} < 10^{8} M_{\odot}` though it could begin to clash with internal limits in the code for interpolation tables (which are set to :math:`M_{min} = 10^6  M_{\odot}` and :math:`M_{min} = M_{turn}/50`.
- :math:`t_{star}` - Typically (0,1). This is represented as a fraction of the Hubble time. Thus, cannot go beyond this range
- :math:`R_{mfp}` - same as above


Heating astrophysical parameters
--------------------------------

For the epoch of heating, there are three additional parameters that can be set. These, can only be used if ``USE_TS_FLUCT=True`` which performs the heating. These include: (i) :math:`L_{X<2keV}/SFR` - the soft-band X-ray luminosity of the heating sources (ii) :math:`E_{0}` - the minimum threshold energy for X-rays escaping into the IGM from their host galaxies and (iii) :math:`\alpha_{X}` - the power-law spectral index of the X-ray spectral energy distribution function.

- :math:`L_{X<2keV}/SFR` - Typically [38, 42], corresponding to a range of :math:`10^{38} - 10^{42} erg\,s^{-1} M^{-1}_{\odot} yr`. This range could easily be extended depending on the assumed sources. This range corresponds to high mass X-ray binaries.
- :math:`E_{0}` - [100, 1500]. Range is in :math:`eV` corresponding to 0.1 - 1.5 :math:`keV`. Luminosity is determined in the soft-band (i.e. < 2 keV), thus wouldn’t want to expand this upper limit too much. Observations limit the lower range to ~0.1 keV.
- :math:`\alpha_{X}` - Typically [-2.,2] but depends on the population of sources being considered (i.e. what is producing the X-ray’s). Note, the X-ray SED is defined as :math:`\propto \nu^{-\alpha_X}`