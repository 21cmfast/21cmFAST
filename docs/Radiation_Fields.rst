Radiation Fields in ``21cmFAST``
================================

The 21-cm signal is highly sensitive to the radiation fields present in the early Universe. Strictly speaking,
by radiation fields we refer to the spatially varying flux of photons in the Lyman-alpha, X-ray, and UV bands.
However, in the context of ``21cmFAST``, we refer to only physical fields that can be represented as a 3D box
at a given time/redshift.

For example, the Lyman-alpha flux is certainly such a field, since all the photons that contribute to the Lyman-alpha flux
at a given point in spacetime have a single frequency, the Lyman-alpha frequency, and thus the Lyman-alpha flux is
only a function of space and time. In contrast, the X-ray flux is a function of space, time, and frequency, and thus
cannot be represented as a single 3D box at a given redshift. Yet, physical quantities that depend only on space and time
can still be derived from the X-ray flux.

Take for example the energy transfer rate in the IGM due to X-ray radiation. Given the local X-ray flux :math:`J_X(\nu)` in
a given point in spacetime (a quantity with units of photon number per unit area, time, frequency and solid angle), the heat
transfer rate in that point is given by

.. math::

    \Gamma_X = 4 \pi \int_{\nu_{min}}^{\infty} d\nu \sum_i (h_{\rm P}\nu - E^i_{\rm th}) f_{\rm heat} f_i \sigma_i(\nu) J_X(\nu),

where :math:`h_{\rm P}` is Planck's constant, :math:`\nu_{min}` is the minimum frequency for ionizing X-ray photons to escape the host
galaxy, :math:`i` runs over the species in the IGM that are relevant to the process of energy transfer by X-ray photons (i.e. HI, HeI, HeII),
:math:`E^i_{\rm th}` is the ionization threshold energy of species :math:`i`, :math:`f_{\rm heat}` is the fraction of the energy of an
ionized electron of energy :math:`E_e = h_{\rm P}\nu - E^i_{\rm th}` that is deposited as heat in the IGM (a function of :math:`E_e` and
the local ionization fraction), :math:`f_i` is the number fraction of species :math:`i` in the IGM, and :math:`\sigma_i(\nu)` is the
cross-section for photo-ionization of a particle from species :math:`i` due to the interaction with an X-ray photon of frequency :math:`\nu`.
To understand the above expression, let us consider the following components:

* :math:`f_i \sigma_i(\nu) J_X(\nu)` is the production rate of electrons (with energy :math:`E_e`) per gas particle per
  frequency (of the incoming photons) per solid angle (of the incoming photons), that were produced from the
  interaction of particles of species :math:`i` with X-ray photons.
* :math:`(h_{\rm P}\nu - E^i_{\rm th})f_{\rm heat}` is the energy deposited to the IGM as heat per electron with energy :math:`E_e`.
* Once we apply :math:`\int_{\nu_{min}}^{\infty} d\nu \sum_i` we account for all the species and all the frequencies. The multiplication
  by :math:`4\pi` is needed in order to account all the directions from which the X-ray photons come. Combining everything together we end
  up with the desired heat transfer rate per gas particle.

Given the heat transfer rate, the X-ray heating rate is given by

.. math::

    \left.\frac{dT_k}{dt}\right|_X = \frac{2}{3 k_B} \frac{\Gamma_X}{1 + x_e},

where :math:`k_B` is Boltzmann's constant and :math:`x_e` is the local ionization fraction. The X-ray heating rate is thus a physical
quantity that can be represented as a 3D box at a given redshift, and is thus considered a "radiation field" in ``21cmFAST``.
Similarly, the photoionization rate due to X-ray radiation, as well as the contribution to the Lyman-alpha flux from X-ray excitation of HI
atoms, are also considered "radiation fields".

Fluxes in ``21cmFAST``
-----------------------

The X-ray flux :math:`J_X(\nu, z, \mathbf{x})` is given by integrating over the contributions of all past and distant sources that
emitted X-ray photons that ultimately reached the point of interest :math:`\mathbf{x}` at redshift :math:`z` and frequency :math:`\nu`.
These sources can be grouped into spherical shells of comoving size :math:`R` around the point of interest, and thus the X-ray flux
can be expressed as an integral over :math:`R`:

.. math::

    J_X(\nu, z, \mathbf{x}) = \frac{(1+z)^2}{4\pi} \int_0^{\infty} dR I_X(\nu') \epsilon_X^{\rm eff}(z', \mathbf{x}) e^{-\tau_X(\nu, z')},

where :math:`z'` is the redshift that corresponds to the time of emission and is given by inverting the relation :math:`R = \int_z^{z'} c dz''/H(z'')`
with :math:`H(z)` being the Hubble parameter and :math:`c` the speed of light, :math:`I_X(\nu')\propto\nu'^{-\alpha_X}` is the spectral energy
distribution (SED) of the X-ray sources, with :math:`\nu' = \nu (1+z')/(1+z)` the frequency of the photon during emission, and
:math:`\tau_X(\nu, z')` is the optical depth to X-ray photons emitted at redshift :math:`z'` and observed at frequency :math:`\nu`. Lastly,
:math:`\epsilon_X^{\rm eff}(z', \mathbf{x})` is the effective X-ray emissivity of the sources from redshift :math:`z'`, as seen
by the point :math:`\mathbf{x}`, a quantity with units of photon number per comoving volume. Quantities with such units in ``21cmFAST``
are considered as "emissivity fields", and can also be represented as 3D boxes at a given redshift.

The X-ray optical depth is given by

.. math::

    \tau_X(\nu, z') = \int_z^{z'} dz'' \frac{c}{(1+z'')H(z'')} \sum_i \bar n_i(z'') \sigma_i(\nu''),

where :math:`\bar n_i(z'')` is the mean number density of species :math:`i` at redshift :math:`z''`. Note that the optical depth in
``21cmFAST`` is homogeneous and isotropic as it is computed using the mean number density of the IGM, :math:`\bar n_i(z'')`,
regardless of the actual density field that the photon encountered in its path, thus allowing ``21cmFAST`` to run much faster than
traditional radiative transfer codes.

The Lyman alpha flux from continuum photons (photons that were emitted with frequency below Lyman beta and redshifted into Lyman alpha) and
injected photons (photons that were emitted with frequency above Lyman beta and went through atomic cascades once they had redshifted
into a Lyman resonance) is computed in a similar manner,

.. math::

    J_\alpha(z, \mathbf{x}) = \frac{(1+z)^2}{4\pi}\sum_{n=2} f_{\rm recycle}(n)\int_z^{z_{\rm max}(n)} \frac{cdz''}{H(z'')} I_\alpha(\nu') \epsilon_\alpha^{\rm eff}(z', \mathbf{x}),

where :math:`f_{\rm recycle}(n)` is the probability that a photon from Lyman resonance :math:`n` will atomically cascade into a Lyman-alpha
photon, and :math:`1+z_{\rm max}(n)=(1+z)[1-(n-1)^{-2}]/(1-n^{-2})`. Note that there is no optical depth term in the Lyman-alpha flux,
as the IGM is assumed to be optically thin to any photon with a frequency between Lyman resonances.

``21cmFAST`` v1.0.0
-----------------------

In the first public release of ``21cmFAST`` (v1.0.0, see Mesinger et al. 2010,
https://arxiv.org/pdf/1003.3878) there were no radiation fields nor emissivity fields outputs. Instead, they were computed internally as
part of the spin temperature calculation, and were not accessible to the user.

Since the X-ray radiation fields (X-ray heating rate, X-ray ionization rate, and Lyman-alpha flux from X-ray excitation of HI) involve two
explicit integrals, one over the frequency :math:`\nu` and one over the comoving distance :math:`R` (or equivalently, over the redshift
:math:`z'`), together with an implicit integral for the optical depth, attempting to compute them naively according to their definitions
would be extremely slow. Instead, ``21cmFAST`` v1.0.0. used the following approximation:

.. math::

    \int_{\nu_{min}}^{\infty} d\nu \int_z^{\infty} dz' e^{-\tau_X(\nu, z')} \approx \int_z^{\infty} dz' \int_{\min[\nu_{min},\nu_{\tau_X=1}]}^{\infty} d\nu,

where :math:`\nu_{\tau_X=1}` is the frequency at which the X-ray optical depth is unity. In other words, the code assumed that only photons
with X-ray optical depth less than unity contribute to the X-ray radiation fields.

Emissivity Fields
~~~~~~~~~~~~~~~~~~~

For the emissivity fields, the code assumed that they follow the derivative of the conditional collapsed fraction of halos above the
turnover mass :math:`M_{\rm turn}` (see more details about the turnover mass in :doc:`M_TURN`). The X-ray emissivity field was
given by

.. math::

    \epsilon_X(z, \delta) = \bar\rho_b(z=0) (1+\delta) \zeta_X f_* \frac{d}{dt}f_{\rm coll}(z, M_{\rm turn} ; \delta, M_{\rm cond}),

where :math:`\bar\rho_b` is the mean baryon number density, :math:`f_*` is the star formation efficiency, :math:`\zeta_X` is the number of
X-ray photons per unit stellar mass (the latter two are free parameters). Meanwhile, the Lyman-alpha emissivity field was given by

.. math::

    \epsilon_\alpha(z, \delta) = \bar n_b(z=0) (1+\delta) N_{\gamma /{\rm b}} f_* \frac{d}{dt}f_{\rm coll}(z, M_{\rm turn} ; \delta, M_{\rm cond}),

where :math:`\bar n_b` is the mean baryon number density and :math:`N_{\gamma /{\rm b}}` is the number of stellar photons per baryon
(the latter is a free parameter).

Note that both :math:`\epsilon_X` and :math:`\epsilon_\alpha` are functions of the local overdensity :math:`\delta` and thus
are spatially varying fields.

As for the conditional collapsed fraction, it is given by

.. math::

    f_{\rm coll}(z, M_{\rm turn} ; \delta, M_{\rm cond}) = \frac{1}{\bar\rho_m} \int_{M_{\rm turn}}^{\infty} dM_h M_h\frac{dn}{dM_h}(z, M_h ; \delta, M_{\rm cond}),

where :math:`\bar\rho_m` is the mean matter density and :math:`dn/dM_h(z, M_h ; \delta_{\rm cond}, M_{\rm cond})` is the conditional halo
mass function in a region of size :math:`R(M_{\rm cond}) = [3M_{\rm cond}/(4\pi \bar\rho_m)]^{1/3}` and overdensity :math:`\delta_{\rm cond}`.
For the calculation of the conditional collapsed fraction, ``21cmFAST`` v1.0.0 assumed that the conditional halo mass function was given
by the Sheth-Tormen conditional mass function. Under this assumption, the conditional collapsed fraction has an analytical form,

.. math::

    f_{\rm coll}(z, M_{\rm turn} ; \delta, M_{\rm cond}) = {\rm erfc}\left[\frac{\delta_c - \delta}{D(z)\left(2(\sigma^2(M_{\nu}) - \sigma^2(M_{\rm cond}))\right)^{1/2}}\right],

where :math:`{\rm erfc}` is the complementary error function, :math:`\delta_c=1.686` is the critical overdensity for collapse,
and :math:`\sigma^2(M)` is the variance of the linear density field, smoothed with a top-hat filter of
radius :math:`R(M) = [3M/(4\pi \bar\rho_m)]^{1/3}`.

While ``21cmFAST`` v1.0.0 used the analytical result for the conditional collapsed fraction from the extended Press-Schechter formalism
in order to assess the fluctuations in the emissivity fields, the code also normalized the mean collapsed fraction in the box to match
the global collapsed fraction, as given by solving the above integral numerically with the user's selected halo mass function
(which by default was the Sheth-Tormen mass function).

Effective Emissivity Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given the above definitions for the emissivity fields, the effective emissivity fields are achieved by interpolating in time/redshift and
filtering in space. Since ``21cmFAST`` v1.0.0 did not have the emissivity fields on a grid, the interpolation and filtering was done in
the following way. Firstly, given the Eulerian density field :math:`\delta_{\rm E}(z, \mathbf{x})` at redshift :math:`z` (the current
snapshot's redshift), the code evaluated the density field at the integrated redshift :math:`z'` by scaling with the linear growth factor,
namely :math:`\delta_{\rm E}(z', \mathbf{x}) \approx \delta_{\rm E}(z, \mathbf{x}) D(z')/D(z)`. Then, instead of filtering the emissivity
fields, the code filtered the density field with a top-hat filter of radius :math:`R` to get the smoothed density field
:math:`\delta_{\rm E}^R(z', \mathbf{x})`. Thus the effective emissivity fields were given by

.. math::

    \epsilon_X(z, \mathbf{x}) \to \epsilon_X^{\rm eff}(z', \mathbf{x})=\epsilon_X[z', \delta_{\rm E}^R(z,\mathbf{x}) D(z')/D(z)],

and

.. math::
    \epsilon_\alpha(z, \mathbf{x}) \to \epsilon_\alpha^{\rm eff}(z', \mathbf{x})=\epsilon_\alpha[z', \delta_{\rm E}^R(z,\mathbf{x}) D(z')/D(z)].

In addition, the code set the conditional mass for the conditional collapsed fraction to be :math:`M_{\rm cond} = (4\pi/3) \bar\rho_m R^3`,
where :math:`R` is the size of the integrated comoving shell in the evaluation of the radiation fields.

``21cmFAST`` v2.0.0
-----------------------

In the second public release of ``21cmFAST`` (v2.0.0, see Park et al. 2018,
https://arxiv.org/pdf/1809.08995.pdf) the radiation fields and the emissivity fields were still computed internally as part of
the spin temperature calculation. However, drastic changes were made in the modeling of the emissivity fields due to the introduction
of a new flag called ``USE_MASS_DEPENDENT_ZETA``. In this context, "zeta" refers to the ionizing efficiency from galaxies, which was
assumed to be mass-independent in ``21cmFAST`` v1.0.0, but was now allowed to be mass-dependent in ``21cmFAST`` v2.0.0 when
``USE_MASS_DEPENDENT_ZETA`` was set to True. Under this configuration, the modeling of the emissivity fields was very different
than in ``21cmFAST`` v1.0.0 (which corresponded to setting ``USE_MASS_DEPENDENT_ZETA`` to False in ``21cmFAST`` v2.0.0).

When ``USE_MASS_DEPENDENT_ZETA`` was set to True, the X-ray emissivity field was given by

.. math::

    \epsilon_X(z, \delta) = \frac{1}{h_{\rm P}}\frac{L_X}{\rm SFR} \dot\rho_*(z, M_{\rm turn} ; \delta, M_{\rm cond}),

where :math:``\dot\rho_*`` is the (conditional) star formation rate density (SFRD), and :math:`L_X/{\rm SFR}` is the galaxy
X-ray luminosity per unit star formation rate (the latter is a free parameter). The Lyman-alpha emissivity field was now
also modeled via the SFRD and was given by

.. math::

    \epsilon_\alpha(z, \delta) = \frac{N_{\gamma /{\rm b}}}{m_b} \dot\rho_*(z, M_{\rm turn} ; \delta, M_{\rm cond}),

where :math:`m_b` is the mean baryon mass.

Meanwhile, the SFRD was given by

.. math::

    \dot\rho_*(z, M_{\rm turn} ; \delta, M_{\rm cond}) = (1+\delta)\int_{0}^{\infty} dM_h \dot M_*(M_h, z) \frac{dn}{dM_h}(z, M_h ; \delta, M_{\rm cond}) f_{\rm duty}(M_h; M_{\rm turn}),

where :math:`\dot M_*(M_h, z)` is the star formation rate (SFR) in a halo of mass :math:`M_h` at redshift :math:`z`, given by

.. math::

    \dot M_*(M_h, z) = \frac{\Omega_b}{\Omega_m}f_{* , 10}\left(\frac{M_h}{10^{10} M_\odot}\right)^{\alpha_*} \frac{M_h}{t_*H^{-1}(z)},

where :math:`f_{* , 10}` is the star formation efficiency in halos of mass :math:`10^{10} M_\odot`, :math:`\alpha_*` is the power-law index
of the mass-dependent star formation efficiency, and :math:`t_*` is the star formation timescale in units of the Hubble time (the latter
three are free parameters). Lastly, :math:`f_{\rm duty}(M_h; M_{\rm turn})` is the duty fraction of star formation in halos of mass
:math:`M_h`, given by :math:`f_{\rm duty}(M_h; M_{\rm turn}) = \exp(-M_{\rm turn}/M_h)` (see more details about the duty fraction
in :doc:`M_TURN`). Note that for brevity, the above forumla assumes that the mass-dependent star formation efficiency is
:math:`f_*(M_h) = f_{* , 10}(M_h/10^{10} M_\odot)^{\alpha_*}`, but the code ensured that :math:`f_*(M_h) \leq 1` for all halo masses.

Interestingly, while this modeling of the emissivity fields is much different than the one used in ``21cmFAST`` v1.0.0, it is mathematically
related to the previous modeling, by setting :math:`\dot\rho_*= \bar\rho_b f_* df_{\rm coll}/dt`
and :math:`\frac{1}{h_{\rm P}}\frac{L_X}{\rm SFR} = \zeta_X`. In other words, the SFRD in ``21cmFAST`` v1.0.0 was proportional to the
time derivative of the conditional collapsed fraction.

Even though the modeling of the emissivity fields in ``21cmFAST`` v2.0.0 was very different than in ``21cmFAST`` v1.0.0,
the prescription for the effective emissivity fields remained the same (i.e. the Eulerian density field was filtered with a top-hat filter
and was scaled by the linear growth factor for interpolating the field at :math:`z'`).

``21cmFAST`` v3.0.0
-----------------------

In the third public release of ``21cmFAST`` (v3.0.0, see Qin et al. 2020,
https://arxiv.org/pdf/2003.04442), a new population of molecular cooling galaxies (MCGs) that reside in mini-halos was introduced.
The formation of such MCGs is susceptible to feedback from Lyman-Werner (LW) radiation, which can dissociate molecular hydrogen and suppress
star formation in MCGs. Hence, the LW flux was a new radiation field that was introduced in ``21cmFAST`` v3.0.0, though still internally in
the code as in previous versions. The LW flux is given by

.. math::

    J_{\rm LW}(z, \mathbf{x}) = h_{\rm P}\nu_\alpha(1-f_{\rm H_2}^{\rm shield})\frac{(1+z)^2}{4\pi}\sum_{n=2}\int_z^{z_{\rm max}(n)} \frac{cdz''}{H(z'')} I_\alpha(\nu') \epsilon_\alpha^{\rm eff}(z', \mathbf{x}),

where :math:`\nu_\alpha` is the Lyman-alpha frequency and :math:`f_{\rm H_2}^{\rm shield}` accounting for self-shielding of star-forming regions
by the ISM and the circumgalactic medium of the host galaxy. Several notes:

* Note that unlike previous radiation fields, the LW flux contains units of energy, owned by the factor of :math:`h_{\rm P}\nu_\alpha`.
* The contribution to the LW flux came from both atomic cooling galaxies (ACGs) and MCGs, as the SFRD in both populations is modeled a bit
  differently, mostly in the modeling of the star formation efficiency, the turnover mass and the duty fraction (see more details on the
  latter two in :doc:`M_TURN`). Likewise, the contribution to the X-ray and Lyman-alpha fluxes also came from both ACGs and MCGs.
* It was assumed that ACGs contained only popII stars, while MCGs contained only popIII stars. Therefore,
  the shape of the SEDs in ACGs and MCGs were also different in the evaluation of Lyman-alpha and LW fluxes, as well as their amplitudes
  (controlled by the free parameter :math:`N_{\gamma /{\rm b}}`, which was now split into two free parameters). However, for the evaluation
  of the X-ray flux, the SEDs in both ACGs and MCGs were assumed to be the same, and were modeled as in previous versions, namely
  :math:`I_X(\nu')\propto\nu'^{-\alpha_X}`, where :math:`\alpha_X` is a free parameter.

The prescription for the effective emissivity fields in ``21cmFAST`` v3.0.0 remained the same as in ``21cmFAST`` v4.0.0 (i.e. the Eulerian
density field was filtered with a top-hat filter and was scaled by the linear growth factor for interpolating the field at :math:`z'`).

``21cmFAST`` v4.0.0
-----------------------

In the fourth public release of ``21cmFAST`` v4.0.0 (Davies et al. 2025, https://arxiv.org/pdf/2504.17254), the evaluation of the
emissivity fields and the radiation fields was changed for several new configurations that were introduced in that version. This was
controlled by the new enum parameter ``SOURCE_MODEL``, which had five options:

* ``"CONST-ION-EFF"``:
  This option was equivalent to setting ``USE_MASS_DEPENDENT_ZETA`` to False in previous versions.

* ``"E-INTEGRAL"``:
  This option was equivalent to setting ``USE_MASS_DEPENDENT_ZETA`` to True in previous versions.

* ``"L-INTEGRAL"``:
  This is a new source model that was introduced in v4.0.0. It was similar to ``"E-INTEGRAL"``, but the emissivity
  fields were computed on the Lagrangian density grid, and then mapped to the Eulerian grid (see details below).

* ``"DEXM-ESF"``:
  This is a new source model that was introduced in v4.0.0, similar to ``"L-INTEGRAL"``, but the emissivity fields received
  contributions from resolved discrete halos that were found via the excursion-set formalism (ESF), as was implemented by
  the ``DEXM`` algorithm (see Mesinger & Furlanetto 2007, https://arxiv.org/pdf/0704.0946).

* ``"CHMF-SAMPLER"``:
  This is a new source model that was introduced in v4.0.0, similar to ``"DEXM-ESF"``, but now unresolved
  discrete halos were sampled from the conditional halo mass function (CHMF) in each cell of the
  simulation box at the lowest redshift, whereas at higher redshifts progenitor halos were sampled
  from their bigger descendent halos.

The first two source models, ``"CONST-ION-EFF"`` and ``"E-INTEGRAL"``, were considered as "Eulerian" source models
(as the emissivity fields were computed on the Eulerian density grid), while the last three source models, were considered
as "Lagrangian" source models. This distinction was important in how the emissivity and radiation fields were evaluated in ``21cmFAST``
v4.0.0.

For the two "Eulerian" source models, the emissivity and radiation fields were computed very similarly as in previous versions
of the code. Namely, these fields were an internal part of the spin temperature calculation, and the effective emissivity fields
were computed by filtering the Eulerian density field with a top-hat filter and scaling by the linear growth factor for interpolating
the fields at :math:`z'`:

.. math::

    \epsilon(z, \mathbf{x}) \to \epsilon^{\rm eff}(z', \mathbf{x})=\epsilon[z', \delta_{\rm E}^R(z,\mathbf{x}) D(z')/D(z)].

In contrast, for the three "Lagrangian" source models, the emissivity fields were computed on the Lagrangian density grid, and were
then advected to the Eulerian grid by an algorithm similar to 2LPT. Moreover, the emissivity fields in the "Lagrangian" source models
were computed as an intermediate output of the code (stored in the ``HaloBox`` class). This enabled to interpolate the emissivity fields
at redshift :math:`z'`. Then, for each spherical integrated shell with inner radius :math:`R_{\rm i}` and outer radius :math:`R_{\rm o}`,
the interpolated emissivity fields :math:`\epsilon(z', \mathbf{x})` were filtered with a spherical shell filter with the appropriate radii.
All the effective emissivities, for all the spherical integrated shells, were then stored in the ``XraySourceBox`` class, which contained
4D arrays (three spatial dimensions, and an extra dimension that indicates the integrated shell). Thus, in contrast to the "Eulerian"
source models, the effective emissivity fields in the "Lagrangian" source models were given by

.. math::

    \epsilon(z, \mathbf{x}) \to \epsilon^{\rm eff}(z', \mathbf{x})=\epsilon^{R_{\rm i}, R_{\rm o}}(z', \mathbf{x}).

For both "Eulerian" and "Lagrangian" source models, the radiation fields were still computed as an internal part of the spin
temperature calculation, and were therefore still not accessible to the user.

``21cmFAST`` v4.3.0
-----------------------

In ``21cmFAST`` v4.3.0, in both types of source models, "Eulerian" and "Lagrangian", the effective emissivity fields were computed exactly
the same given the emissivity fields. This computation follows the prescription that was used in ``21cmFAST`` v4.0.0 for the "Lagrangian"
source models, namely

.. math::

    \epsilon(z, \mathbf{x}) \to \epsilon^{\rm eff}(z', \mathbf{x})=\epsilon^{R_{\rm i}, R_{\rm o}}(z', \mathbf{x}).

The difference between "Eulerian" and "Lagrangian" source models in v4.3.0 (specifically, between ``"E-INTEGRAL"`` and ``"L-INTEGRAL"``) was
that in the former, the emissivity fields were computed on the Eulerian density grid, while in the latter, the emissivity fields were
computed on the Lagrangian density grid and then advected to the Eulerian grid.

Furthermore, in v4.3.0, the effect of Lyman-alpha multiple scattering could have been applied via a new flag called ``LYA_MULTIPLE_SCATTERING``,
regardless if the simulated source model was "Eulerian" or "Lagrangian". When this flag was set to True, the effective emissivity for
Lyman-alpha photons was achieved by filtering the interpolated emissivity field with a generalization of the spherical shell filter
(see more details in Flitter, Munoz and Mesinger 2026, https://arxiv.org/pdf/2601.14360).

``21cmFAST`` v4.3.0 was also the first version in which the radiation fields were accessible to the user via a new output class called
``RadiationFields``. This class contained the 3D realizations of the following fields:

* ``xray_heating_rate``: the X-ray heating rate.
* ``xray_ionization_rate``: the X-ray ionization rate.
* ``xray_lya_flux``: the Lyman-alpha flux from X-ray excitation of HI.
* ``lya_flux_continuum``: the Lyman-alpha flux from continuum photons.
* ``lya_flux_injected``: the Lyman-alpha flux from injected photons.
* ``lya_flux_continuum_injected``: the total Lyman-alpha flux from both continuum and injected photons.
* ``lyw_flux``: the Lyman-Werner flux.
Meanwhile, the class ``XraySourceBox`` from v4.0.0 was removed in v4.3.0.
