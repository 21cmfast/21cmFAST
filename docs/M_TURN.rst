Turnover Mass in ``21cmFAST``
=============================

One of the most important physical quantities in ``21cmFAST`` is the mass scale at
which the gas inside halos can cool down to form stars. This is often referred to as
the "turnover mass" or :math:`M_{\rm turn}`. The value of :math:`M_{\rm turn}` can have
significant implications for the formation of the first galaxies and the subsequent evolution
of the intergalactic medium (IGM). Mathematically, the effect of the turnover mass is
modeled by the duty fraction function :math:`f_{\rm duty}(M_h;M_{\rm turn})`, which represents
the fraction of halos of mass :math:`M_h` that can host galaxies, given a turnover
mass :math:`M_{\rm turn}`. The duty fraction function is thus expected to have a cutoff
(either sharp or smooth) at the turnover mass.

Over the time, both the turnover mass and the duty fraction function have been modified
in different versions of ``21cmFAST``. We document below the changes in these properties.

``21cmFAST`` v1.0.0
----------------------

In the first public release of ``21cmFAST`` (v1.0.0, see Mesinger et al. 2010,
https://arxiv.org/pdf/1003.3878), the duty fraction function was modeled as a sharp
cutoff at the turnover mass,

.. math::

    f_{\rm duty}(M_h;M_{\rm turn}) = H(M_h-M_{\rm turn}),


where :math:`H(x)` is the Heaviside step function. Meanwhile, the turnover mass was redshift-dependent
and was defined as the halo mass that corresponds to a virial temperature of :math:`10^4` K, which is
the threshold mass for atomic cooling, :math:`M_{\rm atom}(z)`. This mass scale was obtained by inverting
Eq. 26 in Barkana & Loeb 2001 (https://arxiv.org/pdf/astro-ph/0010468) with :math:`T_{\rm vir}=10^4` K
and a mean molecular weight of :math:`\mu=0.59`, corresponding to a fully ionized IGM. Approximately,
the turnover mass can be written as

.. math::

    M_{\rm turn}=M_{\rm atom}(z) \approx 3.3\times 10^7 \left(\frac{1+z}{21}\right)^{-3/2} M_\odot.

``21cmFAST`` v2.0.0
-------------------

In the second public release of ``21cmFAST`` (v2.0.0, see Park et al. 2018,
https://arxiv.org/pdf/1809.08995.pdf), the duty fraction function was modified to a smooth
exponential cutoff at the turnover mass,

.. math::

    f_{\rm duty}(M_h;M_{\rm turn}) = \exp\left(-M_{\rm turn}/M_h\right).

Here, :math:`M_{\rm turn}` was introduced as a new free parameter that the user had to provide.
In addition, this version also introduced a mass-dependent ionization efficiency model.
Note that a sharp cutoff at the turnover mass (like in v1.0.0.) was still available in v2.0.0,
but only if the user set the ionization efficiency to be mass-independent.

``21cmFAST`` v3.0.0
-------------------

In the third public release of ``21cmFAST`` (v3.0.0, see Qin et al. 2020,
https://arxiv.org/pdf/2003.04442) the code has been overhauled to include a
python-wrapper and interface. In addition, two galaxy populations were introduced,
one for atomic cooling galaxies (ACGs) and one for molecular cooling galaxies (MCGs).
Two kinds of duty fraction functions were used in v3, each of which corresponded to
one of the two galaxy populations. The duty fraction function for ACGs was kept the
same as in v2.0.0 (if MCGs were present), namely

.. math::

    f_{\rm duty}^{\rm (ACG)}(M_h;M_{\rm turn}) = \exp\left(-M_{\rm turn}^{\rm (ACG)}/M_h\right).

while the duty fraction function for MCGs was modeled as two exponential cutoffs, one for
the minimum halo mass to host MCGs, :math:`M_{\rm turn}^{\rm (MCG)}`, while the other one marked
the atomic cooling threshold. The MCG duty fraction function was thus given by

.. math::

    f_{\rm duty}^{\rm (MCG)}(M_h;M_{\rm turn}^{\rm (MCG)},M_{\rm atom}) = \exp\left(-M_{\rm turn}^{\rm (MCG)}/M_h\right)\exp\left(-M_h/M_{\rm atom}\right).

``21cmFAST`` v3.0.0 introduced several flags and parameters that allowed the user to switch
between different models:

* ``USE_MASS_DEPNDENT_ZETA``:
  If True, the ionization efficiency (:math:`\zeta`) is allowed to
  depend on the halo mass, as in v2.0.0, otherwise, the ionization efficiency is constant for
  all halo masses, as in v1.0.0. Note that this flag also controlled the duty fraction function,
  as the mass-dependent ionization efficiency model had an exponential cutoff, while the
  mass-independent model had a sharp cutoff.

* ``USE_MINI_HALOS``:
  If True, the user can include molecular cooling galaxies (MCGs) in the
  simulation, which are assumed to reside inside "mini-halos". If False, only atomic cooling galaxies
  (ACGs) are included, as in v2.0.0 and v1.0.0. Note that this flag could have been turned on only if
  ``USE_MASS_DEPNDENT_ZETA`` was also True.

* ``M_MIN_in_Mass``:
  If True, the ACG turnover mass is given by the user-defined parameter ``M_TURN``. If False, the ACG
  turnover mass is given by the atomic cooling threshold, :math:`M_{\rm atom}(z)`, whereas
  the virial temperature is given by the new parameters ``ION_Tvir_MIN`` and ``X_RAY_Tvir_MIN`` (see below).
  Note that this logic however was only applied if ``USE_MASS_DEPNDENT_ZETA`` was False (and therefore
  ``USE_MINI_HALOS`` was False as well). The idea behind this parameterization was that it can capture both
  the atomic cooling threshold and the effect of stellar feedback on the ACG turnover mass.

* ``M_TURN``:
  This parameter was the user-defined ACG turnover mass, which was used only if
  ``M_MIN_in_Mass`` was True. Otherwise, this parameter was ignored.

* ``ION_Tvir_MIN``:
  The minimum virial temperature for halos that can host ionizing sources.

* ``X_RAY_Tvir_MIN``:
  The minimum virial temperature for halos that can host X-ray sources.

In addition, v3.0.0 introduced the generation of a realization of the :math:`v_{\rm cb}` field,
though it was not used in the code to modify the MCG turnover mass until v3.1.0 (see below).

Turnover mass logic in ``21cmFAST`` v3.0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The logic for determining the turnover masses in ``21cmFAST`` v3.0.0 was pretty complicated
since it was different for different flag combinations.

* If ``USE_MINI_HALOS`` is True:
    * The ACG turnover mass was set to be the maximum between (1) :math:`M_{\rm atom}(z)` and
      (2) the reionization feedback mass :math:`M_{\rm reion}(z)` (for the latter, see Sobacchi
      and Mesinger 2013, https://arxiv.org/pdf/1301.6776). The reionization feedback however
      was applied only in the reionization module, whereas it was ignored in the spin temperature
      module, see https://github.com/21cmfast/21cmFAST/issues/470 for more details.
    * The MCG turnover mass was set to be the maximum between (1) :math:`M_{\rm mol}(z;J_{\rm LW})`,
      the molecular cooling threshold that involved Lyman-Werner feedback (see Qin et al. 2020,
      https://arxiv.org/pdf/2003.04442, for more details) and (2) the reionization feedback mass
      :math:`M_{\rm reion}(z)`. Just like for ACGs, the reionization feedback was applied only in
      the reionization module, whereas it was ignored in the spin temperature module.
* Otherwise:
    * If ``USE_MASS_DEPNDENT_ZETA`` is True, the ACG turnover mass was set to be the user-defined
      parameter ``M_TURN``.
    * Otherwise:
        * If ``M_MIN_in_Mass`` is True, the ACG turnover mass was set to be the user-defined parameter
          ``M_TURN``.
        * Otherwise, the ACG turnover mass was set to be :math:`M_{\rm atom}(z)`, where the minimum virial
          temperature was given by the user-defined parameter ``ION_Tvir_MIN`` (``X_RAY_Tvir_MIN``)
          for ionizing (X-ray) sources.

Four important properties to notice from the above logic are:

1. If ``USE_MINI_HALOS`` was True, the free parameter ``M_TURN`` was ignored. This was a bug that
   was later fixed in v4.0.0.

2. The ACG turnover mass was different, depending on the value of ``USE_MINI_HALOS``. In other words,
   the ACG turnover mass depended on the presence or absence of MCGs in the simulation. This was later
   changed in v4.3.0.

3. If ``USE_MINI_HALOS`` was True, the MCG turnover mass was always **inhomogeneous**, as well as
   the ACG turnover mass (at least in the reionization module where the reionization feedback was applied).
   Otherwise, if ``USE_MINI_HALOS`` was False, the ACG turnover mass was always **homogeneous** (i.e.,
   the same for all cells in the simulation box). This was later changed in v4.3.0.

4. The MCG **upper** turnover mass was set to be :math:`M_{\rm atom}(z)` and was therefore not the same
   as the ACG **lower** turnover mass. This was a bug that was later fixed in v4.3.0.

``21cmFAST`` v3.1.0
-------------------

In ``21cmFAST`` v3.1.0 (Munoz et al. 2021, https://arxiv.org/pdf/2110.13919), the effect of
the relative velocity between (cold) dark matter and baryons during kinematic decoupling,
:math:`v_{\rm cb}`, was included in the code, for the case of ``USE_MINI_HALOS`` being True.
The effect of :math:`v_{\rm cb}` was to increase the MCG turnover mass, :math:`M_{\rm turn}^{\rm (MCG)}`,
in regions of high relative velocity, which suppressed the formation of MCGs in those regions.
In the reionization module, the MCG turnover mass was now set to be the maximum between
(1) :math:`M_{\rm mol}(z, J_{\rm LW}, v_{\rm cb})`, (2) :math:`M_{\rm reion}(z)`, and
(3) :math:`M_{\rm mol}(z, J_{\rm LW}=0, v_{\rm cb}^{\rm const})`, where ``v_{\rm cb}^{\rm const}``
is a constant value that was determined from the new parameters ``FIX_VCB_AVG`` and ``FIXED_VAVG``
(see below). In the spin temperature module, the MCG turnover mass was set to
:math:`M_{\rm mol}(z, J_{\rm LW}, v_{\rm cb})`. The ACG turnover mass remained the same as in v3.0.0.

New parameters and flags were introduced in v3.1.0 to control the effect of :math:`v_{\rm cb}`
on the MCG turnover mass:

* ``USE_RELATIVE_VELOCITIES``:
  If True, the effect of the **inhomogeneous** :math:`v_{\rm cb}`
  on the MCG turnover mass was included, as well as suppressing the matter density power
  spectrum on small scales (see Eq. 14 in https://arxiv.org/pdf/2110.13919). The effect of the
  **inhomogeneous** :math:`v_{\rm cb}` on the turnover mass was possible however only if ``FIX_VCB_AVG``
  was set to False.

* ``FIX_VCB_AVG``:
  If True, the effect of the **homogeneous** :math:`v_{\rm cb}` on the MCG turnover
  mass was included, with a value set by the user-defined parameter ``FIXED_VAVG``. If False, the
  local value of :math:`v_{\rm cb}` was either fluctuating if ``USE_RELATIVE_VELOCITIES`` was True, or
  set to zero if ``USE_RELATIVE_VELOCITIES`` was False.

* ``FIXED_VAVG``:
  The value of the **homogeneous** :math:`v_{\rm cb}` in units of km/s, which was used
  only if ``FIX_VCB_AVG`` was True and if ``USE_RELATIVE_VELOCITIES`` was False.

Note that having ``USE_RELATIVE_VELOCITIES`` on True required the user to work with ``POWER_SPECTRUM``
set to ``"CLASS"``, since the generation of a realization of a fluctuating :math:`v_{\rm cb}` field
required to have the :math:`v_{\rm cb}` transfer function from ``CLASS``. This transfer function however
was a fixed table that corresponded to a single set of cosmological parameters, that was later
changed in v4.0.0.

``21cmFAST`` v4.0.0
-------------------

In ``21cmFAST`` v4.0.0 (Davies et al. 2025, https://arxiv.org/pdf/2504.17254), a new source
model with stochastic discrete halos was introduced. The source model that was used in the
simulation was controlled by a new parameter called ``SOURCE_MODEL``. This parameter had several options:

* ``"CONST-ION-EFF"``:
  This option was equivalent to setting ``USE_MASS_DEPNDENT_ZETA`` on False
  in previous versions, namely the ionization efficiency was constant for all halo masses, the ACG
  turnover mass was set according to the ``M_MIN_in_Mass`` flag, and the ACG duty fraction function
  was a sharp cutoff at the ACG turnover mass.

* ``"E-INTEGRAL"``:
  This option was equivalent to setting ``USE_MASS_DEPNDENT_ZETA`` on True in
  previous versions, namely the ionization efficiency was mass-dependent, the duty fraction functions
  were smooth exponential cutoffs, and the turnover masses were set according to the logic of v3.1.0
  (with one change, see below). This option was named like that, since the emissivity fields were
  computed on the Eulerian density grid, as in previous versions of ``21cmFAST``.

* ``"L-INTEGRAL"``:
  This is a new source model that was introduced in v4.0.0, where the emissivity
  fields were computed on the Lagrangian density grid, and then mapped to the Eulerian grid. The
  ionization efficiency was mass-dependent, the duty fraction functions were smooth exponential cutoffs,
  and a new logic for the turnover masses was introduced (see below). No discrete halos were used in
  this source model, unlike the source models described below.

* ``"DEXM-ESF"``:
  This is a new source model that was introduced in v4.0.0, where the emissivity
  fields were computed on the Lagrangian density grid, but using resolved discrete halos that were
  found via the excursion-set formalism (ESF), as was implemented by the ``DEXM`` algorithm (see
  Mesinger & Furlanetto 2007, https://arxiv.org/pdf/0704.0946). All the astrophysical properties,
  including the ionization efficiency, duty fraction functions, and turnover masses, were the same
  as in the ``"L-INTEGRAL"`` source model, although they were now also computed individually
  for each discrete halo.

* ``"CHMF-SAMPLER"``:
  This is a new source model that was introduced in v4.0.0, where now unresolved
  discrete halos were sampled from the conditional halo mass function (CHMF) in each cell of the
  simulation box at the lowest redshift, whereas at higher redshifts progenitor halos were sampled
  from their bigger descendent halos. All the astrophysical properties, including the ionization
  efficiency, duty fraction functions, and turnover masses, were the same as in the
  ``"DEXM-ESF"`` source model.

Furthermore, the transfer functions from ``CLASS`` were now computed on the fly (when
``POWER_SPECTRUM`` was set to ``"CLASS"``), consistently with the user's chosen values for
the cosmological parameters.

Turnover mass logic in ``21cmFAST`` v4.0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only two out of the five source models in v4.0.0 were supported in v3.1.0, these
are ``"CONST-ION-EFF"`` and ``"E-INTEGRAL"`` source models. For the ``"CONST-ION-EFF"`` source model (analogous to v3.1.0 with ``USE_MASS_DEPNDENT_ZETA``
set to False), the logic for the ACG turnover mass has remained exactly the same as in v3.1.0. However, for the ``"E-INTEGRAL"`` source model (analogous to v3.1.0 with ``USE_MASS_DEPNDENT_ZETA``
set to True), a subtle but important change has been made. Now, the stellar feedback mass,
which is given by the user-defined parameter ``M_TURN``, was taken into account in the logic
for the turnover masses when ``USE_MINI_HALOS`` was True: The new logic for the turnover masses
in this case was similar to the logic in v3.1.0, except that in v4.0.0 the turnover masses were
set to be ``M_TURN`` if they were the largest mass scale (an exception however occurred in
the reionization module for determining the MCG turnover mass, where ``M_TURN`` was still ignored).
As before, if ``USE_MINI_HALOS`` was False, the ACG turnover mass was set to be ``M_TURN``.

As for the new source models that were introduced in v4.0.0, ``"L-INTEGRAL"``, ``"DEXM-ESF"``,
and ``"CHMF-SAMPLER"``, the logic for the turnover masses was as follows:

* If ``USE_MINI_HALOS`` is True:
    * The ACG turnover mass was set to be the maximum between (1) :math:`M_{\rm atom}(z)`,
      (2) :math:`M_{\rm reion}(z)`, and (3) ``M_TURN``.
    * The MCG turnover mass was set to be the maximum between (1) :math:`M_{\rm mol}(z, J_{\rm LW}, v_{\rm cb})`,
      (2) :math:`M_{\rm reion}(z)`, and (3) ``M_TURN``.
* Otherwise, the ACG turnover mass was set to be ``M_TURN``.

``21cmFAST`` v4.3.0
-------------------

In ``21cmFAST`` v4.3.0, the logic for the turnover masses was simplified and unified across most
source models. This became possible by introducing two new parameters, ``V_CB_MODEL`` and
``REIONIZATION_FEEDBACK_MODEL``. In addition, for better clarity, the free astrophysical
parameter ``M_TURN`` was renamed to ``M_TURN_STELLAR_FEEDBACK``, making it clear that this
parameter was meant to capture only the effect of stellar feedback on the turnover masses.
``M_TURN`` therefore became deprecated in v4.3.0.

``V_CB_MODEL``
~~~~~~~~~~~~~~~

The options for the new parameter ``V_CB_MODEL`` were:

* ``"NONE"``:
  The :math:`v_{\rm cb}` effect on the MCG turnover mass was ignored, setting it to zero.
  This was equivalent to setting both ``USE_RELATIVE_VELOCITIES`` and ``FIX_VCB_AVG`` on False in v4.0.0.

* ``"AVG-AUTO"``:
  This was a new feature in v4.3.0, compared to v4.0.0, where the homogeneous
  :math:`v_{\rm cb}` effect on the MCG turnover mass was accounted by treating the :math:`v_{\rm cb}` field
  as a constant, and the mean value of :math:`v_{\rm cb}` was automatically computed from the output of
  the ``CLASS`` simulation, when ``POWER_SPECTRUM`` was set to ``"CLASS"`` (otherwise, a default
  value of 27 km/s was used for setting the homogeneous :math:`v_{\rm cb}` value).

* ``"FLUCTS"``:
  The inhomogeneous :math:`v_{\rm cb}` effect on the MCG turnover mass was accounted.
  This was equivalent to setting ``USE_RELATIVE_VELOCITIES`` on True and ``FIX_VCB_AVG`` on False in v4.0.0.

* ``"AVG-DEBUG"``:
  The homogeneous :math:`v_{\rm cb}` effect on the MCG turnover mass was accounted,
  with a value set by the user-defined parameter ``V_CB_AVG_DEBUG``. This was equivalent to
  setting ``FIX_VCB_AVG`` on True in v4.0.0.

In addition, the suppression in the matter density power spectrum on small scales due to
the :math:`v_{\rm cb}` effect was applied as long as ``V_CB_MODEL`` was not set to ``NONE``.
``V_CB_MODEL`` thus replaced the flags ``USE_RELATIVE_VELOCITIES`` and ``FIX_VCB_AVG`` in v4.0.0,
whereas the parameter ``V_CB_AVG_DEBUG`` replaced the parameter ``FIXED_VAVG`` in v4.0.0.
``USE_RELATIVE_VELOCITIES``, ``FIX_VCB_AVG`` and ``FIXED_VAVG`` therefore became deprecated in v4.3.0.

``REIONIZATION_FEEDBACK_MODEL``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The options for the new parameter ``REIONIZATION_FEEDBACK_MODEL`` were:

* ``"NONE"``:
  The reionization feedback effect was ignored in both ACGs and MCGs.

* ``"ACG"``:
  The reionization feedback effect was accounted only in ACGs.

* ``"MCG"``:
  The reionization feedback effect was accounted only in MCGs (relevant
  only when ``USE_MINI_HALOS`` was set to True).

* ``"BOTH"``:
  The reionization feedback effect was accounted in both ACGs and MCGs (if the latters
  were present, according to the value of ``USE_MINI_HALOS``).

``REIONIZATION_FEEDBACK_MODEL`` allowed then to apply the inhomogeneous reionization feedback effect
on the ACG turnover mass, regardless if MCGs were present or not in the simulation. This decoupled
the application of the reionization feedback from the presence of MCGs in the simulation,
which was still the logic in v4.0.0.

Turnover mass logic in ``21cmFAST`` v4.3.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The turnover mass logic in v4.3.0 was greatly simplified and unified across almost all source models.
For all source models (except for ``"CONST-ION-EFF"``):

* The ACG turnover mass was set to be the maximum between (1) :math:`M_{\rm atom}(z)`,
  (2) :math:`M_{\rm reion}(z)`, and (3) ``M_TURN_STELLAR_FEEDBACK`` (the reionization feedback was accounted
  only if ``REIONIZATION_FEEDBACK_MODEL`` was set to either ``"ACG"`` or ``"BOTH"``).
* The MCG turnover mass was set to be the maximum between (1) :math:`M_{\rm mol, J_{\rm LW}, v_{\rm cb}}(z)`,
  (2) :math:`M_{\rm reion}(z)`, and (3) ``M_TURN_STELLAR_FEEDBACK`` (the reionization feedback was
  accounted only if ``REIONIZATION_FEEDBACK_MODEL`` was set to either ``"MCG"`` or ``"BOTH"``).
* The ACG turnover mass was the same, whether or not MCGs were present in the simulation (i.e.,
  regardless of the value of ``USE_MINI_HALOS``).

An exception in the above logic still occurred when ``SOURCE_MODEL`` was set to ``"E-INTEGRAL"``: in
the spin temperature module, the reionization feedback was still ignored in both ACGs and MCGs,
see more details in https://github.com/21cmfast/21cmFAST/issues/470.

When ``SOURCE_MODEL`` was set to ``"CONST-ION-EFF"``, the logic for the ACG turnover mass remained
the same as in v4.0.0 (and as in v3.0.0), namely:

* If ``M_MIN_in_Mass`` is True, the ACG turnover mass was set to be ``M_TURN_STELLAR_FEEDBACK``.
* Otherwise, the ACG turnover mass was set to be :math:`M_{\rm atom}(z)`, where the minimum virial
  temperature was given by the user-defined parameter ``ION_Tvir_MIN`` (``X_RAY_Tvir_MIN``)
  for ionizing (X-ray) sources.

Duty fraction function logic in ``21cmFAST`` v4.3.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ``SOURCE_MODEL`` set to ``"CONST-ION-EFF"``, the duty fraction function remained as a sharp
cutoff at the ACG turnover mass, as in previous versions of ``21cmFAST``, namely:

.. math::

    f_{\rm duty}(M_h;M_{\rm turn}) = H(M_h-M_{\rm turn}),\qquad\text{for SOURCE_MODEL = "CONST-ION-EFF"}.

For all other four source models, the duty fraction function for ACGs and MCGs remained almost the
same as in v4.0.0 (or v3.0.0):

.. math::

    f_{\rm duty}^{\rm (ACG)}(M_h;M_{\rm turn}) = \exp\left(-M_{\rm turn}^{\rm (ACG)}/M_h\right).

.. math::

    f_{\rm duty}^{\rm (MCG)}(M_h;M_{\rm turn}^{\rm (MCG)},M_{\rm atom}) = \exp\left(-M_{\rm turn}^{\rm (MCG)}/M_h\right)\exp\left(-M_h/M_{\rm turn}^{\rm (ACG)}\right)H(M_{\rm turn}^{\rm (ACG)} - M_{\rm turn}^{\rm (MCG)}).

Two subtle differences were made in the MCG duty fraction function compared to previous versions:

1. MCGs can exist only if :math:`M_{\rm turn}^{\rm (MCG)} < M_{\rm turn}^{\rm (ACG)}`, which is enforced
   by the Heaviside step function. This change was made in order to reflect the physical property that
   galaxies are formed via atomic cooling once their mass exceeds the atomic cooling threshold. It
   also means that when the reionization feedback becomes the dominant mechanism in determining the
   two turnover masses, only ACGs survive, as the MCG duty fraction function becomes zero.

2. The MCG **upper** turnover mass was defined to be the ACG **lower** turnover mass. This change
   was made to ensure that there is no mass gap between MCGs and ACGs where galaxies cannot form.
