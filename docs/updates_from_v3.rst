=====================
Changes from v3 to v4
=====================

This page outlines the main differences between the usage of ``21cmFAST`` versions 3 and
1. Use this documentation as a reference for understanding and adapting to the updates in
``21cmFAST`` version 4.

Stochastic Halo Sampling
========================
The main addition the ``21cmFAST`` in version 4 is the stochastic halo sampler. This
samples conditional halo mass functions instead of integrating over them, producing a
discrete source field which is then used in the spin temperature and ionization field
calculations. This not only includes the effects of stochasticity in the IGM observables,
but also creates several new outputs which can be further used in forecasting galaxy
survey, line intensity mapping, and cosmic background statistics. The sampler is
activated with the flag ``HALO_STOCHASTICITY`` and serves as a faster replacement to the
previous excursion-set halo finder, with greatly increased functionality. Halos are
sampled in a backward time-loop in each run before the main IGM calculations start.
Halo catlogues can be found in the :class:`HaloCatalog` (Initial Lagrangian) and
:class:`PerturbedHaloCatalog` (Final Eulerian) classes. Each catalogue contains the
coordinates and masses of each halo, as well as the correlated RNG used to determine
their galaxy properties. Converting from the RNG to the properties can be done with
:func:`~py21cmfast.wrapper.cfuncs.convert_halo_properties`. Galaxy properties are not
directly stored in these objects for efficiency and so we can correctly account for
feedback in the forward time-loop.

The conditional mass functions used to perform integrals have been extended with the
Sheth-Tormen CHMF (Sheth+2002) which has been applied to ``21cmFAST`` in both halo and
grid based source models, when the user sets ``HMF=='ST'``. All other mass functions
rescale the Extended Press-Schechter (EPS) conditional mass function.

Input Parameters
================
The most significant changes for a user will be how input parameters are passed into
simulation in ``21cmFASTv4``. In version 3, the user specified four input parameter
structures (``CosmoParams``, ``UserParams``, ``AstroParams``, ``FlagOptions``) which
defined the simulation. In version 4 the user will mostly deal with a single structure,
``InputParameters``, while many of the underlying parameters are unchanged, there
are now simpler ways to construct parameter sets, and consistency between your inputs is
more explicily enforced. See :ref:`tutorials/defining_input_params` for in-depth
advice on how to specify input parameters in v4. Key differences include:

* We allow for the creation of TOML templates to easily save parameter sets, and provide
  several native templates for common modes. Call
  :func:`~py21cmfast._templates.list_templates` for a full list.
* We no longer fix parameter sets behind the scenes to make them consistent, it is now
  up to the user to ensure their parameters pass our consistency checks.
* While the parameters are still separated by type in the backend (mostly for caching
  purposes), all parameters are passed as one object to each of our functions.

The most common operating mode will simply load from a set of built-in templates:

.. code-block:: python

    inputs = py21cmfast.InputParameters.from_template(
        ['latest', 'small'],
        random_seed=1234,
        SIGMA_8=0.9
    )

where the list of template names specifies a base set of parameters, which can be
explicitly over-ridden (e.g. by setting ``SIGMA_8`` explicitly). A coeval run would
then simply be performed by calling :func:`~py21cmfast.drivers.coeval.run_coeval` as in
v3:

.. code-block:: python

    output = py21cmfast.run_coeval(inputs=inputs, out_redshifts=(6,7,8,9))

replacing the four parameter structures with the single ``inputs`` keyword. The same
keyword is provided to :func:`~py21cmfast.drivers.lightcone.run_lightcone` to specify the
inputs.

More details on running coeval cubes and lightcones can be found in the tutorials
:ref:`tutorials/coeval_cubes` and :ref:`tutorials/lightcones`.

The :class:`~py21cmfast.InputParameters` class contains all the fields needed to specify
a run, including:

* ``random_seed``: Always required. Defines the random seed for drawing initial
  conditions and all other RNG.
* ``node_redshifts``: specifies the snapshots of the simulation. Since many quantities
  depend on evolution, changing the ``node_redshifts`` will alter the simulation.

For efficient caching and use in MCMC, the parameter fields are divided into five
subgroups. Users should not need to interact with these often if they are using
templates, but they are briefly described below:

* ``CosmoParams``: same as v3, defines the cosmology used in the simulation.
* ``SimulationOptions``: Most of the previous ``UserParams`` were moved here. this
  defines general options such as box size and grid dimensions.
* ``MatterOptions``: These are options which enable different modules and algorithms
  which affect the matter fields (ICs, density, halos), such as which halo mass function
  to use.
* ``AstroParams``: same as v3 with some additions. Physical parameters such as escape
  fraciton, which affect the approximate RT (IonizationBox etc.)
* ``AstroOptions``: previously ``FlagOptions``, these enable different modules which
  *only* affect the spin temperature and ionisation fields.


Caching
=======
Significant improvements were made to the caching mechanism for ``21cmFAST`` outputs.
We provide the class ``OutputCache`` for searching, reading and writing low-level
outputs to file, and ``RunCache`` for specifying all the outputs needed to complete a
run with given inputs. We also provide ``CacheConfig`` for a user to specify exactly
which output strucutres are cached. high-level functions such as ``run_coeval`` and
``run_lightcone`` accept the ``cache`` keyword argument to specify an ``OutputCache`` to
write data to, and the ``write`` keyword argument to specify a ``CacheConfig``, or
simple boolean if one wants to turn on or off caching entirely. For examples on using
the cache, see :ref:`tutorials/caching`.

Backwards compatibility
-----------------------
Unfortunately, we do not currently provide functions for loading output files from v3
into our structures, since the parameter structures have changed too much. If a user
wishes to access their old data, they may still do so using ``h5py`` or any HDF5 reader.
We do not recommend using these in further computation using ``21cmFASTv4``.

The file structure for a coeval in v3 resembled the following::

  - File
    - OutputStruct_1
      - Array Fields...
    - OutputStruct_2
      - Array Fields
    -...
    - cosmo_params
      - attrs
        - Param fields...
    - user_params
      - ...
    - astro_params
    - flag_options
    - _globals

In v4 we use the following::

  - File
    - OutputStruct_1
      - OutputFields
        - Array Fields...
      - InputParameters
        - cosmo_params
        - simulation_options
        - matter_options
        - astro_params
        - astro_options
    - ...

Single Field Function Names
===========================
Functions dealing with the generation of single fields have changed somewhat. A full
list is provided below in the order which they are called in ``run_lightcone``:

* :func:`~py21cmfast.drivers.single_field.compute_initial_conditions`
* :func:`~py21cmfast.drivers.single_field.perturb_field`
* :func:`~py21cmfast.drivers.single_field.determine_halo_catalog`
* :func:`~py21cmfast.drivers.single_field.perturb_halo_catalog`
* :func:`~py21cmfast.drivers.single_field.compute_halo_grid`
* :func:`~py21cmfast.drivers.single_field.compute_xray_source_field`
* :func:`~py21cmfast.drivers.single_field.compute_spin_temperature`
* :func:`~py21cmfast.drivers.single_field.compute_ionization_field`
* :func:`~py21cmfast.drivers.single_field.brightness_temperature`

Output Field Names
==================
Similar to the function names, some output fields have also been renamed for clarity:

* Fields in TsBox

  * ``x_e_box`` ➟  ``xray_ionised_fraction``
  * ``Tk_box`` ➟ ``kinetic_temp_neutral``
  * ``J_21_LW_box`` ➟ ``J_21_LW``
  * ``Ts_box`` ➟ ``spin_temperature``

* Fields in IonizedBox

  * ``xH_box`` ➟ ``neutral_fraction``
  * ``Gamma12_box`` ➟ ``ionisation_rate_G12``
  * ``MFP_box`` ➟ ``mean_free_path``
  * ``z_re_box`` ➟ ``z_reion``
  * ``dNrec_box`` ➟ ``cumulative_recombinations``
  * ``temp_kinetic_all_gas`` ➟ ``kinetic_temperature``
  * ``Fcoll`` ➟ ``unnormalised_nion``
  * ``Fcoll_MINI`` ➟ ``unnormalised_nion_mini``
