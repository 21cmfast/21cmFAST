"""Low-level wrapping utilities and functions for the core C-code.

The main wrapper for the underlying 21cmFAST C-code.

The module provides both low- and high-level wrappers, using the very low-level machinery
in :mod:`~py21cmfast._utils`, and the convenient input and output structures from
:mod:`~py21cmfast.inputs` and :mod:`~py21cmfast.outputs`.

This module provides a number of:

* Low-level functions which simplify calling the background C functions which populate
  these output objects given the input classes.
* High-level functions which provide the most efficient and simplest way to generate the
  most commonly desired outputs.

**Low-level functions**

The low-level functions provided here ease the production of the aforementioned output
objects. Functions exist for each low-level C routine, which have been decoupled as far
as possible. So, functions exist to create :func:`initial_conditions`,
:func:`perturb_field`, :class:`ionize_box` and so on. Creating a brightness temperature
box (often the desired final output) would generally require calling each of these in
turn, as each depends on the result of a previous function. Nevertheless, each function
has the capability of generating the required previous outputs on-the-fly, so one can
instantly call :func:`ionize_box` and get a self-consistent result. Doing so, while
convenient, is sometimes not *efficient*, especially when using inhomogeneous
recombinations or the spin temperature field, which intrinsically require consistent
evolution of the ionization field through redshift. In these cases, for best efficiency
it is recommended to either use a customised manual approach to calling these low-level
functions, or to call a higher-level function which optimizes this process.

Finally, note that :mod:`py21cmfast` attempts to optimize the production of the large amount of
data via on-disk caching. By default, if a previous set of data has been computed using
the current input parameters, it will be read-in from a caching repository and returned
directly. This behaviour can be tuned in any of the low-level (or high-level) functions
by setting the `write`, `direc`, `regenerate` and `match_seed` parameters (see docs for
:func:`initial_conditions` for details). The function :func:`~query_cache` can be used
to search the cache, and return empty datasets corresponding to each (and these can then be
filled with the data merely by calling ``.read()`` on any data set). Conversely, a
specific data set can be read and returned as a proper output object by calling the
:func:`~py21cmfast.cache_tools.readbox` function.


**High-level functions**

As previously mentioned, calling the low-level functions in some cases is non-optimal,
especially when full evolution of the field is required, and thus iteration through a
series of redshift. In addition, while :class:`InitialConditions` and
:class:`PerturbedField` are necessary intermediate data, it is *usually* the resulting
brightness temperature which is of most interest, and it is easier to not have to worry
about the intermediate steps explicitly. For these typical use-cases, two high-level
functions are available: :func:`run_coeval` and :func:`run_lightcone`, whose purpose
should be self-explanatory. These will optimally run all necessary intermediate
steps (using cached results by default if possible) and return all datasets of interest.


Examples
--------
A typical example of using this module would be the following.

>>> import py21cmfast as p21

Get coeval cubes at redshift 7,8 and 9, without spin temperature or inhomogeneous
recombinations:

>>> coeval = p21.run_coeval(
>>>     redshift=[7,8,9],
>>>     cosmo_params=p21.CosmoParams(hlittle=0.7),
>>>     matter_params=p21.MatterParams(HII_DIM=100)
>>> )

Get coeval cubes at the same redshift, with both spin temperature and inhomogeneous
recombinations, pulled from the natural evolution of the fields:

>>> all_boxes = p21.run_coeval(
>>>                 redshift=[7,8,9],
>>>                 matter_params=p21.MatterParams(HII_DIM=100),
>>>                 astro_flags=p21.AstroFlags(INHOMO_RECO=True),
>>>                 do_spin_temp=True
>>>             )

Get a self-consistent lightcone defined between z1 and z2 (`z_step_factor` changes the
logarithmic steps between redshift that are actually evaluated, which are then
interpolated onto the lightcone cells):

>>> lightcone = p21.run_lightcone(redshift=z2, max_redshift=z2, z_step_factor=1.03)
"""
