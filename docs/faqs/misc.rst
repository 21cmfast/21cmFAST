Miscellaneous FAQs
==================

My global signal doesn't make sense, what's wrong?
--------------------------------------------------
If you plot the 21-cm global signal as a function of redshift, and it doesn't look as expected, it is very likely due to setting ``USE_TS_FLUCT=False``.

Historically, ``21cmFAST`` was initially designed to simulate very fast reionization maps of the Universe. At sufficiently low redshifts, due to strong Lyman-alpha coupling and X-ray heating, the spin temperature of the IGM exceeds the CMB temperature to such an extent that the brightness temperature does not depend on the spin temperature. This is known as the "saturation limit". For that reason, in order to speed up the calculations, the default value of ``USE_TS_FLUCT`` is ``False``, namely ``21cmFAST`` assumes by default that the saturation limit holds at all redshifts and therefore does not bother to calculate the spin temperature. At sufficiently high redshifts however the saturation limit breaks, which might explain why your 21-cm global signal increases with redshift and doesn't exhibit any absorption features. Try to set ``USE_TS_FLUCT=True`` to recover the true global signal when the saturation limit is relaxed (note that the runtime is expected to increase when ``USE_TS_FLUCT=True``).

In general, before running a full lightcone simulation via ``run_lightcone``, it's a good practice to make a quick calculation with ``run_global_evolution`` (find more information and limitations of this feature in `the global evolution tutorial <../tutorials/global_evolution.ipynb>`_).


My run seg-faulted, what should I do?
-------------------------------------
Since ``21cmFAST`` is written in C, there is the off-chance that something
catastrophic will happen, causing a segfault. Typically, if this happens, Python will
not print a traceback where the error occurred, and finding the source of such errors
can be difficult. However, one has the option of using the standard library
`faulthandler <https://docs.python.org/3/library/faulthandler.html>`_. Specifying
``-X faulthandler`` when invoking Python will cause a minimal traceback to be printed
to ``stderr`` if a segfault occurs.

Configuring 21cmFAST
--------------------
``21cmFAST`` has a few global configuration options that affect many calculations.
To set the configuration for a particular session, you can set the global ``config``
instance, for example::

    >>> import py21cmfast as p21
    >>> p21.config['ignore_R_BUBBLE_MAX_error'] = True
    >>> p21.run_lightcone(...)

Sometimes, you may want to be a little more careful -- perhaps you want to change
the configuration for a set of calls, but have it change back to the defaults after that.
We provide a context manager to do this::

    >>> with p21.config.use(ignore_R_BUBBLE_MAX_error=True):
    >>>     p21.run_lightcone()
    >>>     print(p21.config['ignore_R_BUBBLE_MAX_error'])  # prints "True"
    >>> print(p21.config['ignore_R_BUBBLE_MAX_error'])  # prints "False"

To make the current configuration permanent, simply use the ``write`` method::

    >>> p21.config['direc'] = 'my_own_cache'
    >>> p21.config.write("config.yaml")

How can I read a Coeval object from disk?
-----------------------------------------

The simplest way to read a :class:`py21cmfast.outputs.Coeval` object that has been
written to disk is by doing::

    import py21cmfast as p21c
    coeval = p21c.Coeval.from_file("my_coeval.h5")

However, you may want to read parts of the data, or read the data using a different
language or environment. You can do this as long as you have the HDF5 library (i.e.
h5py for Python). HDF5 is self-documenting, so you should be able to determine the
structure of the file yourself interactively. But here is an example using h5py::

    import h5py

    fl = h5py.File("my_coeval.h5", "r")

    # print a dict of all the MatterParams
    # the CosmoParams, AstroFlags and AstroParams are accessed the same way.
    print(dict(fl['matter_params'].attrs))

    # Get the redshift and random seed of the coeval box
    redshift = fl.attrs['redshift']
    seed = fl.attrs['random_seed']

    # Get the Initial Conditions:
    print(np.max(fl['InitialConditions']['hires_density'][:]))

    # Or brightness temperature
    print(np.max(fl['BrightnessTemp']['brightness_temperature'][:]))

    # Basically, the different stages of computation are groups in the file, and all
    # their consituent boxes are datasets in that group.
    # Print out the keys of the group to see what is available:
    print(fl['TsBox'].keys())

How can I read a LightCone object from file?
--------------------------------------------
Just like the :class:`py21cmfast.outputs.Coeval` object documented above, the
:class:`py21cmfast.outputs.LightCone` object is most easily read via its ``.read()`` method.
Similarly, it is written using HDF5. Again, the input parameters are stored in their
own sub-objects. However, the lightcone boxes themselves are in the "lightcones" group,
while the globally averaged quantities are in the ``global_quantities`` group::

    import h5py
    import matplotlib.pyplot as plt

    fl = h5py.File("my_lightcone.h5", "r")

    Tb = fl['lightcones']['brightness_temp'][:]
    assert Tb.ndim==3

    global_Tb = fl['global_quantities']['brightness_temp'][:]
    redshifts = fl['node_redshifts']

    plt.plot(redshifts, global_Tb)

Can I instantiate my own OutputStruct objects?
----------------------------------------------
Usually, you create instances of an :class:`py21cmfast.wrapper.outputs.OutputStruct`
object by running either :func:`py21cmfast.run_coeval` or some lower-level function,
like :func:`py21cmfast.compute_initial_conditions`. However, it's possible you want to
switch out a simulation step from ``21cmFAST`` and insert your own, but then go on using
that box in further ``21cmFAST`` simulation components. The way to do this is as follows,
using the ``InitialConditions`` as an example::

    ics = p21c.InitialConditions.new(inputs=p21c.InputParameters())
    ics.set('lowres_density', my_computed_value)

You would use this ``.set()`` method on each of the fields you needed to set. Now this
data should be properly shared with the backend C-code, and the object can be used
in subsequent steps within ``21cmFAST``.
