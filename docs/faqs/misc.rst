Miscellaneous FAQs
==================

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
``21cmFAST`` has a configuration file located at ``~/.21cmfast/config.yml``. This file
specifies some options to use in a rather global sense, especially to do with I/O.
You can directly edit this file to change how ``21cmFAST`` behaves for you across
sessions.
For any particular function call, any of the options may be overwritten by supplying
arguments to the function itself.
To set the configuration for a particular session, you can also set the global ``config``
instance, for example::

    >>> import py21cmfast as p21
    >>> p21.config['regenerate'] = True
    >>> p21.run_lightcone(...)

All functions that use the ``regenerate`` keyword will now use the value you've set in the
config. Sometimes, you may want to be a little more careful -- perhaps you want to change
the configuration for a set of calls, but have it change back to the defaults after that.
We provide a context manager to do this::

    >>> with p21.config.use(regenerate=True):
    >>>     p21.run_lightcone()
    >>>     print(p21.config['regenerate'])  # prints "True"
    >>> print(p21.config['regenerate'])  # prints "False"

To make the current configuration permanent, simply use the ``write`` method::

    >>> p21.config['direc'] = 'my_own_cache'
    >>> p21.config.write()

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

    # print a dict of all the UserParams
    # the CosmoParams, FlagOptions and AstroParams are accessed the same way.
    print(dict(fl['user_params'].attrs))

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
-------------------------------------------
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
