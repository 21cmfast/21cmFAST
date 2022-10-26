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

Global Parameters
-----------------
There are a bunch of "global" parameters that are used throughout the C code. These are
parameters that are deemed to be constant enough to not expose them through the
regularly-used input structs, but nevertheless may necessitate modification from
time-to-time. These are accessed through the ``global_params`` object::

    >>> from py21cmfast import global_params

Help on the attributes can be obtained via ``help(global_params)`` or
`in the docs <../reference/_autosummary/py21cmfast.inputs.html>`_. Setting the
attributes (which affects them everywhere throughout the code) is as simple as, eg::

    >>> global_params.Z_HEAT_MAX = 30.0

If you wish to use a certain parameter for a fixed portion of your code (eg. for a single
run), it is encouraged to use the context manager, eg.::

    >>> with global_params.use(Z_HEAT_MAX=10):
    >>>    run_lightcone(...)

How can I read a Coeval object from disk?
-----------------------------------------

The simplest way to read a :class:`py21cmfast.outputs.Coeval` object that has been
written to disk is by doing::

    import py21cmfast as p21c
    coeval = p21c.Coeval.read("my_coeval.h5")

However, you may want to read parts of the data, or read the data using a different
language or environment. You can do this as long as you have the HDF5 library (i.e.
h5py for Python). HDF5 is self-documenting, so you should be able to determine the
structure of the file yourself interactively. But here is an example using h5py::

    import h5py

    fl = h5py.File("my_coeval.h5", "r")

    # print a dict of all the UserParams
    # the CosmoParams, FlagOptions and AstroParams are accessed the same way.
    print(dict(fl['user_params'].attrs))

    # print a dict of all globals used for the coeval
    print(dict(fl['_globals'].attrs))

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
