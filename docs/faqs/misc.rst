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
To set the configuration for a particular session, you can also set the global `config`
instance, for example::

    >>> import py21cmfast as p21
    >>> p21.config['regenerate'] = True
    >>> p21.run_lightcone(...)

All functions that use the `regenerate` keyword will now use the value you've set in the
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
