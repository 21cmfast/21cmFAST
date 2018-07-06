========
Overview
========

.. start-badges

.. end-badges

An extensible MCMC framework for 21cmFAST.

* Free software: MIT license

Installation
============

First, you'll need to have the required C libraries: ``gsl``, ``fftw`` (make sure you install the floating-point version!)
``openmp`` and ``gslcblas``.

Then just do (from top-level directory)::

    pip install -e .

Various options exist to manage compilation via environment variables. Basically, any variable with "INC" in its name
will add to the includes directories, while any variable with "lib" in its name will add to the directories searched
for libraries. To change the C compiler, use ``CC``. Finally, if you want to compile the C-library in dev mode (so you
can do stuff like valgrid and gdb with it), install with DEBUG=True. So for example::

    CC=/usr/bin/gcc DEBUG=True GSL_LIB=/opt/local/lib FFTW_INC=/usr/local/include pip install -e .

While the ``-e`` option will keep your library up-to-date with any (Python) changes, this will *not* work when changing
the C extension. If the C code changes, you need to manually run ``rm -rf build/*`` then re-install as above.

Quick Usage
===========

We support two methods of using ``21CMMC``:

CLI
~~~
The CLI interface always starts with the command ``21CMMC``, and has a number of subcommands. To list the available
subcommands, use::

    $ 21CMMC --help

To get help on any subcommand, simply use::

    $ 21CMMC <subcommand> --help

.. note:: The only subcommands implemented so far (for testing) are ``init`` and ``perturb``.

Library
~~~~~~~
Typically the user will want to use ``21CMMC`` as a library -- calling underlying C routines, and obtaining nicely
wrapped results that are ready for further analysis/plotting. The main namespace is ``py21cmmc``::

    >>> from py21cmmc import initial_conditions, ...


Documentation
=============

To view the docs, install the ``requirements_dev.txt`` packages, go to the docs/ folder, and type "make html", then
open the ``index.html`` file in the ``_build/html`` directory.

.. warning:: This is coming soon...
