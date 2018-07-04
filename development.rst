Development
===========

The layout of the package
~~~~~~~~~~~~~~~~~~~~~~~~~
I should try to explain how I've gone about modifying this to its current state, from Brad's original version.
This version is installable, and the Makefile stuff happens in ``setup.py``. The CLI commands live in cli.py.
Funnily enough, plotting functions live in ``plotting.py``,
and at the moment, each of them takes as its main argument a ``Lightcone`` or ``CoEval`` object, which is basically what is
passed back from the C driver. The ``_utils.py`` module contains a couple of functions for writing out the parameter
files which can be read in by the C driver. I have not made use of these in the rest of the code, however.

The actual Python wrappers of the C, at its basic level, are found in ``wrapper.py``. All the C code lives in the ``_21cmfast``
folder and is compiled by ``setup.py`` from here (this required changing some of the includes in the C files).

The wrapping is done with CFFI, rather than the native ctypes. This allows for less redundant specification of types
etc. The things to watch out for, when using CFFI, is the memory management. If an array is created in Python, and a
pointer to it is set to a C variable, then that Python variable has to stick around otherwise the memory is effectively
free'd, and weird stuff happens. This is usually obvious, but is sometimes obscured when setting a C variable to the
result of a function call, for which no Python variable has ever been specified (and so it quickly gets garbage collected).

The building of the C code is done in ``build_cffi.py``. At the moment, it's a bit rough, due to the number of global
defines that are used. However, the overall structure is such that ``set_source`` literally just includes the main
source code that needs to be there to run. The ``cdef`` defines the signatures of all global parameters and functions
which ought to be wrapped. This *should* be as easy as including a header file, but #defines only get captured if you
specify them manually as static const, and furthermore, there *is* no header file which contains the main functions we
care about. So they are copied in at this point.

As for input parameters to the functions, I've used a series of Structure classes (I've subclassed each of them to give
defaults for each parameter, so the user doesn't have to worry about most of them). How these work should hopefully be
reasonably clear from the code.


Meta-development stuff
~~~~~~~~~~~~~~~~~~~~~~
I'm using a git-flow git system, where we can create features and fixes etc. If you don't like that, feel free to change
it or discuss it. I think we should use the Github issue system to handle all of our "todo's" and then we can each pick
them off easily, and comment on their viability.

To run the all tests run (no tests as yet...)::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox