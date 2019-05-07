=====
Usage
=====

On this page we provide a broad overview of the capabilities of ``21CMMC``, its philosophy, and how to use it.
For more details, see either the tutorials or the :doc:`API Reference <reference/index>`, which itself contains several
examples.

Overall Design
==============
``21CMMC`` has two main goals: firstly to provide a convenient and Pythonic wrapper around the 21cmFAST C library, and
secondly to provide a robust MCMC framework for constraining the parameters of 21cmFAST on data (whether real or mock).
With regards to the first of these goals, ``21CMMC`` is intended to be an almost *total* drop-in replacement of the
underlying library, providing for all its functionality with the ease-of-use of a Python frontend. Thus, familiar
CLI programs are reproduced (eg. ``./init`` has been replaced with ``21CMMC init``), and the various parameters and options
contained in the original source headers are available (with defaults) within Python classes.

Before moving to specific usage, it is useful to consider some broader design choices of ``21CMMC``. Firstly,
installation of ``21CMMC`` dynamically compiles the C code and installs it to the standard library location. This being
so, many of the original parameters, which were written as ``#define`` constants, have been changed to dynamic variables
so that re-compilation is unnecessary, and they can be modified within a Python session. Nevertheless, several constants
which were deemed to be almost always static have been retained as ``#define`` statements, and to change them requires
hand-modifying them and re-installing. All of these constants can be found in the ``Constants.h`` header.

Secondly, when ``21CMMC`` is installed, it automatically creates a configuration directory in the user's home:
``~/.21CMMC``. This houses a number of important configuration options; usually default values of parameters.
At this stage, the location of this directory is not itself configurable. The config directory contains example configuration
files for the CLI interface (see below), which can also be copied anywhere on disk and modified. Importantly, the
``config.yml`` file in this directory specifies some of the more important options for how ``21CMMC`` behaves by default.
One such option is ``boxdir``, which specifies the directory in which ``21CMMC`` will cache results (see below for details).
Finally, the config directory houses several data tables which are used to accelerate several calculations. In principle
these files are over-writeable, but they should only be touched if one knows very well what they are doing.

Thirdly, ``21CMMC`` contains a more robust cataloguing/caching method than the underlying 21cmFAST. Instead of saving
data with a selection of the dependent parameters written into the filename -- a method which is prone to error if a
parameter which is not part of that selection is modified -- ``21CMMC`` writes all data into a configurable central
directory with a hash filename unique to *all* parameters upon which the data depends. Each kind of dataset has
attached methods which efficiently search this central directory for matching data to be read in when necessary.
Several arguments are available for all library functions which produce such datasets that control this output. In this
way, the data that is being retrieved is always reliably produced with the desired parameters, and users need not
concern themselves with how and where the data is saved -- it can be retrieved merely by creating an empty object with
the desired parameters and calling ``.read()``, or even better, by calling the function to *produce* the given dataset,
which will by default just read it in if available.

It should not be of great concern to the user, but in the case that it is, the datasets themselves are stored as HDF5
files, and the various parameters defining them are stored as attributes.

Two main methods of using ``21CMMC`` are supported: a CLI and Python library.

CLI
===
The CLI interface always starts with the command ``21CMMC``, and has a number of subcommands. To list the available
subcommands, use::

    $ 21CMMC --help

To get help on any subcommand, simply use::

    $ 21CMMC <subcommand> --help

Any subcommand which runs some aspect of 21cmFAST will have a ``--config`` option, which specifies a configuration
file (by default this is ``~/.21CMMC/runconfig_example.yml``). This config file specifies the parameters of the run.
Furthermore, any particular parameter that can be specified in the config file can be alternatively specified on the
command line by appending the command with the parameter name, eg.::

    $ 21CMMC init --config=my_config.yml --HII_DIM=40 hlittle=0.7 --DIM 100 SIGMA_8 0.9

The above command shows the numerous ways in which these parameters can be specified (with or without leading dashes,
and with or without "=").

The CLI interface, while simple to use, does have the limitation that more complex arguments than can be passed to the
library functions are disallowed. For example, one cannot pass a previously calculated initial conditions box to the
``perturb`` command. However, if such a box has been calculated with the default option to write it to disk, then it
will automatically be found and used in such a situation, i.e. the following will not re-calculate the init box::

    $ 21CMMC init
    $ 21CMMC perturb redshift=8.0

This means that almost all of the functionality provided in the library is accessible via the CLI.


Library
=======
.. note:: See the coeval cube tutorial for a more in-depth introduction to using the ``21CMMC`` library.

Typically the user will want to use ``21CMMC`` as a library -- calling underlying C routines, and obtaining nicely
wrapped results that are ready for further analysis/plotting. The main namespace is ``py21cmmc``::

    >>> import py21cmmc as p21

To see details of the API for each class/function in the library, see the :doc:`API Reference <reference/index>`. The
two primary functions that should be used are ``run_coeval`` and ``run_lightcone``::

    >>> init, perturb, xHI, Tb = p21.run_coeval(
    >>>                              redshift=[7,8,9],
    >>>                              cosmo_params=p21.CosmoParams(hlittle=0.7),
    >>>                              user_params=p21.UserParams(HII_DIM=100)
    >>>                          )

Or::

    >>> lightcone = p21.run_lightcone(redshift=z2, max_redshift=z2, z_step_factor=1.03)

The parameters of the run are typically defined in appropriate classes: ``p21.CosmoParams``, ``p21.UserParams``,
``p21.AstroParams`` and ``p21.FlagOptions``. A further class, ``p21.global_params`` is provided to set/get a number
of global parameters which interface with the C code. These are set globally.

The outputs of the above functions are various objects, within which are defined a number of data cubes. The relevant
quantities in each object can be queried with, for example, ``dir(init)``. Many of them contain views of the underlying
data which make it simpler to deal with (eg. reshaping an underlying flattened 1D array into the standard 3D cube).

.. warning:: This page needs a huge amount of updating.

Catching Segfaults
==================
Since 21cmFAST is still written in C, there is the off-chance that something catastrophic will happen, causing a
segfault. Typically, if this happens, Python will not print a traceback where the error occurred, and finding the
source of such errors can be difficult. However, one has the option of using the standard library
`faulthandler <https://docs.python.org/3/library/faulthandler.html>`_. Specifying ``-X faulthandler`` when invoking
Python will cause a minimal traceback to be printed to ``stderr`` if a segfault occurs.


