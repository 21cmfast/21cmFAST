======================================
Design Philosophy and Features for v3+
======================================

Here we describe in broad terms the design philosophy of the *new* ``21cmFAST``,
and some of its new features.
This is useful to get an initial bearing of how to go about using ``21cmFAST``, though
most likely the :doc:`tutorials <tutorials>` will be better for that.
It is also useful for those who have used the "old" ``21cmFAST`` (versions 2.1 and less)
and want to know why they should use this new version (and how to convert).
In doing so, we'll go over some of the key features of ``21cmFAST`` v3+.
To get a more in-depth view of all the options and features available, look at the
very thorough :doc:`API Reference <reference/index>`.


Design Philosophy
=================
The goal of v3 of ``21cmFAST`` is to provide the same computational efficiency and
scientific features of the previous generations, but packaged in a form that adopts the
best modern programming standards, including:

* simple installation
* comprehensive documentation
* comprehensive test suite
* more modular code
* standardised code formatting
* truly open-source and collaborative design (via Github)

Partly to enable these standards, and partly due to the many *extra* benefits it brings,
v3 also has the major distinction of being wrapped entirely in Python. The *extra*
benefits brought by this include:

* a native python library interface (eg. get your output box directly as a ``numpy`` array).
* better file-writing, into the HDF5 format, which saves metadata along with the box data.
* a caching system so that the same data never has to be calculated twice.
* reproducibility: know which exact version of ``21cmFAST``, with what parameters, produced a given dataset.
* significantly improved warnings and error checking/propagation.
* simplicity for adding new additional effects, and inserting them in the calculation pipeline.

We hope that additional features and benefits will be found by the growing community
of ``21cmFAST`` developers and users.

How it Works
============
v3 is *not* a complete rewrite of ``21cmFAST``. Most of the C-code of previous versions
is kept, though it has been modularised and modified in many places. The fundamental
routines are the same (barring bugfixes!).

The major programs of the original version (``init``, ``perturb``, ``ionize`` etc.) have
been converted into modular *functions* in C. Furthermore, most of the global parameters
(and, more often than not, global ``#define`` options) have been modularised and converted
into a series of input "parameter" ``structs``. These get passed into the functions.
Furthermore, each C function, instead of writing a bunch of files, returns an output
``struct`` containing all the stuff it computed.

Each of these functions and structs are wrapped in Python using the ``cffi`` package.
CFFI compiles the C code once upon *installation*. Due to the fact that parameters are
now passed around to the different functions, rather than being global defines, we no
longer need to re-compile every time an option is changed. Python itself can handle
changing the parameters, and can use the outputs in whatever way the user desires.

To maintain continuity with previous versions, a CLI interface is provided (see below)
that acts in a similar fashion to previous versions.

High-level configuration of ``21cmFAST`` can be set using the ``py21cmfast.config``
object. It is essentially a dictionary with its key/values the parameters. To make any
changes in the object permanent, use the ``py21cmfast.config.write()`` method.
One global configuration option is ``direc``, which specifies the directory in which
``21cmFAST`` will cache results by default (this can be overriden directly in any
function, see below for details).

Finally, ``21cmFAST`` contains a more robust cataloguing/caching method. Instead of
saving data with a selection of the dependent parameters written into the filename --
a method which is prone to error if a parameter which is not part of that selection is
modified -- ``21cmFAST`` writes all data into a configurable central directory with a hash
filename unique to *all* parameters upon which the data depends. Each kind of dataset has
attached methods which efficiently search this central directory for matching data to be
read when necessary.
Several arguments are available for all library functions which produce such datasets
that control this output. In this way, the data that is being retrieved is always
reliably produced with the desired parameters, and users need not concern themselves
with how and where the data is saved -- it can be retrieved merely by creating an empty
object with the desired parameters and calling ``.read()``, or even better, by calling
the function to *produce* the given dataset, which will by default just read it in if
available.

CLI
===
The CLI interface always starts with the command ``21cmfast``, and has a number of
subcommands. To list the available subcommands, use::

    $ 21cmfast --help

To get help on any subcommand, simply use::

    $ 21cmfast <subcommand> --help

Any subcommand which runs some aspect of ``21cmFAST`` will have a ``--config`` option,
which specifies a configuration file. This config file specifies the parameters of the
run. Furthermore, any particular parameter that can be specified in the config file can
be alternatively specified on the command line by appending the command with the
parameter name, eg.::

    $ 21cmfast init --config=my_config.yml --HII_DIM=40 hlittle=0.7 --DIM 100 SIGMA_8 0.9

The above command shows the numerous ways in which these parameters can be specified
(with or without leading dashes, and with or without "=").

The CLI interface, while simple to use, does have the limitation that more complex
arguments than can be passed to the library functions are disallowed. For example,
one cannot pass a previously calculated initial conditions box to the ``perturb``
command. However, if such a box has been calculated with the default option to write it
to disk, then it will automatically be found and used in such a situation, i.e. the
following will not re-calculate the init box::

    $ 21cmfast init
    $ 21cmfast perturb redshift=8.0

This means that almost all of the functionality provided in the library is accessible
via the CLI.
