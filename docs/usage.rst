=====
Usage
=====

On this page we provide a broad overview of the capabilities of ``21cmFAST``, its
philosophy, and how to use it. For more details, see either the
:doc:`tutorials <tutorials>` or the :doc:`API Reference <reference/index>`,
which itself contains several examples.

Design of the "New" 21cmFAST
============================
``21cmFAST`` as of v1.0.0+ has been modified significantly, operating under a python
 wrapper which enables more dynamic handling of outputs. While the previous command-line
 usage has been preserved (though modified), there is now also the option to directly
 call ``21cmFAST`` within a Python session, and receive outputs as more native Python
 objects which are easier to work with.

Now, installation of ``21cmFAST`` dynamically compiles the C code and
installs it to the standard library location. This being so, many of the original
parameters, which were written as ``#define`` constants, have been changed to dynamic
variables so that re-compilation is unnecessary, and they can be modified within a
Python session. Nevertheless, several constants which were deemed to be almost always
static have been retained as ``#define`` statements, and to change them requires
hand-modifying them and re-installing. All of these constants can be found in the
``Constants.h`` header.

When ``21cmFAST`` is installed, it automatically creates a configuration directory in
the user's home: ``~/.21cmfast``. This houses a number of important configuration
options; usually default values of parameters. At this stage, the location of this
directory is not itself configurable. The config directory contains example
configuration files for the CLI interface (see below), which can also be copied anywhere
on disk and modified. Importantly, the ``config.yml`` file in this directory specifies
some of the more important options for how ``21CMMC`` behaves by default.
One such option is ``boxdir``, which specifies the directory in which ``21cmFAST`` will
cache results (see below for details). Finally, the config directory houses several data
tables which are used to accelerate several calculations. In principle
these files are over-writeable, but they should only be touched if one knows very well
what they are doing.

Finally, ``21cmFAST`` contains a more robust cataloguing/caching method. Instead of
saving data with a selection of the dependent parameters written into the filename --
a method which is prone to error if a parameter which is not part of that selection is
modified -- ``21CMMC`` writes all data into a configurable central directory with a hash
filename unique to *all* parameters upon which the data depends. Each kind of dataset has
attached methods which efficiently search this central directory for matching data to be
read in when necessary.
Several arguments are available for all library functions which produce such datasets
that control this output. In this way, the data that is being retrieved is always
reliably produced with the desired parameters, and users need not concern themselves
with how and where the data is saved -- it can be retrieved merely by creating an empty
object with the desired parameters and calling ``.read()``, or even better, by calling
the function to *produce* the given dataset, which will by default just read it in if
available.

It should not be of great concern to the user, but in the case that it is, the datasets
themselves are stored as HDF5 files, and the various parameters defining them are stored
as attributes.

CLI
===
The CLI interface always starts with the command ``21cmfast``, and has a number of
subcommands. To list the available subcommands, use::

    $ 21cmfast --help

To get help on any subcommand, simply use::

    $ 21cmfast <subcommand> --help

Any subcommand which runs some aspect of ``21cmFAST`` will have a ``--config`` option,
which specifies a configuration file (by default this is
``~/.21cmfast/runconfig_example.yml``). This config file specifies the parameters of the
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


Library
=======
.. note:: See the coeval cube tutorial for a more in-depth introduction to using the
          ``21cmfast`` library.

Typically the user will want to use ``21cmfast`` as a library -- calling underlying C
routines, and obtaining nicely wrapped results that are ready for further
analysis/plotting. The main namespace is ``py21cmfast``::

    >>> import py21cmfast as p21

To see details of the API for each class/function in the library, see the
:doc:`API Reference <reference/index>`. The two primary functions that should be used
are ``run_coeval`` and ``run_lightcone``::

    >>> init, perturb, xHI, Tb = p21.run_coeval(
    >>>                              redshift=[7,8,9],
    >>>                              cosmo_params=p21.CosmoParams(hlittle=0.7),
    >>>                              user_params=p21.UserParams(HII_DIM=100)
    >>>                          )

Or::

    >>> lightcone = p21.run_lightcone(redshift=z2, max_redshift=z2, z_step_factor=1.03)

The parameters of the run are typically defined in appropriate classes:
``p21.CosmoParams``, ``p21.UserParams``, ``p21.AstroParams`` and ``p21.FlagOptions``.
A further class, ``p21.global_params`` is provided to set/get a number
of global parameters which interface with the C code. These are set globally.

The outputs of the above functions are various objects, within which are defined a
number of data cubes. The relevant quantities in each object can be queried with, for
example, ``dir(init)``. Many of them contain views of the underlying data which make it
simpler to deal with (eg. reshaping an underlying flattened 1D array into the standard
3D cube).



