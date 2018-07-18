=====
Usage
=====

We support two methods of using ``21CMMC``:

CLI
===
The CLI interface always starts with the command ``21CMMC``, and has a number of subcommands. To list the available
subcommands, use::

    $ 21CMMC --help

To get help on any subcommand, simply use::

    $ 21CMMC <subcommand> --help

.. note:: The only subcommands implemented so far (for testing) are ``init`` and ``perturb``.

Library
=======
Typically the user will want to use ``21CMMC`` as a library -- calling underlying C routines, and obtaining nicely
wrapped results that are ready for further analysis/plotting. The main namespace is ``py21cmmc``::

    >>> from py21cmmc import initial_conditions, ...

To see details of the API for each class/function in the library, see the :doc:`API Reference <reference/index>`.

.. warning:: This page needs a huge amount of updating.

