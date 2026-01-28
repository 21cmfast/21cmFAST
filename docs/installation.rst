============
Installation
============

.. attention:: Due to ``classy`` not being available on ``conda-forge``, we do not
          support installation via ``conda`` from v4 (it was support for v3.x.x).
          Please use ``pip`` or ``uv`` installation as described below.


Non-Python Dependencies
-----------------------
``21cmFAST`` relies on some C libraries which must be installed on your system
prior to installation.

The first thing you will need is a C compiler. We support ``gcc`` (with
``-fopenmp`` support) and ``clang`` (with ``-fopenmp`` support via ``libomp``).

The C libraries required are:

* ``gsl``
* ``fftw`` (compiled with floating-point enabled, and ``--enable-shared``)

As it turns out, though these are fairly common libraries, getting them installed in a
way that ``21cmFAST`` understands on various operating systems can be slightly non-trivial.

See the `FAQ <faqs/installation_faq>`_ for more detailed questions on installation.
If you are on MacOS and are having trouble with installation (or would like to share
a successful installation strategy!) please see
`this discussion <https://github.com/21cmfast/21cmFAST/discussions/208>`_.


.. tab-set::

    .. tab-item:: Conda on any OS

        If you are using ``conda``, the easiest way to get the dependencies is to
        install them via ``conda`` itself. You can do this with::

            conda install -c conda-forge gsl fftw

        Note that if you are using ``conda`` to install these packages, you may
        need to point to the relevant ``lib/`` and ``include/`` folders for both
        ``gsl`` and ``fftw`` during installation of ``21cmFAST`` (see below).

    .. tab-item:: Linux

        Most linux distros come with packages for the requirements, and also ``gcc`` by
        default. As long as these packages install into the standard location,
        a standard installation of ``21cmFAST`` will be automatically possible (see below).

        E.g. on Arch-based distros::

            sudo pacman -S fftw gsl

        On Debian-based distros::

            sudo apt-get install libfftw3-dev libgsl-dev

    .. tab-item:: MacOS

        The easiest way to get the dependencies (other than ``conda``) is via
        ``homebrew``. You can install them with::

            brew install gsl fftw

    .. tab-item:: HPC

        These libraries will often be available on a HPC environment by using
        ``module load gsl fftw3`` (or similar) commands.

        Note also that while ``fftw`` may be available to load, it may not have the correct
        compilation options (i.e. float-enabled and multiprocessing-enabled). In that
        case, you may need to install your own local copy of ``fftw`` (and possibly
        ``gsl`` too) from source.


Setting correct env variables for installation
----------------------------------------------

It is not guaranteed that the libraries you just installed will be able to be found
by the compiler when installing ``21cmFAST``. If they are installed to a place not on the
``LD_LIBRARY``/``INCLUDE`` paths, then you must use compilation options to specify where they are.
The following options will be set as options at the front of the installation command
(see below).

To specify the C compiler, use ``CC`` (e.g. ``CC=gcc`` or ``CC=clang``). Usually this
will not be necessary sinc the default compiler should work. However, if you'd like to
use ``clang`` on linux, for example, you may need to set ``CC`` explicitly.

To add a library search path, use ``<LIBRARY>_LIB`` where ``<LIBRARY>`` is either
``GSL`` or ``FFTW``. For example, if ``gsl`` is installed in ``/opt/local/lib``, then
you would add ``GSL_LIB=/opt/local/lib`` to the installation command.

If you don't know where the library is installed, you can try running::

    locate libgsl.so
    locate libfftw3.so

to find the libs, or::

    locate gsl.h
    locate fftw3.h

to find the include files.

.. note:: On MacOS (at least, 10.14-10.15), you may need to also add the following
          installation flag (see `here <https://github.com/21cmfast/21cmFAST/discussions/208>`_)::

              CFLAGS="-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX<input version>.sdk"



Other Compile Options
---------------------

Similar to the ``CC``, ``*_LIB`` and ``*_INC`` options above, there are some other
options that can be set at compile time.

Debug Mode
~~~~~~~~~~
To compile in debug mode (which enables tools like ``valgrind`` and ``gdb`` to give
useful output), set the ``DEBUG`` environment variable to ``True`` (or ``1``).


Logging in C-Code
~~~~~~~~~~~~~~~~~
By default, the C-code will only print to stderr when it encounters warnings or
critical errors. However, there exist several levels of logging output that can be
switched on, but only at compilation time. To enable these, use the following
compilation option::

    LOG_LEVEL=<log_level>

The ``<log_level>`` can be any non-negative integer, or one of the following
(case-insensitive) identifiers::

    NONE, ERROR, WARNING, INFO, DEBUG, SUPER_DEBUG, ULTRA_DEBUG

If an integer is passed, it corresponds to the above levels in order (starting
from zero). Be careful if the level is set to 0 (or NONE), as useful error
and warning messages will not be printed. By default, the log level is 2 (or
WARNING), unless the DEBUG=1 environment variable is set, in which case the
default is 4 (or DEBUG). Using very high levels (eg. ULTRA_DEBUG) can print out
*a lot* of information and make the run time much longer, but may be useful
in some specific cases.


Installation Instructions
-------------------------

After following the above instructions, you are ready to install ``21cmFAST``.
How you install ``21cmFAST`` depends on whether you are a user or a developer, and
which tool you'd like to use to install it.

First, make an isolated virtual environment and activate it (note the python
version should be the latest compatible version):


.. tab-set::

    .. tab-item:: conda (rec. for MacOS)
        :sync: conda

        .. code-block:: bash

            conda create -n 21cmfast_env python=3.13 numpy scipy click pyyaml cffi astropy h5py matplotlib attrs
            conda activate 21cmfast_env

    .. tab-item:: uv (rec. for linux)
        :sync: uv

        .. code-block:: bash

            uv venv --python 3.13
            source activate .venv/bin/activate

    .. tab-item:: uv (new project)
        :sync: uvp

        .. code-block:: bash

            uv init --name project-name --python=3.13  # see other uv options
            cd project-name

    .. tab-item:: venv

        .. code-block:: bash

            python3 -m venv 21cmfast_env
            source 21cmfast_env/bin/activate


Then install the latest stable release from PyPI with (see above for compile options):

.. tab-set::

    .. tab-item:: conda/pip
        :sync: conda

        .. code-block:: bash

            [COMPILE OPTIONS] pip install 21cmFAST

    .. tab-item:: uv (rec. for linux)
        :sync: uv

        .. code-block:: bash

            [COMPILE OPTIONS] uv pip install 21cmfast

    .. tab-item:: uv project
        :sync: uvp

        .. code-block:: bash

            [COMPILE OPTIONS] uv add 21cmfast
