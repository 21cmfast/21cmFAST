============
Installation
============

The easiest way to install ``21cmFAST`` is to use ``conda``. Simply use
``conda install -c conda-forge 21cmFAST``. With this method, all dependencies are taken
care of, and it should work on either Linux or MacOS. If for some reason this is not
possible for you, read on.

Dependencies
------------
We try to have as many of the dependencies automatically installed as possible.
However, since ``21cmFAST`` relies on some C libraries, this is not always possible.

The C libraries required are:

* ``gsl``
* ``fftw`` (compiled with floating-point enabled, and ``--enable-shared``)
* ``openmp``
* A C-compiler with compatibility with the ``-fopenmp`` flag. **Note:** it seems that on
  OSX, if using ``gcc``, you will need ``v4.9.4+``.

As it turns out, though these are fairly common libraries, getting them installed in a
way that ``21cmFAST`` understands on various operating systems can be slightly non-trivial.

HPC
~~~
These libraries will often be available on a HPC environment by using the
``module load gsl`` and similar commands. Note that just because they are loaded
doesn't mean that ``21cmFAST`` will be able to find them. You may have to point to the
relevant ``lib/`` and ``include/`` folders for both ``gsl`` and ``fftw`` (these should
be available using ``module show gsl`` etc.)

Note also that while ``fftw`` may be available to load, it may not have the correct
compilation options (i.e. float-enabled and multiprocessing-enabled). In this case,
see below.

Linux
~~~~~
Most linux distros come with packages for the requirements, and also ``gcc`` by default,
which supports ``-fopenmp``. As long as these packages install into the standard location,
a standard installation of ``21cmFAST`` will be automatically possible (see below).
If they are installed to a place not on the ``LD_LIBRARY``/``INCLUDE`` paths, then you
must use the compilation options (see below) to specify where they are.
For example, you can check if the header file for ``fftw3`` is
in its default location ``/usr/include/`` by running::

    cd /usr/include/
    find fftw3.h

or::

    locate fftw3.h

.. note:: there exists the option of installing ``gsl``, ``fftw`` and ``gcc`` using ``conda``.
          This is discussed below in the context of MacOSX, where it is often the
          easiest way to get the dependencies, but it is equally applicable to linux.

Ubuntu
^^^^^^
If you are installing 21cmFAST just as a user, the very simplest method is ``conda``
-- with this method you simply need ``conda install -c conda-forge 21cmFAST``, and all
dependencies will be automatically installed. However, if you are going to use
``pip`` to install the package directly from the repository, there is
a [bug in pip](https://stackoverflow.com/questions/71340058/conda-does-not-look-for-libpthread-and-libpthread-nonshared-at-the-right-place-w)
that means it cannot find conda-installed shared libraries properly. In that case, it is much
easier to install the basic dependencies (``gcc``, ``gsl`` and ``fftw3``) with your
system's package manager. ``gcc`` is by default available in Ubuntu.
To check if ``gcc`` is installed, run ``gcc --version`` in your terminal.
Install ``fftw3`` and ``gsl`` on your system with  ``sudo apt-get install libfftw3-dev libgsl-dev``.


In your ``21cmfast`` environment, now install the ``21cmFAST`` package using::

    cd /path/to/21cmFAST/
    pip install .

If there is an issue during installation, add ``DEBUG=all`` or ``--DEBUG`` which may provide additional
information.

.. note:: If there is an error during compilation that the ``fftw3`` library cannot be found,
          check where the ``fftw3`` library is actually located using ``locate libfftw3.so``.
          For example, it may be located in ``/usr/lib/x86_64-linux-gnu/``. Then, provide this path
          to the installation command with the ``LIB`` flag. For more details see the note in the
          MacOSX section below.

.. note:: You may choose to install ``gsl`` as an anaconda package as well, however, in that case,
          you need to add both ``INC`` paths in the installation command e.g.:
          ``GSL_INC=/path/to/conda/env/include FFTW_INC=/usr/include``

MacOSX
~~~~~~
On MacOSX, obtaining ``gsl`` and ``fftw`` is typically more difficult, and in addition,
the newer native ``clang`` does not offer ``-fopenmp`` support.

For ``conda`` users (which we recommend using), the easiest way to get ``gsl`` and ``fftw``
is by doing ``conda install -c conda-forge gsl fftw`` in your environment.

.. note:: if you use ``conda`` to install ``gsl`` and ``fftw``, then you will need to point at
          their location when installing `21cmFAST` (see compiler options below for details).
          In this case, the installation command should simply be *prepended* with::

              LIB=/path/to/conda/env/lib INC=/path/to/conda/env/include

To get ``gcc``, either use ``homebrew``, or again, ``conda``: ``conda install -c anaconda gcc``.
If you get the ``conda`` version, you still need to install the headers::

    xcode-select --install

On older versions then you need to do::

    open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_<input version>.pkg

.. note:: some versions of MacOS will also require you to point to the correct gcc
          compiler using the ``CC`` environment variable. Overall, the point is to NOT
          use ``clang``. If ``gcc --version`` shows that it is actually GCC, then you
          can set ``CC=gcc``. If you use homebrew to install ``gcc``, it is likely that
          you'll have to set ``CC=gcc-11``.

For newer versions, you may need to prepend the following command to your ``pip install`` command
when installing ``21cmFAST`` (see later instructions)::

    CFLAGS="-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX<input version>.sdk"

See `<faqs/installation_faq>`_ for more detailed questions on installation.
If you are on MacOSX and are having trouble with installation (or would like to share
a successful installation strategy!) please see the
`open issue <https://github.com/21cmfast/21cmFAST/issues/84>`_.

With the dependencies installed, follow the instructions below,
depending on whether you are a user or a developer.

For Users
---------

.. note:: ``conda`` users may want to pre-install the following packages before running
          the below installation commands::

            conda install numpy scipy click pyyaml cffi astropy h5py


Then, at the command line::

    pip install git+https://github.com/21cmFAST/21cmFAST.git

If developing, from the top-level directory do::

    pip install -e .

Note the compile options discussed below!

For Developers
--------------
If you are developing ``21cmFAST``, we highly recommend using ``conda`` to manage your
environment, and setting up an isolated environment. If this is the case, setting up
a full environment (with all testing and documentation dependencies) should be as easy
as (from top-level dir)::

    conda env create -f environment_dev.yml

Otherwise, if you are using ``pip``::

    pip install -e .[dev]

The ``[dev]`` "extra" here installs all development dependencies. You can instead use
``[tests]`` if you only want dependencies for testing, or ``[docs]`` to be able to
compile the documentation.

Compile Options
---------------
Various options exist to manage compilation via environment variables. Basically,
any variable with "INC" in its name will add to the includes directories, while
any variable with "lib" in its name will add to the directories searched for
libraries. To change the C compiler, use ``CC``. Finally, if you want to compile
the C-library in dev mode (so you can do stuff like valgrid and gdb with it),
install with DEBUG=True. So for example::

    CC=/usr/bin/gcc DEBUG=True GSL_LIB=/opt/local/lib FFTW_INC=/usr/local/include pip install -e .

.. note:: For MacOS a typical installation command will look like
          ``CC=gcc CFLAGS="-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX<input version>.sdk" pip install .``
          (using either ``gcc`` or ``gcc-11`` depending on how you installed gcc), with
          other compile options possible as well.

In addition, the ``BOXDIR`` variable specifies the *default* directory that any
data produced by 21cmFAST will be cached. This value can be updated at any time by
changing it in the ``$CFGDIR/config.yml`` file, and can be overwritten on a
per-call basis.

While the ``-e`` option will keep your library up-to-date with any (Python)
changes, this will *not* work when changing the C extension. If the C code
changes, you need to manually run ``rm -rf build/*`` then re-install as above.

Logging in C-Code
~~~~~~~~~~~~~~~~~
By default, the C-code will only print to stderr when it encounters warnings or
critical errors. However, there exist several levels of logging output that can be
switched on, but only at compilation time. To enable these, use the following::

    LOG_LEVEL=<log_level> pip install -e .

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
