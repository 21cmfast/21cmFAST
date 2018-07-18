============
Installation
============

First, you'll need to have the required C libraries: ``gsl``, ``fftw`` (make sure you install the floating-point version!)
``openmp`` and ``gslcblas``.

Then, at the command line::

    pip install 21CMMC

If developing, from the top-level directory do::

    pip install -e .

Various options exist to manage compilation via environment variables. Basically, any variable with "INC" in its name
will add to the includes directories, while any variable with "lib" in its name will add to the directories searched
for libraries. To change the C compiler, use ``CC``. Finally, if you want to compile the C-library in dev mode (so you
can do stuff like valgrid and gdb with it), install with DEBUG=True. So for example::

    CC=/usr/bin/gcc DEBUG=True GSL_LIB=/opt/local/lib FFTW_INC=/usr/local/include pip install -e .

In addition, the ``BOXDIR`` variable specifies the *default* directory that any data
produced by 21CMMC will be cached. This value can be updated at any time by changing it in the ``$CFGDIR/config.yml``
file, and can be overwritten on a per-call basis.

While the ``-e`` option will keep your library up-to-date with any (Python) changes, this will *not* work when changing
the C extension. If the C code changes, you need to manually run ``rm -rf build/*`` then re-install as above.

