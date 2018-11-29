====================
Notes For Developers
====================

If you are new to developing 21CMMC, please read the :ref:`contributing:Contributing` section first, which outlines the general
concepts for contributing to development. This page lists some more detailed notes which may be helpful through the
development process.

Note that when developing, we recommend using an isolated python environment, and installing all requirements as follows::

$ pip install -r requirements_dev.txt

Compiling for debugging
-----------------------
When developing, it is usually a good idea to compile the underlying C code in DEBUG mode. This may allow extra print
statements in the C, but also will allow running the C under ``valgrind`` or ``gdb``. To do this::

$ DEBUG=True pip install -e .

See :ref:`installation:Installation` for more installation options.

Running with Valgrind
---------------------
If any changes to the C code are made, it is ideal to run tests under valgrind, and check for memory leaks. To do this,
install ``valgrind`` (we have tested v3.14+), which is probably available via your package manager. We provide a
suppression file for ``valgrind`` in the ``devel/`` directory of the main repository.

It is ideal if you install a development-version of python especially for running these tests. To do this, download
the version of python you want and then configure/install with::

$ ./configure --prefix=<your-home>/<directory> --without-pymalloc --with-pydebug --with-valgrind
$ make ; make install

Construct a ``virtualenv`` on top of this installation, and create your environment, and install all requirements.

If you do not wish to run with a modified version of python, you may continue with your usual version, but may get some
extra cruft in the output. If running with Python version > 3.6, consider running with environment variable
``PYTHONMALLOC=malloc`` (see https://stackoverflow.com/questions/20112989/how-to-use-valgrind-with-python ).

The general pattern for using valgrind with python is::

$ valgrind --tool=memcheck --track-origins=yes --leak-check=full --suppressions=devel/valgrind-suppress-all-but-c.supp <python script>

One useful command is to run valgrind over the test suite (from the top-level repo directory)::

$ valgrind --tool=memcheck --track-origins=yes --leak-check=full --suppressions=devel/valgrind-suppress-all-but-c.supp pytest

While we will attempt to keep the suppression file updated to the best of our knowledge so that only relevant leaks
and errors are reported, you will likely have to do a bit of digging to find the relevant parts.

Valgrind will likely run very slowly, and sometimes  you will know already which exact tests are those which may
have problems, or are relevant to your particular changes. To run these::

$ valgrind --tool=memcheck --track-origins=yes --leak-check=full --suppressions=devel/valgrind-suppress-all-but-c.supp pytest -v tests/<test_file>::<test_func>

