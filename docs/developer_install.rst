
Install as a Developer
======================

First clone the repository from GitHub::

    git clone git@github.com:21cmfast/21cmFAST.git
    cd 21cmFAST

Then, install in "editable" or "development" mode with a choice of package manager:

.. tab-set::

    .. tab-item:: uv (rec. for linux)
        :sync: uv

        .. code-block:: bash

            [COMPILE OPTIONS] uv pip install -e .[dev]

    .. tab-item:: conda/pip
        :sync: conda

        .. code-block:: bash

            [COMPILE OPTIONS] pip install -e .[dev]

The ``[dev]`` "extra" here installs all development dependencies. You can instead use
``[tests]`` if you only want dependencies for testing, or ``[docs]`` to be able to
compile the documentation.

Finally, install pre-commit hooks to ensure code quality::

    pre-commit install
