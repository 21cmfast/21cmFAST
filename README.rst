

.. image:: https://raw.githubusercontent.com/nikos-triantafyllou/21cmFAST/readme-updates/docs/images/Logo_horizontal_transparent.png

===========================================
A fast simulator of the first billion years
===========================================


.. start-badges
.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/21cmFAST/21cmFAST
.. image:: https://img.shields.io/pypi/v/21cmFAST.svg
    :target: https://pypi.org/pypi/21cmFAST
.. image:: https://img.shields.io/pypi/l/21cmFAST.svg
    :target: https://github.com/21cmFAST/21cmFAST/blob/main/LICENSE
.. image:: https://codecov.io/gh/21cmfast/21cmFAST/graph/badge.svg?token=sPc47SaC7Y
    :target: https://codecov.io/gh/21cmfast/21cmFAST
.. image:: https://img.shields.io/pypi/pyversions/21cmFAST.svg
    :target: https://pypi.python.org/pypi/21cmFAST
.. image:: https://github.com/21cmfast/21cmFAST/actions/workflows/test_suite.yaml/badge.svg
    :target: https://github.com/21cmfast/21cmFAST/actions/workflows/test_suite.yaml
.. image:: https://readthedocs.org/projects/21cmfast/badge/?version=latest
    :target: https://21cmfast.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation
.. image:: https://img.shields.io/conda/dn/conda-forge/21cmFAST
    :target: https://github.com/conda-forge/21cmfast-feedstock
    :alt: Conda
.. image:: https://joss.theoj.org/papers/10.21105/joss.02582/status.svg
    :target: https://doi.org/10.21105/joss.02582
.. end-badges


``21cmFAST`` is a semi-numerical code that produces 3D cosmological realisations of many
physical fields in the early Universe.
It is super-fast, combining the excursion set formalism with perturbation theory to
efficiently generate density, velocity, halo, ionization, spin temperature, 21-cm, and
even ionizing flux fields (see the above lightcones!).
It has been tested extensively against numerical simulations, with excellent agreement
at the relevant scales.

``21cmFAST`` has been widely used, for example, by the `MWA <https://www.mwatelescope.org/>`_,
`LOFAR <https://www.lofar.org/>`_, `HERA <https://reionization.org/>`_,
and `SKA <https://www.skatelescope.org/>`_ teams to model the
large-scale cosmological 21-cm signal. The speed of ``21cmFAST`` is important
to produce simulations that are large enough (several Gpc across) to represent modern
low-frequency observations.

Full **documentation** (with `examples <https://21cmfast.readthedocs.io/tutorials.html>`_,
`installation instructions <https://21cmfast.readthedocs.io/en/latest/installation.html>`_ and
`full API reference <https://21cmfast.readthedocs.io/en/latest/reference/index.html>`_)
found at https://21cmfast.readthedocs.io.

To acknowledge the use of ``21cmFAST``, please see the
`acknowledgements page <https://21cmfast.readthedocs.io/en/latest/acknowledge.html>`_.

New Features in 4.0.0
=====================

* A discrete halo sampler allowing for the creation of lightcones of galaxy properties and the
  inclusion of stochasticity. These discrete sources are self-consistently used in the IGM calculations
* The Inclusion of the Sheth-Tormen conditional halo mass function.
* Re-designed input/output structures which prioritise transparency.
* Refactoring of several of the C backend files for much easier development.
* A lower-level testing framework for calculations done in the backend.

As of ``v3.0.0``, ``21cmFAST`` is conveniently wrapped in Python to enable more dynamic code.
See the `Changelog <https://21cmfast.readthedocs.io/en/latest/changelog.html>`_ for a
full list of changes.
