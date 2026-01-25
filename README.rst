

.. image:: ./images/Logo_horizontal_blue_red.jpg

========
21cmFAST: **A semi-numerical cosmological simulation code for the radio 21-cm signal.**
========


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





This is the official repository for ``21cmFAST``: a semi-numerical code that is able to
produce 3D cosmological realisations of many physical fields in the early Universe.
It is super-fast, combining the excursion set formalism with perturbation theory to
efficiently generate density, velocity, halo, ionization, spin temperature, 21-cm, and
even ionizing flux fields (see the above lightcones!).
It has been tested extensively against numerical simulations, with excellent agreement
at the relevant scales.

``21cmFAST`` has been widely used, for example, by the Murchison Widefield Array (MWA),
LOw-Frequency ARray (LOFAR) and Hydrogen Epoch of Reionization Array (HERA), to model the
large-scale cosmological 21-cm signal. In particular, the speed of ``21cmFAST`` is important
to produce simulations that are large enough (several Gpc across) to represent modern
low-frequency observations.

Documentation
=============
Full documentation (with examples, installation instructions and full API reference)
found at https://21cmfast.readthedocs.org.

New Features in 4.0.0
=====================

* A discrete halo sampler allowing for the creation of lightcones of galaxy properties and the
  inclusion of stochasticity. These discrete sources are self-consistently used in the IGM calculations
* The Inclusion of the Sheth-Tormen conditional halo mass function.
* Re-designed input/output structures which prioritise transparency.
* Refactoring of several of the C backend files for much easier development.
* A lower-level testing framework for calculations done in the backend.

As of ``v3.0.0``, ``21cmFAST`` is conveniently wrapped in Python to enable more dynamic code.


New Features in 3.0.0+
======================

* Robust on-disk caching/writing both for efficiency and simplified reading of
  previously processed data (using HDF5).
* Convenient data objects which simplify access to and processing of the various density
  and ionization fields.
* De-coupled functions mean that arbitrary functionality can be injected into the process.
* Improved exception handling and debugging
* Comprehensive documentation
* Comprehensive test suite.
* Strict `semantic versioning <https://semver.org>`_.


Installation
============
We support Linux and MacOS (please let us know if you are successful in installing on
Windows!). On these systems, the simplest way to get ``21cmFAST`` is by using
`conda <https://www.anaconda.com/>`_::

    conda install -c conda-forge 21cmFAST

``21cmFAST`` is also available on PyPI, so that ``pip install 21cmFAST`` also works. However,
it depends on some external (non-python) libraries that may not be present, and so this
method is discouraged unless absolutely necessary. If using ``pip`` to install ``21cmFAST``
(especially on MacOS), we thoroughly recommend reading the detailed
`installation instructions <https://21cmfast.readthedocs.io/en/latest/installation.html>`_.



Acknowledging
=============
If you use ``21cmFAST v3+`` in your research please cite both of:

    Murray et al., (2020). 21cmFAST v3: A Python-integrated C code for generating 3D
    realizations of the cosmic 21cm signal. Journal of Open Source Software, 5(54),
    2582, https://doi.org/10.21105/joss.02582

    Andrei Mesinger, Steven Furlanetto and Renyue Cen, "21CMFAST: a fast, seminumerical
    simulation of the high-redshift 21-cm signal", Monthly Notices of the Royal
    Astronomical Society, Volume 411, Issue 2, pp. 955-972 (2011),
    https://ui.adsabs.harvard.edu/link_gateway/2011MNRAS.411..955M/doi:10.1111/j.1365-2966.2010.17731.x

In addition, the following papers introduce various features into ``21cmFAST``. If you use
these features, please cite the relevant papers.

Discrete Halo Sampler / version 4:

    Davies, J. E., Mesinger, A., Murray, S. G.,
    "Efficient simulation of discrete galaxy populations and associated radiation fields during the first billion years",
    eprint arXiv:2504.17254, 2025. https://doi.org/10.48550/arXiv.2504.17254

Mini-halos:

    Muñoz, J.B., Qin, Y., Mesinger, A., Murray, S., Greig, B., and Mason, C.,
    "The Impact of the First Galaxies on Cosmic Dawn and Reionization",
    Monthly Notices of the Royal Astronomical Society, vol. 511, no. 3,
    pp 3657-3681, 2022 https://doi.org/10.1093/mnras/stac185
    (for DM-baryon relative velocities)

    Qin, Y., Mesinger, A., Park, J., Greig, B., and Muñoz, J. B.,
    “A tale of two sites - I. Inferring the properties of minihalo-hosted galaxies from
    current observations”, Monthly Notices of the Royal Astronomical Society, vol. 495,
    no. 1, pp. 123–140, 2020. https://doi.org/10.1093/mnras/staa1131.
    (for Lyman-Werner and first implementation)

Mass-dependent ionizing efficiency:

    Park, J., Mesinger, A., Greig, B., and Gillet, N.,
    “Inferring the astrophysics of reionization and cosmic dawn from galaxy luminosity
    functions and the 21-cm signal”, Monthly Notices of the Royal Astronomical Society,
    vol. 484, no. 1, pp. 933–949, 2019. https://doi.org/10.1093/mnras/stz032.

If you are unsure which modules are used within your simulations, we provide a handy function
to print out which works to refer ``py21cmfast.utils.show_references``, which accepts a single instance of
the ``InputParameters`` class and shows which papers are relevant for your simulation.
