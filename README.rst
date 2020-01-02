========
21cmFAST
========

.. start-badges
.. image:: https://travis-ci.org/21cmFAST/21cmFAST.svg
    :target: https://travis-ci.org/21cmFAST/21cmFAST
.. image:: https://coveralls.io/repos/github/21cmFAST/21cmFAST/badge.svg
    :target: https://coveralls.io/github/21cmFAST/21cmFAST
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black
.. image:: https://readthedocs.org/projects/21cmfast/badge/?version=latest
    :target: https://21cmfast.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. end-badges

**A semi-numerical cosmological simulation code for the radio 21cm signal.**

This is the official repository for 21cmFAST. As of `v3.0.0`, it is conveniently
wrapped in Python to enable more dynamic code.


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


Documentation
=============

Full documentation (with examples, installation instructions and full API reference)
found at https://21cmfast.readthedocs.org.

Acknowledging
=============
If you find `21cmFAST` useful in your research please cite at least one of the following
(whichever is most suitable to you):

    Andrei Mesinger and Steven Furlanetto, "Efficient Simulations of Early Structure
    Formation and Reionization", The Astrophysical Journal, Volume 669, Issue 2,
    pp. 663-675 (2007),
    https://ui.adsabs.harvard.edu/link_gateway/2007ApJ...669..663M/doi:10.1086/521806

    Andrei Mesinger, Steven Furlanetto and Renyue Cen, "21CMFAST: a fast, seminumerical
    simulation of the high-redshift 21-cm signal", Monthly Notices of the Royal
    Astronomical Society, Volume 411, Issue 2, pp. 955-972 (2011),
    https://ui.adsabs.harvard.edu/link_gateway/2011MNRAS.411..955M/doi:10.1111/j.1365-2966.2010.17731.x
