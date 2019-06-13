========
21cmFAST
========

.. start-badges
.. image:: https://travis-ci.org/21cmFAST/21cmFAST.svg
    :target: https://travis-ci.org/21cmFAST/21cmFAST
.. image:: https://coveralls.io/repos/github/21cmFAST/21cmFAST/badge.svg
    :target: https://coveralls.io/github/21cmFAST/21cmFAST
.. end-badges

**A semi-numerical cosmological simulation code for the radio 21cm signal.**

This is the official repository for 21cmFAST. As of version 3.0.0, it is conveniently
wrapped in Python to enable more dynamic code.



New Features in 3.0.0+
======================

* Robust on-disk caching/writing both for efficiency and simplified reading of
  previously processed data (using HDF5).
* Convenient data objects which simplify access to and processing of the various density
  and ionization fields.
* De-coupled functions mean that arbitrary functionality can be injected into the process.
* Improved exception handling and debugging


Documentation
=============

Full documentation (with examples, installation instructions and full API reference)
found at https://21cmfast.readthedocs.org.