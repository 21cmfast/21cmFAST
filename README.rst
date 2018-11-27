======
21CMMC
======

.. start-badges
.. image:: https://travis-ci.org/BradGreig/Hybrid21CM.svg?branch=develop-steven
    :target: https://travis-ci.org/BradGreig/Hybrid21CM
.. image:: https://coveralls.io/repos/github/BradGreig/Hybrid21CM/badge.svg?branch=develop-steven
    :target: https://coveralls.io/github/BradGreig/Hybrid21CM?branch=develop-steven

.. end-badges

An extensible MCMC framework for 21cmFAST.

* Free software: MIT license

Features
========

* Simple interface to the popular ``21cmFAST`` ionization code.
* Robust on-disk caching/writing both for efficiency and simplified reading of previously processed data (using HDF5).
* The most up-to-date parameterization of ``21cmFAST``, with consistent spin temperature and inhomogeneous recombinations
  available.
* Convenient data objects which simplify access to and processing of the various density and ionization fields.
* De-coupled functions mean that arbitrary functionality can be injected into the process.
* Seamless integration with ``emcee``-based MCMC.
* MCMC is easily extensible via the addition of different likelihoods using the same underlying data.

Quick Usage
===========

Using 21CMMC is as easy as::

    >>> import py21cmmc as p21
    >>> lightcone = p21.run_lightcone(redshift=8.0)

Documentation
=============

To view the docs, install the ``requirements_dev.txt`` packages, go to the docs/ folder, and type "make html", then
open the ``index.html`` file in the ``_build/html`` directory.
