---
title: '21cmFAST v3: A fully Python-integrated simulation code to generate 3D 21cm fluctuation fields through cosmic history.'
tags:
  - Python
  - astronomy
  - cosmology
  - simulation
authors:
  - name: Steven G. Murray
    orcid: 0000-0003-3059-3823
    affiliation: 1
  - name: Bradley Greig
    orcid: xxxxxx
    affiliation: 2
  - name: Andrei Mesinger
    orcid: xxxxxx
    affiliation: 3
affiliations:
 - name: School of Earth and Space Exploration, Arizona State University, Phoenix, USA
   index: 1
date: 29 Feb 2020
bibliography: paper.bib
---

# Summary

## Intro to field
The field of 21cm cosmology -- in which the hyperfine spectral line of neutral hydrogen
is mapped over large swathes of the Universe's history -- has developed radically over
the last decade.
Notwithstanding the overwhelming technical challenges associated with the observation
and eventual detection of this signal, the promise of the field is to revolutionise our
knowledge of the first light sources: a worthy prize indeed.
In order to interpret the eventual observational data, a range of physical models have
been developed -- from simple analytic models of the global history of reionization,
through to fully hydrodynamical simulations of the 3D evolution of the brightness
temperature of the spectral line.
Between these extremes lies an especially versatile middle-ground: fast semi-analytic
models that approximate the full 3D evolution of the field.
These have the advantage of being comparable to the full first-principles
hydrodynamic simulations, but significantly quicker to run; so much so that they can
be used to produce thousands of realisations in order to explore the very wide
parameter space that still remains consistent with the data.


## Intro to 21cmFAST older versions
Amongst practioners in the field of 21cm cosmology, the `21cmFAST` program has become
the *de facto* standard for such semi-analytic simulators. `21cmFAST` [@mesinger2010]
is a high-performance C code that uses the excursion set formalism [@furlanetto2004] to
identify regions of ionized hydrogen atop a cosmological density field evolved using
first- or second-order perturbation theory.
However, `21cmFAST` is a highly specialized code, and as such has suffered from its
difficulty of use.
This is more serious than the simple discomfort of end-users (or the reduction in
end-user-ship that it entails); the lack of modularity within the code has led to
widespread code "branching" as researchers hack new physical features of interest
into the C code; the lack of a streamlined API has led derivative codes which run
multiple realizations of `21cmFAST` simulations (such as the Monte Carlo simulator,
`21CMMC`) to re-write large portions of the code in order to serve their purpose.
It is thus of critical importance, as the field moves forward in its understanding -- and
the range and scale of physical models of interest continues to increase -- to
reformulate the `21cmFAST` code in order to provide a fast, modular, well-documented,
well-tested, stable simulator for the community.


## Summary/motivation for new 21cmFAST
In `21cmFAST` v3+, which this paper presents, this goal has come a significant way towards
achievement.
While keeping the same core functionality of previous versions of `21cmFAST`, it has
been fully integrated into a Python package, with a simple and intuitive interface, and
a great deal more flexibility.
At a higher level, in order to maintain best practices, a community of users and
developers has coalesced into a formal collaboration which maintains the project via a
Github organization.
This allows the code to be consistently monitored for quality, maintaining high test
coverage, stylistic integrity, dependable release strategies and versioning,
and peer code review.
It also provides a single point-of-reference for the community to obtain the code,
report bugs and request new features (or get involved in development).

A significant part of the work of moving to a Python interface has been the
development of a robust series of underlying Python structures which handle the passing
of data between Python and C via the `CFFI` library.
This foundational work provides a platform for future versions to extend the scientific
capabilities of the underlying simulation code.
The primary *new* usability features of `21cmFAST` v3+ are:

* Convenient (Python) data objects which simplify access to and processing of the various
  fields that form the brightness temperature.
* Enhancement of modularity: the underlying C functions for each step of the simulation
  have been de-coupled, so that arbitrary functionality can be injected into the process.
* Simple `pip`-based installation.
* Robust on-disk caching/writing of data, both for efficiency and simplified reading of
  previously processed data (using HDF5).
* Simple high-level API to generate either coeval cubes or full lightcone data.
* Improved exception handling and debugging.
* OpenMP parallel processing.
* Convenient plotting routines.
* Comprehensive API documentation and tutorials.
* Comprehensive test suite (and continuous integration).
* Strict `semantic versioning <https://semver.org>`_.

While in v3 we have focused on the establishment of a stable and extendible infrastructure,
we have also incorporated several new scientific features:

* Generate transfer functions using the `CLASS` Boltzmann code.
* Simulate the effects of relative velocities between dark matter and Baryons [@munoz2018?].
* Incorporation of the full mass-dependent parameterization of [@park2018].
* Calculate luminosity functions from the input astrophysical parameters [@xxx?]
* Mitigation of non-conservation of photons inherent in the excursion set formalism.
* Mini-halo populations.

## Future plans
`21cmFAST` is still in very active development.
Amongst further usability and performance improvements,
future versions will see several new physical models implemented,
including milli-charged dark matter models and forward-modelled CMB auxiliary data.

In addition, `21cmFAST` will be incorporated into large-scale inference codes, such as
`21CMMC`, and is being used to create large data-sets for inference via machine learning.
We hope that with this new framework, `21cmFAST` will remain an important component
 of 21cm cosmology for years to come.

# Acknowledgements


# References
