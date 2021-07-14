#!/usr/bin/env python
"""Setup the package."""


from setuptools import find_packages, setup

import glob
import io
import os
import re
import shutil
from os.path import dirname, expanduser, join


def _read(*names, **kwargs):
    return open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ).read()


pkgdir = os.path.dirname(os.path.abspath(__file__))


# Enable code coverage for C code: we can't use CFLAGS=-coverage in tox.ini, since that
# may mess with compiling dependencies (e.g. numpy). Therefore we set SETUPPY_
# CFLAGS=-coverage in tox.ini and copy it to CFLAGS here (after deps have been safely installed).
if "TOXENV" in os.environ and "SETUPPY_CFLAGS" in os.environ:
    os.environ["CFLAGS"] = os.environ["SETUPPY_CFLAGS"]

test_req = [
    "pre-commit",
    "pytest>=5.0",
    "pytest-cov",
    "tox",
    "pytest-remotedata>=0.3.2",
    "powerbox",
    "pytest-plt",
    "questionary",
]

doc_req = ["nbsphinx", "numpydoc", "sphinx >= 1.3", "sphinx-rtd-theme"]

setup(
    name="21cmFAST",
    license="MIT license",
    description="A semi-numerical cosmological simulation code for the 21cm signal",
    long_description="%s\n%s"
    % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", _read("README.rst")
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", _read("CHANGELOG.rst")),
    ),
    author="The 21cmFAST coredev team",
    author_email="21cmfast.coredev@gmail.com",
    url="https://github.com/21cmFAST/21cmFAST",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords=["Epoch of Reionization", "Cosmology"],
    install_requires=[
        "click",
        "numpy",
        "pyyaml",
        "cffi>=1.0",
        "scipy",
        "astropy>=2.0",
        "h5py>=2.8.0",
        "cached_property",
        "matplotlib",
        "bidict",
    ],
    extras_require={"tests": test_req, "docs": doc_req, "dev": test_req + doc_req},
    setup_requires=["cffi>=1.0", "setuptools_scm"],
    entry_points={"console_scripts": ["21cmfast = py21cmfast.cli:main"]},
    cffi_modules=[f"{pkgdir}/build_cffi.py:ffi"],
    use_scm_version=True,
)
