#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Setup the package."""

from __future__ import absolute_import
from __future__ import print_function

import glob
import io
import os
import re
import shutil
from distutils.dir_util import copy_tree
from os.path import dirname
from os.path import expanduser
from os.path import join
from setuptools import find_packages
from setuptools import setup


def _read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ).read()


def _find_version(*file_paths):
    version_file = _read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# ======================================================================================
# Create a user-level config directory for 21cmFAST, for configuration.
cfgdir = expanduser(join("~", ".21cmfast"))

pkgdir = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(cfgdir):
    os.makedirs(cfgdir)


def _safe_copy_tree(src, dst, safe=None):
    safe = safe or []
    for fl in glob.glob(join(src, "*")):
        fname = os.path.basename(fl)
        if fname not in safe or not os.path.exists(join(dst, fname)):
            if os.path.isdir(join(src, fname)):
                if os.path.exists(join(dst, fname)):
                    shutil.rmtree(join(dst, fname))
                shutil.copytree(join(src, fname), join(dst, fname))
            else:
                shutil.copy(join(src, fname), join(dst, fname))


# Copy the user data into the config directory.
# We *don't* want to overwrite the config file that is already there, because maybe the user
# has changed the configuration, and that would destroy it!
_safe_copy_tree(join(pkgdir, "user_data"), cfgdir, safe="config.yml")
# ======================================================================================

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
]

doc_req = ["nbsphinx", "numpydoc", "sphinx >= 1.3", "sphinx-rtd-theme"]

setup(
    name="21cmFAST",
    version=_find_version("src", "py21cmfast", "__init__.py"),
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
        # 'tqdm',
        "numpy",
        "pyyaml",
        "cffi>=1.0",
        "scipy",
        "astropy>=2.0",
        "h5py>=2.8.0",
        "cached_property",
        "matplotlib",
    ],
    extras_require={"tests": test_req, "docs": doc_req, "dev": test_req + doc_req},
    setup_requires=["cffi>=1.0"],
    entry_points={"console_scripts": ["21cmfast = py21cmfast.cli:main"]},
    cffi_modules=["{pkgdir}/build_cffi.py:ffi".format(pkgdir=pkgdir)],
)
