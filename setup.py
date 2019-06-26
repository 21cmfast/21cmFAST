#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import os
import re
from distutils.dir_util import copy_tree
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import expanduser
from os.path import join
from os.path import splitext
from shutil import copyfile

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ).read()


def find_version(*file_paths):
    version_file = read(*file_paths)
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

copyfile(join(pkgdir, "config.yml"), join(cfgdir, "config.yml"))
copyfile(join(pkgdir, "runconfig_example.yml"), join(cfgdir, "runconfig_example.yml"))
copy_tree(join(pkgdir, "External_tables"), join(cfgdir, "External_tables"))

boxdir = os.environ.get("BOXDIR", None)

if boxdir:
    with open(join(cfgdir, "config.yml"), "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("boxdir"):
                lines[i] = line.replace(line.split(": ")[-1], boxdir)

    with open(join(cfgdir, "config.yml"), "w") as f:
        f.write("\n".join(lines))

# ======================================================================================================================

# Enable code coverage for C code: we can't use CFLAGS=-coverage in tox.ini, since that may mess with compiling
# dependencies (e.g. numpy). Therefore we set SETUPPY_CFLAGS=-coverage in tox.ini and copy it to CFLAGS here (after
# deps have been safely installed).
if "TOXENV" in os.environ and "SETUPPY_CFLAGS" in os.environ:
    os.environ["CFLAGS"] = os.environ["SETUPPY_CFLAGS"]

setup(
    name="21cmFAST",
    version=find_version("src", "py21cmfast", "__init__.py"),
    license="MIT license",
    description="A semi-numerical cosmological simulation code for the 21cm signal",
    long_description="%s\n%s"
    % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", read("README.rst")
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
    ),
    author="The 21cmFAST coredev team",
    author_email="21cmfast.coredev@gmail.com",
    url="https://github.com/21cmFAST/21cmFAST",
    packages=find_packages("src"),
    package_dir={"": "src"},
    # py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
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
    ],
    setup_requires=["cffi>=1.0"],
    entry_points={"console_scripts": ["21cmfast = py21cmfast.cli:main"]},
    cffi_modules=["{pkgdir}/build_cffi.py:ffi".format(pkgdir=pkgdir)],
)
