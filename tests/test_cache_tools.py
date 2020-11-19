"""
Tests for the tools in the wrapper.
"""

import pytest

import numpy as np

from py21cmfast import cache_tools


def test_query(ic):
    things = list(cache_tools.query_cache())

    print(things)

    classes = [t[1] for t in things]
    assert ic in classes


def test_bad_fname(tmpdirec):
    with pytest.raises(ValueError):
        cache_tools.readbox(direc=str(tmpdirec), fname="a_really_fake_file.h5")


def test_readbox_data(tmpdirec, ic):
    box = cache_tools.readbox(direc=str(tmpdirec), fname=ic.filename)

    assert np.all(box.hires_density == ic.hires_density)


def test_readbox_filter(ic, tmpdirec):
    ic2 = cache_tools.readbox(
        kind="InitialConditions", hsh=ic._md5, direc=str(tmpdirec)
    )
    assert np.all(ic2.hires_density == ic.hires_density)


def test_readbox_seed(ic, tmpdirec):
    ic2 = cache_tools.readbox(
        kind="InitialConditions",
        hsh=ic._md5,
        seed=ic.random_seed,
        direc=str(tmpdirec),
    )
    assert np.all(ic2.hires_density == ic.hires_density)


def test_readbox_nohash(ic, tmpdirec):
    with pytest.raises(ValueError):
        cache_tools.readbox(
            kind="InitialConditions", seed=ic.random_seed, direc=str(tmpdirec)
        )
