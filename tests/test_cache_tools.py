"""
Tests for the tools in the wrapper.
"""

from py21cmmc import wrapper, cache_tools
import pytest
import numpy as np


@pytest.fixture(scope="module")
def ic(tmpdirec):
    return wrapper.initial_conditions(user_params={"HII_DIM":25}, direc=tmpdirec.strpath)


def test_query(tmpdirec, ic):
    things = list(cache_tools.query_cache(direc=tmpdirec.strpath))

    print(things)

    classes = [t[1] for t in things]
    assert ic in classes


def test_bad_fname(tmpdirec):
    with pytest.raises(ValueError):
        cache_tools.readbox(direc=tmpdirec.strpath, fname="a_really_fake_file.h5")


def test_readbox_data(tmpdirec, ic):
    box = cache_tools.readbox(direc=tmpdirec.strpath, fname=ic.filename)

    assert np.all(box.hires_density == ic.hires_density)


def test_readbox_filter(ic, tmpdirec):
    ic2 = cache_tools.readbox(kind="InitialConditions", hash=ic._md5, direc=tmpdirec.strpath)
    assert np.all(ic2.hires_density == ic.hires_density)

def test_readbox_seed(ic, tmpdirec):
    ic2 = cache_tools.readbox(kind="InitialConditions", hash=ic._md5, seed=ic.random_seed, direc=tmpdirec.strpath)
    assert np.all(ic2.hires_density == ic.hires_density)

def test_readbox_nohash(ic, tmpdirec):
    with pytest.raises(ValueError):
        ic2 = cache_tools.readbox(kind="InitialConditions", seed=ic.random_seed, direc=tmpdirec.strpath)
    
