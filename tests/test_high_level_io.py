from os import path

import pytest

import numpy as np

from py21cmfast import Coeval
from py21cmfast import LightCone
from py21cmfast import run_coeval
from py21cmfast import run_lightcone


@pytest.fixture(scope="module")
def coeval(ic):
    return run_coeval(redshift=10.0, init_box=ic)


def test_lightcone_roundtrip(test_direc, lc):
    fname = lc.save(direc=test_direc)
    lc2 = LightCone.read(fname)

    assert lc == lc2
    assert lc.get_unique_filename() == lc2.get_unique_filename()
    assert np.all(np.isclose(lc.brightness_temp, lc2.brightness_temp))


def test_lightcone_io_abspath(lc, test_direc):
    lc.save(test_direc / "abs_path_lightcone.h5")
    assert (test_direc / "abs_path_lightcone.h5").exists()


def test_coeval_roundtrip(test_direc, coeval):
    fname = coeval.save(direc=test_direc)
    coeval2 = Coeval.read(fname)

    assert coeval == coeval2
    assert coeval.get_unique_filename() == coeval2.get_unique_filename()
    assert np.all(
        np.isclose(
            coeval.brightness_temp_struct.brightness_temp,
            coeval2.brightness_temp_struct.brightness_temp,
        )
    )
