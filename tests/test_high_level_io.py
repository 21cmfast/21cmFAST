from os import path

import pytest

import numpy as np

from py21cmfast import Coeval
from py21cmfast import LightCone
from py21cmfast import run_coeval
from py21cmfast import run_lightcone


@pytest.fixture(scope="module")
def lc():
    return run_lightcone(
        redshift=10.0,
        user_params={"HII_DIM": 35, "BOX_LEN": 50},
        regenerate=True,
        write=False,
    )


@pytest.fixture(scope="module")
def coeval():
    return run_coeval(
        redshift=10.0,
        user_params={"HII_DIM": 35, "BOX_LEN": 50},
        regenerate=True,
        write=False,
    )


def test_lightcone_roundtrip(tmpdirec, lc):
    fname = lc.save(direc=tmpdirec.strpath)
    lc2 = LightCone.read(fname)

    assert lc == lc2
    assert lc.get_unique_filename() == lc2.get_unique_filename()
    assert np.all(np.isclose(lc.brightness_temp, lc2.brightness_temp))


def test_lightcone_io_abspath(lc, tmpdirec):
    lc.save(tmpdirec / "abs_path_lightcone.h5")
    assert path.exists(tmpdirec / "abs_path_lightcone.h5")


def test_coeval_roundtrip(tmpdirec, coeval):
    fname = coeval.save(direc=tmpdirec.strpath)
    coeval2 = Coeval.read(fname)

    assert coeval == coeval2
    assert coeval.get_unique_filename() == coeval2.get_unique_filename()
    assert np.all(
        np.isclose(
            coeval.brightness_temp_struct.brightness_temp,
            coeval2.brightness_temp_struct.brightness_temp,
        )
    )
