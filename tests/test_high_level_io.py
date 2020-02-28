from os import path

import pytest

from py21cmfast import LightCone
from py21cmfast import run_lightcone


@pytest.fixture(scope="module")
def lc():
    return run_lightcone(
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


def test_lightcone_io_abspath(lc, tmpdirec):
    lc.save(tmpdirec / "abs_path_lightcone.h5")
    assert path.exists(tmpdirec / "abs_path_lightcone.h5")
