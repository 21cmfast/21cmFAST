"""Test of the Lightconer subclasses."""

import pytest

import attr
import numpy as np
from astropy import units as un
from astropy_healpix import HEALPix
from dataclasses import dataclass

from py21cmfast import (
    BrightnessTemp,
    Coeval,
    CosmoParams,
    InitialConditions,
    IonizedBox,
    PerturbedField,
    UserParams,
)
from py21cmfast import lightcones as lcn


@pytest.fixture(scope="module")
def equal_cdist():
    return lcn.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2.0 * un.Mpc,  # Mpc
        quantities=("brightness_temp",),
    )


@pytest.fixture(scope="module")
def equal_z():
    return lcn.RectilinearLightconer.with_equal_redshift_slices(
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2 * un.Mpc,
        quantities=("brightness_temp",),
    )


@pytest.fixture(scope="module")
def equal_z_angle():
    hp = HEALPix(nside=4)
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    return lcn.AngularLightconer.with_equal_redshift_slices(
        latitude=lat.radian,
        longitude=lon.radian,
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2 * un.Mpc,
        quantities=("brightness_temp",),
    )


@pytest.fixture(scope="module")
def freqbased(equal_cdist):
    return lcn.RectilinearLightconer.from_frequencies(
        freqs=1420.4 * un.MHz / (1 + equal_cdist.lc_redshifts),
        quantities=("brightness_temp",),
    )


@dataclass
class MockCoeval:
    redshift: float
    brightness_temp: np.ndarray
    user_params: UserParams
    cosmo_params: CosmoParams


def get_uniform_coeval(redshift, fill=1.0, BOX_LEN=100, HII_DIM=50):
    up = UserParams(BOX_LEN=BOX_LEN, HII_DIM=HII_DIM)

    return MockCoeval(
        redshift=redshift,
        brightness_temp=fill * np.ones((up.HII_DIM, up.HII_DIM, up.HII_DIM)),
        user_params=up,
        cosmo_params=CosmoParams(),
    )


def test_equality(
    equal_cdist: lcn.RectilinearLightconer, freqbased: lcn.RectilinearLightconer
):
    assert equal_cdist == freqbased


@pytest.mark.parametrize("lc", ["equal_z", "equal_z_angle"])
def test_uniform_coevals(request, lc):
    """Test that uniform boxes interpolate the way we want."""
    lc = request.getfixturevalue(lc)
    z6 = get_uniform_coeval(redshift=6.0, fill=0)
    z7 = get_uniform_coeval(redshift=7.0, fill=1.0)

    q, idx, out = next(lc.make_lightcone_slices(z6, z7))

    assert np.all(out >= 0)
    assert np.all(out <= 1)
    assert np.all(out[..., 0] == 0)
