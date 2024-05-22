"""Test of the Lightconer subclasses."""

import pytest

import attr
import numpy as np
import re
from astropy import units as un
from astropy.cosmology import Planck18, z_at_value
from astropy_healpix import HEALPix
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from py21cmfast import (
    BrightnessTemp,
    Coeval,
    CosmoParams,
    FlagOptions,
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
        get_los_velocity=True,
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


def test_incompatible_coevals(equal_cdist):
    z6 = get_uniform_coeval(redshift=6.0, fill=0)
    z7 = get_uniform_coeval(redshift=7.0, fill=1.0)

    # artificially change a cosmo param
    orig = z7.cosmo_params.SIGMA_8
    z7.cosmo_params.update(SIGMA_8=2 * orig)

    with pytest.raises(ValueError, match="c1 and c2 must have the same cosmo"):
        next(equal_cdist.make_lightcone_slices(z6, z7))

    z7.cosmo_params.update(SIGMA_8=orig)

    orig = z7.user_params.BOX_LEN
    z7.user_params.update(BOX_LEN=2 * z7.user_params.BOX_LEN)

    with pytest.raises(
        ValueError, match="c1 and c2 must have the same user parameters"
    ):
        next(equal_cdist.make_lightcone_slices(z6, z7))


def test_coeval_redshifts_outside_box(equal_cdist):
    z0 = equal_cdist.lc_redshifts[0] - 3
    z1 = equal_cdist.lc_redshifts[0] - 1

    z6 = get_uniform_coeval(redshift=z0, fill=0)
    z7 = get_uniform_coeval(redshift=z1, fill=1.0)

    q, idx, out = next(equal_cdist.make_lightcone_slices(z6, z7))
    assert q is None
    assert len(idx) == 0
    assert out is None


def test_bad_coeval_sizes_for_redshift_interp(equal_cdist):
    c1 = np.zeros((10, 10, 10))
    c2 = np.zeros((20, 20, 20))

    with pytest.raises(
        ValueError, match="coeval_a and coeval_b must have the same shape"
    ):
        equal_cdist.redshift_interpolation(3.0, c1, c2, 1, 4)


def test_bad_kind_for_redshift_interp(equal_cdist):
    c1 = np.zeros((10, 10, 10))
    c2 = np.zeros((10, 10, 10))

    with pytest.raises(ValueError, match="kind must be 'mean' or 'mean_max'"):
        equal_cdist.redshift_interpolation(3.0, c1, c2, 1, 4, kind="min")


def test_bad_instantiation():
    with pytest.raises(
        ValueError, match="Either lc_distances or lc_redshifts must be provided"
    ):
        lcn.RectilinearLightconer()

    with pytest.raises(ValueError, match="lc_distances must be non-negative"):
        lcn.RectilinearLightconer(lc_distances=np.linspace(-1, 1, 10) * un.Mpc)

    with pytest.raises(ValueError, match="lc_redshifts must be non-negative"):
        lcn.RectilinearLightconer(lc_redshifts=np.linspace(-1, 1, 10))


def test_equal_cdist_endpoint():
    hp = HEALPix(nside=4)
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))

    d0 = Planck18.comoving_distance(6.0)
    res = 2 * un.Mpc
    d1 = d0 + 20 * res
    zmax = z_at_value(Planck18.comoving_distance, d1)

    lc = lcn.AngularLightconer.with_equal_cdist_slices(
        latitude=lat.radian,
        longitude=lon.radian,
        min_redshift=6.0,
        max_redshift=zmax,
        resolution=res,
    )
    assert np.isclose(lc.lc_distances.max(), d1, atol=d1 / 20)


def test_equal_redshift_bad_instantiation():
    hp = HEALPix(nside=4)
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))

    with pytest.raises(ValueError, match="Either dz or resolution must be provided"):
        lcn.AngularLightconer.with_equal_redshift_slices(
            latitude=lat.radian,
            longitude=lon.radian,
            min_redshift=6.0,
            max_redshift=7.0,
            resolution=None,
            dz=None,
        )


def test_angular_lightconer_bad_instantiation():
    hp = HEALPix(nside=4)
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))

    with pytest.raises(ValueError, match="longitude must be 1-dimensional"):
        lcn.AngularLightconer(
            longitude=lon[None, :].radian,
            latitude=lat.radian,
            lc_redshifts=np.linspace(6, 7, 10),
        )

    with pytest.raises(
        ValueError, match=re.escape("longitude must be in the range [0, 2pi]")
    ):
        lcn.AngularLightconer(
            longitude=lon.radian + 2 * np.pi,
            latitude=lat.radian,
            lc_redshifts=np.linspace(6, 7, 10),
        )

    with pytest.raises(
        ValueError, match="longitude and latitude must have the same shape"
    ):
        lcn.AngularLightconer(
            longitude=lon.radian,
            latitude=lat[None, :].radian,
            lc_redshifts=np.linspace(6, 7, 10),
        )


def test_rotation_equality():
    hp = HEALPix(nside=4)
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))

    lc1 = lcn.AngularLightconer.with_equal_cdist_slices(
        latitude=lat.radian,
        longitude=lon.radian,
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2 * un.Mpc,
    )

    lc2 = lcn.AngularLightconer.with_equal_cdist_slices(
        latitude=lat.radian,
        longitude=lon.radian,
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2 * un.Mpc,
    )

    assert lc1 == lc2

    rot = Rotation.from_euler("z", np.pi / 2)
    lc3 = lcn.AngularLightconer.with_equal_cdist_slices(
        latitude=lat.radian,
        longitude=lon.radian,
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2 * un.Mpc,
        rotation=rot,
    )

    assert lc1 != lc3


def test_validation_options_angular(equal_z_angle):
    with pytest.raises(ValueError, match="APPLY_RSDs must be False"):
        equal_z_angle.validate_options(
            flag_options=FlagOptions(APPLY_RSDS=True), user_params=UserParams()
        )

    with pytest.raises(ValueError, match="To get the LoS velocity, you need to set"):
        equal_z_angle.validate_options(
            user_params=UserParams(KEEP_3D_VELOCITIES=False),
            flag_options=FlagOptions(APPLY_RSDS=False),
        )
