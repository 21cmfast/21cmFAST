"""Test of the Lightconer subclasses."""

import re
from dataclasses import dataclass

import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology import Planck18, z_at_value
from astropy_healpix import HEALPix
from scipy.spatial.transform import Rotation

from py21cmfast import (
    CosmoParams,
    InputParameters,
    SimulationOptions,
    AstroOptions
)
from py21cmfast import lightconers as lcn


@pytest.fixture(scope="module")
def rect_lcner():
    return lcn.RectilinearLightconer.between_redshifts(
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2.0 * un.Mpc,  # Mpc
        quantities=("brightness_temp",),
    )


@pytest.fixture(scope="module")
def ang_lcner():
    hp = HEALPix(nside=4)
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    return lcn.AngularLightconer.between_redshifts(
        latitude=lat.radian,
        longitude=lon.radian,
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2 * un.Mpc,
        quantities=("brightness_temp",),
    )


@dataclass
class MockCoeval:
    """A mock coeval object for testing."""

    redshift: float
    brightness_temp: np.ndarray
    simulation_options: SimulationOptions
    cosmo_params: CosmoParams
    astro_options: AstroOptions


def get_uniform_coeval(redshift, fill=1.0, BOX_LEN=100, HII_DIM=50):
    up = SimulationOptions(BOX_LEN=BOX_LEN, HII_DIM=HII_DIM)

    return MockCoeval(
        redshift=redshift,
        brightness_temp=fill * np.ones((up.HII_DIM, up.HII_DIM, up.HII_DIM)),
        simulation_options=up,
        cosmo_params=CosmoParams(),
        astro_options=AstroOptions()
    )


def test_equality(
    rect_lcner: lcn.RectilinearLightconer,
):
    assert rect_lcner == rect_lcner


@pytest.mark.parametrize("lc", ["rect_lcner", "ang_lcner"])
def test_uniform_coevals(request, lc):
    """Test that uniform boxes interpolate the way we want."""
    lc = request.getfixturevalue(lc)
    z6 = get_uniform_coeval(redshift=6.0, fill=0)
    z7 = get_uniform_coeval(redshift=7.0, fill=1.0)

    q, idx, out = next(lc.make_lightcone_slices(z6, z7))

    assert np.all(out >= 0)
    assert np.all(out <= 1)
    assert np.all(out[..., 0] == 0)


def test_incompatible_coevals(rect_lcner):
    z6 = get_uniform_coeval(redshift=6.0, fill=0)
    z7 = get_uniform_coeval(redshift=7.0, fill=1.0)

    # artificially change a cosmo param
    orig = z7.cosmo_params.SIGMA_8
    z7.cosmo_params = z7.cosmo_params.clone(SIGMA_8=2 * orig)

    with pytest.raises(ValueError, match="c1 and c2 must have the same cosmo"):
        next(rect_lcner.make_lightcone_slices(z6, z7))

    z7.cosmo_params = z7.cosmo_params.clone(SIGMA_8=orig)

    orig = z7.simulation_options.BOX_LEN
    z7.simulation_options = z7.simulation_options.clone(
        BOX_LEN=2 * z7.simulation_options.BOX_LEN
    )

    with pytest.raises(
        ValueError, match="c1 and c2 must have the same user parameters"
    ):
        next(rect_lcner.make_lightcone_slices(z6, z7))


def test_coeval_redshifts_outside_box(rect_lcner):
    z0 = rect_lcner.lc_redshifts[0] - 3
    z1 = rect_lcner.lc_redshifts[0] - 1

    z6 = get_uniform_coeval(redshift=z0, fill=0)
    z7 = get_uniform_coeval(redshift=z1, fill=1.0)

    q, idx, out = next(rect_lcner.make_lightcone_slices(z6, z7))
    assert q is None
    assert len(idx) == 0
    assert out is None


def test_bad_coeval_sizes_for_redshift_interp(rect_lcner):
    c1 = np.zeros((10, 10, 10))
    c2 = np.zeros((20, 20, 20))

    with pytest.raises(
        ValueError, match="coeval_a and coeval_b must have the same shape"
    ):
        rect_lcner.redshift_interpolation(3.0, c1, c2, 1, 4)


def test_bad_kind_for_redshift_interp(rect_lcner):
    c1 = np.zeros((10, 10, 10))
    c2 = np.zeros((10, 10, 10))

    with pytest.raises(ValueError, match="kind must be 'mean' or 'mean_max'"):
        rect_lcner.redshift_interpolation(3.0, c1, c2, 1, 4, kind="min")


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

    lc = lcn.AngularLightconer.between_redshifts(
        latitude=lat.radian,
        longitude=lon.radian,
        min_redshift=6.0,
        max_redshift=zmax,
        resolution=res,
    )
    assert np.isclose(lc.lc_distances.max(), d1, atol=d1 / 20)


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

    lc1 = lcn.AngularLightconer.between_redshifts(
        latitude=lat.radian,
        longitude=lon.radian,
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2 * un.Mpc,
    )

    lc2 = lcn.AngularLightconer.between_redshifts(
        latitude=lat.radian,
        longitude=lon.radian,
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2 * un.Mpc,
    )

    assert lc1 == lc2

    rot = Rotation.from_euler("z", np.pi / 2)
    lc3 = lcn.AngularLightconer.between_redshifts(
        latitude=lat.radian,
        longitude=lon.radian,
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2 * un.Mpc,
        rotation=rot,
    )

    assert lc1 != lc3


def test_validation_options_angular(ang_lcner):
    inputs = InputParameters(node_redshifts=np.array([5.0, 8.0]), random_seed=42)
    inputs.evolve_input_structs(KEEP_3D_VELOCITIES=False, INCLUDE_DVDR_IN_TAU21=True)
    with pytest.raises(
        ValueError, match="To account for RSDs or velocity corrections in an angular lightcone, you need to set"
    ):
        ang_lcner.validate_options(inputs=inputs)
