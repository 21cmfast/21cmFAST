"""Tests of HEALPix angular lightcones (AngularLightconer.healpix)."""

import re
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un
from astropy_healpix import HEALPix

from py21cmfast import CosmoParams, SimulationOptions
from py21cmfast import lightconers as lcn

_needs_subsamples = pytest.mark.skipif(
    not lcn._COSMOTILE_AVERAGES_SUBSAMPLES,
    reason="pixels averaged over sub-samples need cosmotile >= 0.3",
)


@pytest.mark.parametrize("order", ["ring", "nested"])
def test_healpix_pixel_centres(order):
    lightconer = lcn.AngularLightconer.healpix(
        nside=4, order=order, lc_redshifts=np.linspace(6, 7, 5)
    )
    lon, lat = HEALPix(nside=4, order=order).healpix_to_lonlat(np.arange(192))

    np.testing.assert_allclose(lightconer.latitude, lat.radian)
    np.testing.assert_allclose(lightconer.longitude, lon.radian)
    assert lightconer.get_shape(SimulationOptions()) == (192, 5)


def test_healpix_between_redshifts():
    kw = {"min_redshift": 6.0, "max_redshift": 7.0, "resolution": 2 * un.Mpc}
    lightconer = lcn.AngularLightconer.healpix(nside=4, quantities=("density",), **kw)
    rect = lcn.RectilinearLightconer.between_redshifts(**kw)

    np.testing.assert_allclose(lightconer.lc_distances, rect.lc_distances)
    assert lightconer.quantities == ("density",)


def test_healpix_partial_sky():
    full = lcn.AngularLightconer.healpix(nside=8, lc_redshifts=np.linspace(6, 7, 5))
    pixels = HEALPix(nside=8).cone_search_lonlat(0 * un.deg, 30 * un.deg, 20 * un.deg)
    part = lcn.AngularLightconer.healpix(
        nside=8, pixels=pixels, lc_redshifts=np.linspace(6, 7, 5)
    )

    assert 0 < len(pixels) < 768
    np.testing.assert_allclose(part.latitude, full.latitude[pixels])
    np.testing.assert_allclose(part.longitude, full.longitude[pixels])
    assert part.get_shape(SimulationOptions()) == (len(pixels), 5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": "galactic"},
        {"subsample_level": -1},
        {"pixels": [0, 768]},
        {"pixels": [-1]},
        {"pixels": [0.5]},
        {"pixels": []},
    ],
)
def test_healpix_bad_inputs(kwargs):
    with pytest.raises(ValueError, match=r"order|subsample_level|pixels"):
        lcn.AngularLightconer.healpix(
            nside=8, lc_redshifts=np.linspace(6, 7, 5), **kwargs
        )


def test_healpix_subsamples_need_cosmotile_03(monkeypatch):
    monkeypatch.setattr(lcn, "_COSMOTILE_AVERAGES_SUBSAMPLES", False)
    with pytest.raises(ValueError, match=re.escape("needs cosmotile >= 0.3")):
        lcn.AngularLightconer.healpix(
            nside=4, subsample_level=1, lc_redshifts=np.linspace(6, 7, 5)
        )

    lon, lat = HEALPix(nside=4).healpix_to_lonlat(np.arange(192))
    with pytest.raises(ValueError, match=re.escape("need cosmotile >= 0.3")):
        lcn.AngularLightconer(
            longitude=np.stack([lon.radian] * 2),
            latitude=np.stack([lat.radian] * 2),
            lc_redshifts=np.linspace(6, 7, 5),
        )


def _random_coeval_pair(seed: int = 2, hii_dim: int = 16):
    rng = np.random.default_rng(seed)
    so = SimulationOptions(BOX_LEN=32.0, HII_DIM=hii_dim)

    def coeval(redshift):
        return SimpleNamespace(
            redshift=redshift,
            simulation_options=so,
            cosmo_params=CosmoParams(),
            **{
                name: rng.normal(size=(hii_dim,) * 3).astype(np.float32)
                for name in ("density", "velocity_x", "velocity_y", "velocity_z")
            },
        )

    return coeval(6.0), coeval(6.1)


@_needs_subsamples
def test_healpix_subsample_average():
    """With sub-samples, each pixel is the mean of its 4**k HEALPix children."""
    c1, c2 = _random_coeval_pair()
    kw = {
        "min_redshift": 6.0,
        "max_redshift": 6.1,
        "resolution": 2 * un.Mpc,
        "origin": np.array([8.0, 8.0, 8.0]) * un.pixel,
        "quantities": ("density", "los_velocity"),
    }
    coarse = lcn.AngularLightconer.healpix(
        nside=4, order="nested", subsample_level=1, **kw
    )
    fine = lcn.AngularLightconer.healpix(nside=8, order="nested", **kw)
    assert coarse.latitude.shape == (4, 192)
    assert coarse.get_shape(c1.simulation_options)[0] == 192

    n_slices = 0
    for (q, idx, avg), (q2, idx2, children) in zip(
        coarse.make_lightcone_slices(c1, c2),
        fine.make_lightcone_slices(c1, c2),
        strict=True,
    ):
        assert (q, idx) == (q2, idx2)
        assert avg.shape == (192,)
        # The children of nested pixel p are the nested pixels 4p, ..., 4p + 3.
        np.testing.assert_allclose(
            avg, np.asarray(children).reshape(192, 4).mean(axis=1), rtol=1e-5, atol=1e-6
        )
        n_slices += 1
    assert n_slices > 0


@_needs_subsamples
def test_healpix_subsamples_ring_order():
    """In RING order, the sub-sampled pixels are those of the RING map."""
    ring = lcn.AngularLightconer.healpix(
        nside=4, subsample_level=1, lc_redshifts=np.linspace(6, 7, 5)
    )
    nested = lcn.AngularLightconer.healpix(
        nside=4, order="nested", subsample_level=1, lc_redshifts=np.linspace(6, 7, 5)
    )
    ring_to_nested = HEALPix(nside=4, order="ring").ring_to_nested(np.arange(192))

    np.testing.assert_allclose(ring.latitude, nested.latitude[:, ring_to_nested])
    np.testing.assert_allclose(ring.longitude, nested.longitude[:, ring_to_nested])
