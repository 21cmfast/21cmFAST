"""Test of the Lightconer subclasses."""

import re
from dataclasses import dataclass
from types import SimpleNamespace

import deprecation
import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology import Planck18, z_at_value
from astropy_healpix import HEALPix
from scipy.spatial.transform import Rotation

from py21cmfast import AstroOptions, CosmoParams, InputParameters, SimulationOptions
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
        astro_options=AstroOptions(),
    )


def test_equality(
    rect_lcner: lcn.RectilinearLightconer,
):
    assert rect_lcner == rect_lcner  # noqa: PLR0124  # reflexivity of __eq__


@pytest.mark.parametrize("lc", ["rect_lcner", "ang_lcner"])
def test_uniform_coevals(request, lc):
    """Test that uniform boxes interpolate the way we want."""
    lc = request.getfixturevalue(lc)
    z6 = get_uniform_coeval(redshift=6.0, fill=0)
    z7 = get_uniform_coeval(redshift=7.0, fill=1.0)

    _q, _idx, out = next(lc.make_lightcone_slices(z6, z7))

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


def test_with_equal_cdist_slices_deprecated_warning():
    """Test that with_equal_cdist_slices raises a deprecation warning."""
    with pytest.warns(
        deprecation.DeprecatedWarning, match="with_equal_cdist_slices is deprecated"
    ):
        lcn.RectilinearLightconer.with_equal_cdist_slices(
            min_redshift=6.0,
            max_redshift=7.0,
            resolution=2 * un.Mpc,
        )


@deprecation.fail_if_not_removed
def test_with_equal_cdist_slices_is_removed():
    """Fails when removed_in version is reached, reminding you to delete with_equal_cdist_slices."""
    lcn.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2 * un.Mpc,
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
    inputs.evolve_input_structs(KEEP_3D_VELOCITIES=False)
    with pytest.raises(
        ValueError,
        match="To account for RSDs or velocity corrections in an angular lightcone, you need to set",
    ):
        ang_lcner.validate_options(
            inputs=inputs, include_dvdr_in_tau21=True, apply_rsds=True
        )


# ======================================================================================
# Angular lightcones: sampling the coevals first gives the same slices
# ======================================================================================


def _random_coevals(seed: int = 1, hii_dim: int = 24, box_len: float = 48.0):
    """Two coeval-like objects at z = 6.0 and 6.2, with random fields.

    ``z_reion`` is -1 in "neutral" cells and positive elsewhere, with more neutral
    cells at the higher redshift, so that its "mean_max" rule matters.
    """
    rng = np.random.default_rng(seed)
    so = SimulationOptions(BOX_LEN=box_len, HII_DIM=hii_dim)
    shape = (hii_dim,) * 3

    def coeval(redshift, neutral):
        fields = {
            name: rng.normal(size=shape).astype(np.float32)
            for name in (
                "brightness_temp",
                "density",
                "tau_21",
                "velocity_x",
                "velocity_y",
                "velocity_z",
            )
        }
        fields["z_reion"] = np.where(
            rng.uniform(size=shape) < neutral, -1, rng.uniform(6, 12, size=shape)
        ).astype(np.float32)
        return SimpleNamespace(
            redshift=redshift,
            simulation_options=so,
            cosmo_params=CosmoParams(),
            **fields,
        )

    return coeval(6.0, 0.3), coeval(6.2, 0.6)


def _angular_lightconer(interpolation_order: int, nside: int = 8, **kwargs):
    hp = HEALPix(nside=nside)
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    return lcn.AngularLightconer.between_redshifts(
        min_redshift=6.0,
        max_redshift=6.2,
        resolution=2 * un.Mpc,
        latitude=lat.radian,
        longitude=lon.radian,
        interpolation_order=interpolation_order,
        origin=np.array([12.0, 12.0, 12.0]) * un.pixel,
        quantities=("brightness_temp", "density", "z_reion", "tau_21", "los_velocity"),
        **kwargs,
    )


def _full_box_slices(lightconer, c1, c2):
    """Make the slices by interpolating the whole boxes in distance, then sampling.

    This is how ``AngularLightconer`` made them before it sampled the coevals first.
    """
    for q, idx, new in lightconer.make_lightcone_slices(c1, c2):
        cosmo = c1.cosmo_params.cosmo
        eq = un.pixel_scale(c1.simulation_options.cell_size / un.pixel)
        dc1 = cosmo.comoving_distance(c1.redshift).to(un.pixel, eq)
        dc2 = cosmo.comoving_distance(c2.redshift).to(un.pixel, eq)
        lcd = lightconer.get_lc_distances_in_pixels(c1.simulation_options.cell_size)[
            idx
        ]
        if q == "los_velocity":
            boxes = [
                lightconer.redshift_interpolation(
                    lcd,
                    getattr(c1, f"velocity_{ax}"),
                    getattr(c2, f"velocity_{ax}"),
                    dc1,
                    dc2,
                )
                for ax in "xyz"
            ]
            old = lightconer.construct_los_velocity_lightcone(lcd, boxes)
        else:
            box = lightconer.redshift_interpolation(
                lcd,
                getattr(c1, q),
                getattr(c2, q),
                dc1,
                dc2,
                kind=lightconer.interp_kinds.get(q, "mean"),
            )
            old = lightconer.construct_lightcone(lcd, box)
        yield q, idx, np.asarray(new), np.asarray(old)


@pytest.mark.parametrize("tile_first", [True, False, None])
@pytest.mark.parametrize("interpolation_order", [0, 1, 3])
def test_angular_slices_sample_first(interpolation_order, tile_first):
    """Every quantity, los_velocity included, gives the same slices either way."""
    c1, c2 = _random_coevals()
    lightconer = _angular_lightconer(interpolation_order, tile_first=tile_first)

    seen = set()
    for q, idx, new, old in _full_box_slices(lightconer, c1, c2):
        assert new.shape == (len(lightconer.longitude),)
        np.testing.assert_allclose(
            new, old, rtol=0, atol=2e-6 * np.abs(old).max(), err_msg=f"{q}, {idx}"
        )
        seen.add(q)
    assert seen == set(lightconer.quantities)


def test_angular_mean_max_needs_cells():
    """Applying z_reion's "mean_max" rule to samples would not give the same slice.

    So the test above does check the eight-cell evaluation of the rule.
    """
    c1, c2 = _random_coevals()
    lightconer = _angular_lightconer(interpolation_order=1, tile_first=True)
    cosmo = c1.cosmo_params.cosmo
    eq = un.pixel_scale(c1.simulation_options.cell_size / un.pixel)
    dc1 = cosmo.comoving_distance(c1.redshift).to(un.pixel, eq)
    dc2 = cosmo.comoving_distance(c2.redshift).to(un.pixel, eq)
    lcd = lightconer.get_lc_distances_in_pixels(c1.simulation_options.cell_size)[1]

    interp = lightconer._get_interpolator(lcd)
    exact = lightconer.interpolate_slice(
        lcd, c1.z_reion, c2.z_reion, dc1, dc2, None, kind="mean_max"
    )
    on_samples = lightconer.redshift_interpolation(
        lcd, interp(c1.z_reion), interp(c2.z_reion), dc1, dc2, kind="mean_max"
    )
    assert not np.allclose(exact, on_samples)


def test_angular_mean_max_subsamples():
    """The "mean_max" rule also works for pixels averaged over sub-samples."""
    cosmotile = pytest.importorskip("cosmotile")
    if not hasattr(cosmotile, "healpix_subpixel_lonlat"):
        pytest.skip("sub-sampled pixels need cosmotile >= 0.3")

    c1, c2 = _random_coevals()
    lightconer = _angular_lightconer(interpolation_order=1)
    lat, lon = cosmotile.healpix_subpixel_lonlat(nside=4, subsample_level=1)
    interp = cosmotile.make_lightcone_slice_interpolator(
        latitude=lat,
        longitude=lon,
        distance_to_shell=600.0,
        origin=np.array([12.0, 12.0, 12.0]),
    )

    out = lightconer._trilinear_mean_max(
        interp, 600.0, c1.z_reion, c2.z_reion, 590, 610
    )
    box = lightconer.redshift_interpolation(
        600.0, c1.z_reion, c2.z_reion, 590, 610, kind="mean_max"
    )
    assert out.shape == (lat.shape[1],)
    np.testing.assert_allclose(out, interp(box), rtol=1e-12, atol=1e-12)


def _interpolator_with(n_directions: int) -> SimpleNamespace:
    """Stand in for a cosmotile interpolator onto ``n_directions`` directions."""
    return SimpleNamespace(
        keywords={"coordinates": np.broadcast_to(0.0, (3, n_directions))}
    )


def test_angular_samples_first_choice():
    """The cheaper way is chosen, unless one is forced or the order needs the box."""
    lightconer = _angular_lightconer(interpolation_order=1)

    # A big box seen through few directions (the EOS26 ring): sample first.
    assert lightconer.samples_first(1500**3, _interpolator_with(2**15))
    # A HEALPix map of Nside 1024 has about as many directions as a 228^3 box has
    # cells: interpolate the box first.
    assert not lightconer.samples_first(228**3, _interpolator_with(12 * 1024**2))

    small_box, many = 64**3, _interpolator_with(10**6)
    assert not lightconer.samples_first(small_box, many)
    forced = _angular_lightconer(interpolation_order=1, tile_first=True)
    assert forced.samples_first(small_box, many)
    forced = _angular_lightconer(interpolation_order=0, tile_first=False)
    assert not forced.samples_first(1500**3, _interpolator_with(10))

    # Spline orders pre-filter the whole box, so they never sample first.
    spline = _angular_lightconer(interpolation_order=3, tile_first=True)
    assert not spline.samples_first(1500**3, _interpolator_with(10))


@pytest.mark.parametrize("tile_first", [True, False])
def test_angular_slices_use_the_chosen_way(tile_first, monkeypatch):
    """Interpolating the boxes first goes through Lightconer.interpolate_slice."""
    calls = []
    base = lcn.Lightconer.interpolate_slice

    def counting(self, *args, **kwargs):
        calls.append(args[0])
        return base(self, *args, **kwargs)

    monkeypatch.setattr(lcn.Lightconer, "interpolate_slice", counting)
    c1, c2 = _random_coevals()
    lightconer = _angular_lightconer(interpolation_order=1, tile_first=tile_first)
    n_slices = sum(
        q == "density" for q, _, _ in lightconer.make_lightcone_slices(c1, c2)
    )

    assert n_slices > 0
    assert len(calls) == (0 if tile_first else 4 * n_slices)
