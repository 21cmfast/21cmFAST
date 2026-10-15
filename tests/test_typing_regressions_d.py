"""Regression tests for bugs found while type-annotating the coeval/lightcone drivers.

Covers ``drivers/coeval.py``, ``drivers/lightcone.py``, ``lightconers.py`` and
``rsds.py``.
"""

from dataclasses import dataclass

import h5py
import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology import Planck15
from astropy_healpix import HEALPix

from py21cmfast import (
    CosmoParams,
    InputParameters,
    LightCone,
    OutputCache,
    SimulationOptions,
)
from py21cmfast import lightconers as lcn
from py21cmfast.drivers import coeval as coeval_module
from py21cmfast.drivers import lightcone as lightcone_module


@dataclass
class _VelocityCoeval:
    """A mock coeval with a brightness temperature and 3D velocities."""

    redshift: float
    brightness_temp: np.ndarray
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    velocity_z: np.ndarray
    simulation_options: SimulationOptions
    cosmo_params: CosmoParams


def _velocity_coeval(redshift: float, fill: float) -> _VelocityCoeval:
    simopts = SimulationOptions(BOX_LEN=100, HII_DIM=50)
    shape = (simopts.HII_DIM,) * 3
    rng = np.random.default_rng(int(redshift))
    return _VelocityCoeval(
        redshift=redshift,
        brightness_temp=fill * np.ones(shape),
        velocity_x=rng.normal(size=shape),
        velocity_y=rng.normal(size=shape),
        velocity_z=rng.normal(size=shape),
        simulation_options=simopts,
        cosmo_params=CosmoParams(),
    )


def _healpix_lonlat() -> tuple[np.ndarray, np.ndarray]:
    hp = HEALPix(nside=4)
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    return lon.radian, lat.radian


def test_like_rectilinear_uses_given_cosmology():
    """The cosmology passed to like_rectilinear must be the lightconer's cosmology."""
    simopts = SimulationOptions(BOX_LEN=100, HII_DIM=20)
    lc = lcn.AngularLightconer.like_rectilinear(
        simulation_options=simopts,
        match_at_z=6.0,
        max_redshift=7.0,
        cosmo=Planck15,
    )
    assert lc.cosmo is Planck15
    # The slices should be spaced by the cell size, in the given cosmology.
    assert np.isclose(lc.lc_distances.min(), Planck15.comoving_distance(6.0), rtol=1e-6)


def test_angular_lightconer_los_velocity_slices():
    """The angular LoS-velocity slices can be constructed.

    Covers passing plain floats (in pixels) to cosmotile's interpolator.
    """
    lon, lat = _healpix_lonlat()
    lcner = lcn.AngularLightconer.between_redshifts(
        latitude=lat,
        longitude=lon,
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2 * un.Mpc,
        quantities=("brightness_temp", "los_velocity"),
    )
    slices = list(
        lcner.make_lightcone_slices(
            _velocity_coeval(6.0, 0.0), _velocity_coeval(7.0, 1.0)
        )
    )
    vel_slices = [s for s in slices if s[0] == "los_velocity"]
    bt_slices = [s for s in slices if s[0] == "brightness_temp"]
    assert len(vel_slices) == len(bt_slices) > 0
    for _, _, slc in vel_slices:
        assert slc.shape == lon.shape
        assert np.all(np.isfinite(slc))


def test_los_velocity_computed_when_listed_first():
    """LoS velocity slices are produced even if it is the first listed quantity."""
    lcner = lcn.RectilinearLightconer.between_redshifts(
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2.0 * un.Mpc,
        quantities=("los_velocity", "brightness_temp"),
    )
    slices = list(
        lcner.make_lightcone_slices(
            _velocity_coeval(6.0, 0.0), _velocity_coeval(7.0, 1.0)
        )
    )
    names = [s[0] for s in slices]
    assert names.count("los_velocity") == names.count("brightness_temp") > 0


class _StopAtRedshiftsError(Exception):
    def __init__(self, user_redshifts):
        super().__init__()
        self.user_redshifts = user_redshifts


def test_generate_coeval_accepts_array_out_redshifts(monkeypatch, tmp_path):
    """An array of out_redshifts must not raise 'truth value of an array'."""

    def _stop(inputs, user_redshifts):
        raise _StopAtRedshiftsError(user_redshifts)

    monkeypatch.setattr(coeval_module, "_get_required_redshifts_coeval", _stop)
    inputs = InputParameters(random_seed=1)

    with pytest.raises(_StopAtRedshiftsError) as exc:
        next(
            coeval_module.generate_coeval(
                inputs=inputs,
                out_redshifts=np.array([8.0, 7.0]),
                cache=OutputCache(tmp_path),
            )
        )
    assert exc.value.user_redshifts == [8.0, 7.0]

    # A scalar is also accepted, and wrapped into a list.
    with pytest.raises(_StopAtRedshiftsError) as exc:
        next(
            coeval_module.generate_coeval(
                inputs=inputs, out_redshifts=7.0, cache=OutputCache(tmp_path)
            )
        )
    assert exc.value.user_redshifts == [7.0]


def test_obtain_starting_point_without_cache():
    """Without a cache there is nothing to resume from (instead of an AttributeError)."""
    inputs = InputParameters.from_template(
        "simple", random_seed=1, node_redshifts=[8.0, 7.0, 6.0]
    )
    assert not inputs.matter_options.has_discrete_halos

    idx, coeval = coeval_module._obtain_starting_point_for_scrolling(
        inputs=inputs,
        initial_conditions=None,
        photon_nonconservation_data={},
        cache=None,
        regenerate=False,
    )
    assert idx == -1
    assert coeval is None


def _small_lightcone(**kwargs) -> LightCone:
    inputs = InputParameters(random_seed=1, node_redshifts=[8.0, 7.0, 6.0])
    dist = np.linspace(6000, 6100, 5) * un.Mpc
    return LightCone(
        lightcone_distances=dist,
        inputs=inputs,
        lightcones={"brightness_temp": np.zeros((3, 3, len(dist)))},
        **kwargs,
    )


def test_lightcone_without_global_quantities_save_and_compare(tmp_path):
    """A LightCone with the default (None) global quantities can be saved/compared."""
    lc = _small_lightcone()
    assert lc.global_quantities is None
    same = lc
    assert lc == same  # this used to raise AttributeError
    assert lc == _small_lightcone(global_quantities={})

    path = tmp_path / "lc.h5"
    lc.save(path)  # this used to raise AttributeError
    lc.make_checkpoint(path, lcidx=2, node_index=0)  # as did this

    with h5py.File(path, "r") as fl:
        assert len(fl["global_quantities"]) == 0
        assert fl.attrs["last_completed_lcidx"] == 2


def test_full_lightcone_returns_early(monkeypatch):
    """A lightcone that is already complete is yielded once without further work."""

    @dataclass
    class _PF:
        redshift: float

    perturbed_fields = [_PF(8.0), _PF(7.0), _PF(6.0)]
    full_lc = _small_lightcone(last_completed_node=len(perturbed_fields) - 1)

    monkeypatch.setattr(
        lightcone_module, "setup_lightcone_instance", lambda **kw: full_lc
    )

    def _fail(**kw):
        raise AssertionError("should not look for a starting point")

    monkeypatch.setattr(lightcone_module, "_obtain_starting_point_for_scrolling", _fail)

    out = list(
        lightcone_module._run_lightcone_from_perturbed_fields(
            initial_conditions=None,
            perturbed_fields=perturbed_fields,
            lightconer=None,
            inputs=full_lc.inputs,
            lc_distances=full_lc.lightcone_distances,
            photon_nonconservation_data={},
            include_dvdr_in_tau21=False,
            apply_rsds=False,
            n_rsd_subcells=4,
            halofield_list=[],
        )
    )
    assert out == [(None, None, None, full_lc)]
