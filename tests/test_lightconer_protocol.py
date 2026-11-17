"""Tests of the lightconer protocol, and of the lightcone driver's use of it.

The lightconers defined here are written against the protocol only (they do not
inherit from anything in 21cmFAST), to check that the driver assumes nothing more.
"""

import dataclasses
import sys
from collections import deque
from types import SimpleNamespace

import attrs
import deprecation
import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology import FLRW

import py21cmfast as p21c
from py21cmfast import lightconers as lcn
from py21cmfast.drivers.lightcone import LightCone
from py21cmfast.lightconers import (
    Chunk,
    CoevalSnapshot,
    LightconeContext,
    LightconerProtocol,
    LightconeSession,
    Snapshot,
    prepare_lightconer,
)


# ======================================================================================
# A lightconer that needs more than two snapshots
# ======================================================================================
@dataclasses.dataclass(frozen=True)
class QuadraticLightconer:
    """A rectilinear lightconer that interpolates quadratically in distance.

    Each slice is made from *three* consecutive snapshots, so the slices between two
    snapshots can only be made when the next one arrives (or at the end of the run).
    Each slice is returned in two parts, to check that the driver does not assume
    that a chunk is a whole slice.
    """

    lc_distances: un.Quantity
    cosmo: FLRW
    quantities: tuple[str, ...] = ("brightness_temp",)
    #: How each slice is split in two chunks: by "rows" (slices), or by a "mask".
    split: str = "rows"

    def evolve(self, **changes):
        """Return a copy with some attributes changed."""
        return dataclasses.replace(self, **changes)

    def required_fields(self):
        """Return the fields the quantities are made from."""
        return tuple(
            "velocity_z" if q == "los_velocity" else q for q in self.quantities
        )

    def start(self, context):
        """Start a run."""
        return QuadraticSession(self, context)


class QuadraticSession:
    """The making of a lightcone by a :class:`QuadraticLightconer`."""

    # After snapshot k, the slices between snapshots k-1 and k are still to be made,
    # from snapshots k-1, k and k+1: a new session needs the last two to carry on.
    resume_depth = 2

    def __init__(self, lightconer, context):
        self.lightconer = lightconer
        self.context = context
        self.shape = (*context.box_shape[:2], len(lightconer.lc_distances))
        self.distances = lightconer.lc_distances.to_value(un.Mpc)
        # The last three snapshots: (distance, {field: box}).
        self.history = deque(maxlen=3)
        self.n_ingested = 0

    def ingest(self, snapshot):
        """Make the slices between the two snapshots before this one."""
        self.n_ingested += 1
        distance = self.context.cosmo.comoving_distance(snapshot.redshift)
        # Get the boxes now: the driver may release the simulation's memory later.
        boxes = {f: snapshot[f] for f in self.lightconer.required_fields()}
        self.history.append((distance.to_value(un.Mpc), boxes))
        if len(self.history) == 3:
            # The slices between the two older snapshots.
            yield from self._make(self.history[1][0], self.history[0][0])

    def finish(self):
        """Make the slices between the last two snapshots."""
        if len(self.history) >= 2:
            yield from self._make(self.history[-1][0], self.history[-2][0])

    def _make(self, dmin, dmax):
        nodes = np.array([d for d, _ in self.history])
        index = np.nonzero(
            (self.distances >= dmin * (1 - 1e-6)) & (self.distances < dmax)
        )[0]
        for i in index:
            d = self.distances[i]
            # Lagrange weights of the snapshots, at the distance of the slice.
            weights = [
                np.prod([(d - m) / (n - m) for m in nodes if m != n]) for n in nodes
            ]
            for quantity, field in zip(
                self.lightconer.quantities,
                self.lightconer.required_fields(),
                strict=True,
            ):
                values = sum(
                    w * boxes[field].take(i, axis=2, mode="wrap")
                    for w, (_, boxes) in zip(weights, self.history, strict=True)
                )
                if self.lightconer.split == "rows":
                    yield Chunk(quantity, (slice(0, None, 2), ..., int(i)), values[::2])
                    yield Chunk(
                        quantity, (slice(1, None, 2), ..., int(i)), values[1::2]
                    )
                else:
                    mask = np.zeros(values.shape, dtype=bool)
                    mask[::2, 1::3] = True
                    yield Chunk(quantity, (mask, int(i)), values[mask])
                    yield Chunk(quantity, (~mask, int(i)), values[~mask])


def _fake_snapshots(redshifts, context, coefficients=(3.0, -2e-3, 4e-7), seed=0):
    """Snapshots of a field that is quadratic in distance (times a random box)."""
    rng = np.random.default_rng(seed)
    pattern = rng.uniform(1, 2, size=context.box_shape)
    a, b, c = coefficients

    def amplitude(distance):
        return a + b * distance + c * distance**2

    snapshots = [
        SimpleNamespace(
            redshift=z,
            brightness_temp=pattern
            * amplitude(context.cosmo.comoving_distance(z).to_value(un.Mpc)),
        )
        for z in redshifts
    ]
    return [CoevalSnapshot(s) for s in snapshots], pattern, amplitude


@pytest.fixture(scope="module")
def small_context(default_cosmo_params):
    return LightconeContext(
        cosmo=default_cosmo_params.cosmo,
        box_shape=(6, 6, 6),
        cell_size=2 * un.Mpc,
        node_redshifts=(9.0, 8.5, 8.0, 7.5, 7.0),
    )


def _fill(lightconer, snapshots, context):
    """Run the loop of the driver, without a simulation."""
    session = lightconer.start(context)
    out = {q: np.full(session.shape, np.nan) for q in lightconer.quantities}
    counts = np.zeros(session.shape, dtype=int)
    for snapshot in snapshots:
        for quantity, index, values in session.ingest(snapshot):
            out[quantity][index] = values
            counts[index] += 1
    for quantity, index, values in session.finish():
        out[quantity][index] = values
        counts[index] += 1
    return out, counts


@pytest.mark.parametrize("split", ["rows", "mask"])
def test_quadratic_lightconer_is_exact(small_context, split):
    """A lightconer that uses three snapshots per slice works in the driver's loop."""
    cosmo = small_context.cosmo
    snapshots, pattern, amplitude = _fake_snapshots(
        small_context.node_redshifts, small_context
    )
    distances = (
        np.arange(
            cosmo.comoving_distance(7.0).to_value(un.Mpc),
            cosmo.comoving_distance(9.0).to_value(un.Mpc),
            2.0,
        )
        * un.Mpc
    )
    quad = QuadraticLightconer(lc_distances=distances, cosmo=cosmo, split=split)
    assert isinstance(quad, LightconerProtocol)
    assert isinstance(quad.start(small_context), LightconeSession)

    out, counts = _fill(quad, snapshots, small_context)

    # Every cell of the lightcone was returned exactly once.
    assert np.all(counts == 1)
    index = np.arange(len(distances)) % 6
    expected = pattern[:, :, index] * amplitude(distances.value)
    np.testing.assert_allclose(out["brightness_temp"], expected, rtol=1e-10)

    # The lightconer of 21cmFAST interpolates linearly, which is not exact here.
    rect = p21c.RectilinearLightconer(lc_distances=distances, cosmo=cosmo)
    linear, counts = _fill(rect, snapshots, small_context)
    assert np.all(counts == 1)
    assert not np.allclose(linear["brightness_temp"], out["brightness_temp"], rtol=1e-6)
    np.testing.assert_allclose(
        linear["brightness_temp"], out["brightness_temp"], rtol=1e-2
    )


# ======================================================================================
# The objects of the protocol
# ======================================================================================
def test_rectilinear_follows_the_protocol(rectlcn, default_input_struct_lc):
    assert isinstance(rectlcn, LightconerProtocol)
    assert rectlcn.required_fields() == ("brightness_temp",)

    context = lcn.context_from_inputs(default_input_struct_lc)
    assert context.box_shape == (35, 35, 35)
    assert context.cell_size == default_input_struct_lc.simulation_options.cell_size
    assert context.node_redshifts == tuple(default_input_struct_lc.node_redshifts)

    session = rectlcn.start(context)
    assert isinstance(session, LightconeSession)
    assert session.shape == (35, 35, len(rectlcn.lc_distances))
    assert session.resume_depth == 1


def test_evolve_and_required_fields(rectlcn):
    more = rectlcn.evolve(quantities=("density", "los_velocity"))
    assert more.quantities == ("density", "los_velocity")
    assert more.required_fields() == ("density", "velocity_z")
    assert rectlcn.quantities == ("brightness_temp",)

    longer = rectlcn.evolve(lc_distances=np.linspace(4000, 5000, 11) * un.Mpc)
    assert len(longer.lc_distances) == 11
    # The redshifts follow the distances.
    assert len(longer.lc_redshifts) == 11


def test_coeval_snapshot():
    calls = []

    class FakeCoeval:
        redshift = 7.0

        def __getattr__(self, name):
            if name != "density":
                raise AttributeError(name)
            calls.append(name)
            return np.ones((2, 2, 2))

    snapshot = CoevalSnapshot(FakeCoeval())
    assert isinstance(snapshot, Snapshot)
    assert snapshot.redshift == 7.0
    assert not calls  # nothing is got before it is asked for
    assert snapshot["density"] is snapshot["density"]
    assert calls == ["density"]
    with pytest.raises(KeyError, match="'nope' is not available"):
        snapshot["nope"]


def test_non_cubic_box_shape():
    so = p21c.SimulationOptions(HII_DIM=10, DIM=20, BOX_LEN=20, NON_CUBIC_FACTOR=1.5)
    inputs = p21c.InputParameters(
        random_seed=1, simulation_options=so, node_redshifts=(8.0, 7.0)
    )
    assert lcn.context_from_inputs(inputs).box_shape == (10, 10, 15)


# ======================================================================================
# The driver
# ======================================================================================
@pytest.fixture(scope="module")
def quad_lcn(rectlcn):
    return QuadraticLightconer(lc_distances=rectlcn.lc_distances, cosmo=rectlcn.cosmo)


@pytest.fixture(scope="module")
def quad_lc(quad_lcn, ic, cache, default_input_struct_lc, lc):
    # ``lc`` is requested so that the boxes are in the cache already.
    return p21c.run_lightcone(
        lightconer=quad_lcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        cache=cache,
        regenerate=False,
        include_dvdr_in_tau21=False,
    )


def test_custom_lightconer_in_run_lightcone(
    quad_lcn, quad_lc, lc, default_input_struct_lc, cache
):
    """run_lightcone works with a lightconer that is not one of 21cmFAST's."""
    from py21cmfast.io.caching import RunCache

    assert isinstance(quad_lc, LightCone)
    quad = quad_lc.lightcones["brightness_temp"]
    linear = lc.lightcones["brightness_temp"]
    assert quad.shape == linear.shape
    assert np.all(quad != 0)
    # Quadratic and linear interpolation are not the same.
    assert not np.allclose(quad, linear, rtol=1e-4)

    # The lightcone is what the lightconer makes of the boxes of the run.
    inputs = default_input_struct_lc
    run = RunCache.from_inputs(inputs, cache)
    snapshots = (
        CoevalSnapshot(run.get_coeval_at_z(index=i))
        for i in range(len(inputs.node_redshifts))
    )
    expected, counts = _fill(quad_lcn, snapshots, lcn.context_from_inputs(inputs))
    assert np.all(counts == 1)
    np.testing.assert_array_equal(quad, expected["brightness_temp"].astype(np.float32))

    for name, values in lc.global_quantities.items():
        np.testing.assert_allclose(quad_lc.global_quantities[name], values, rtol=1e-6)


@pytest.mark.parametrize("split", ["rows", "mask"])
def test_custom_lightconer_resume(
    quad_lcn, quad_lc, default_input_struct_lc, tmpdirec, cache, split
):
    """A run is resumed from a checkpoint by a lightconer with ``resume_depth`` 2."""
    quad_lcn = quad_lcn.evolve(split=split)
    fname = tmpdirec / f"lightcone_quadratic_{split}.h5"
    kw = {
        "lightconer": quad_lcn,
        "inputs": default_input_struct_lc,
        "lightcone_filename": fname,
        "cache": cache,
        "regenerate": False,
        "include_dvdr_in_tau21": False,
    }

    z = np.inf
    gen = p21c.generate_lightcone(**kw)
    while z > 20.0:
        iz_1, z, _, partial = next(gen)
    gen.close()
    assert (
        1 < partial._last_completed_node < len(default_input_struct_lc.node_redshifts)
    )

    # What is in the file is what was made so far, and no more.
    on_disk = LightCone.from_file(fname).lightcones["brightness_temp"]
    done = on_disk != 0
    assert 0 < done.sum() < done.size
    np.testing.assert_array_equal(
        on_disk[done], quad_lc.lightcones["brightness_temp"][done]
    )

    for iz_2, z, _, finished in p21c.generate_lightcone(**kw):  # noqa: B007
        assert iz_2 > iz_1

    np.testing.assert_array_equal(
        finished.lightcones["brightness_temp"], quad_lc.lightcones["brightness_temp"]
    )
    np.testing.assert_array_equal(
        LightCone.from_file(fname).lightcones["brightness_temp"],
        quad_lc.lightcones["brightness_temp"],
    )
    np.testing.assert_allclose(
        finished.global_quantities["brightness_temp"],
        quad_lc.global_quantities["brightness_temp"],
        rtol=1e-6,
    )


def test_resume_with_another_lightconer(
    rectlcn, default_input_struct_lc, tmpdirec, cache, lc
):
    """A checkpoint cannot be resumed by a lightconer of another shape."""
    fname = tmpdirec / "lightcone_other_shape.h5"
    kw = {
        "inputs": default_input_struct_lc,
        "lightcone_filename": fname,
        "cache": cache,
        "regenerate": False,
        "include_dvdr_in_tau21": False,
    }
    gen = p21c.generate_lightcone(lightconer=rectlcn, **kw)
    for _ in range(3):
        next(gen)
    gen.close()

    shorter = rectlcn.evolve(lc_distances=rectlcn.lc_distances[:-4])
    with pytest.raises(ValueError, match="but the lightconer makes lightcones of"):
        p21c.run_lightcone(lightconer=shorter, **kw)

    other = rectlcn.evolve(quantities=("density",))
    with pytest.raises(ValueError, match="has no 'density' lightcone"):
        p21c.run_lightcone(lightconer=other, **kw)


def test_unavailable_field_fails_before_the_run(quad_lcn, default_input_struct_lc):
    lightconer = quad_lcn.evolve(quantities=("spin_temperature",))
    with pytest.raises(ValueError, match="spin_temperature"):
        p21c.run_lightcone(lightconer=lightconer, inputs=default_input_struct_lc)


# ======================================================================================
# What the driver does to a lightconer before a run
# ======================================================================================
def test_prepare_lightconer_adds_quantities(
    quad_lcn, default_input_struct_lc, default_input_struct_ts, lightcone_min_redshift
):
    same = prepare_lightconer(
        quad_lcn,
        inputs=default_input_struct_lc,
        include_dvdr_in_tau21=False,
        apply_rsds=False,
    )
    assert same is quad_lcn

    with_velocity = prepare_lightconer(
        quad_lcn,
        inputs=default_input_struct_lc,
        include_dvdr_in_tau21=True,
        apply_rsds=False,
    )
    assert with_velocity.quantities == ("brightness_temp", "los_velocity")
    assert len(with_velocity.lc_distances) == len(quad_lcn.lc_distances)

    inputs_ts = default_input_struct_ts.clone(
        node_redshifts=default_input_struct_lc.node_redshifts
    )
    with_tau = prepare_lightconer(
        quad_lcn, inputs=inputs_ts, include_dvdr_in_tau21=True, apply_rsds=False
    )
    assert with_tau.quantities == ("brightness_temp", "tau_21", "los_velocity")


def test_prepare_lightconer_checks(quad_lcn, default_input_struct_lc):
    kw = {"include_dvdr_in_tau21": False, "apply_rsds": False}
    with pytest.raises(ValueError, match="no node_redshifts"):
        prepare_lightconer(
            quad_lcn, inputs=default_input_struct_lc.clone(node_redshifts=()), **kw
        )

    other_cosmo = quad_lcn.evolve(cosmo=quad_lcn.cosmo.clone(H0=60))
    with pytest.raises(ValueError, match="The cosmology used to construct"):
        prepare_lightconer(other_cosmo, inputs=default_input_struct_lc, **kw)

    too_far = quad_lcn.evolve(lc_distances=quad_lcn.lc_distances * 1.5)
    with pytest.raises(ValueError, match="The lightcone redshifts are not compatible"):
        prepare_lightconer(too_far, inputs=default_input_struct_lc, **kw)


def test_prepare_lightconer_extends_for_rsds(default_input_struct_lc):
    """With RSDs, the lightcone is extended on both sides (and trimmed afterwards)."""
    inputs = default_input_struct_lc
    cosmo = inputs.cosmo_params.cosmo
    # Leave room on both sides of the lightcone for the buffer.
    zs = sorted(inputs.node_redshifts)
    kw = {
        "min_redshift": zs[2],
        "max_redshift": zs[-3],
        "resolution": inputs.simulation_options.cell_size,
    }
    rect = p21c.RectilinearLightconer.between_redshifts(cosmo=cosmo, **kw)
    lcd = rect.lc_distances
    quad = QuadraticLightconer(lc_distances=lcd, cosmo=cosmo)

    out = {}
    for name, lightconer in (("rect", rect), ("quad", quad)):
        new = prepare_lightconer(
            lightconer, inputs=inputs, include_dvdr_in_tau21=False, apply_rsds=True
        )
        assert type(new) is type(lightconer)
        assert new.quantities == ("brightness_temp", "los_velocity")
        assert new.lc_distances.min() < lcd.min()
        assert new.lc_distances.max() > lcd.max()
        out[name] = new.lc_distances

    # A lightconer that does not say how its line-of-sight velocity compares to the
    # full velocity (los_velocity_rms_factor) gets the larger, safe, buffer.
    assert out["quad"].min() <= out["rect"].min()
    assert out["quad"].max() >= out["rect"].max()


# ======================================================================================
# Lightconers written for v4.3
# ======================================================================================
@pytest.fixture(scope="module")
def old_style_class():
    """The RectilinearLightconer of v4.3: a subclass of ``Lightconer``."""
    with pytest.warns(
        deprecation.DeprecatedWarning, match="OldRectilinear is a subclass"
    ):

        @attrs.define(kw_only=True, slots=False)
        class OldRectilinear(lcn.Lightconer):
            index_offset: int = attrs.field()

            @index_offset.default
            def _index_offset_default(self) -> int:
                return len(self.lc_distances)

            def coeval_subselect(self, lcd, coeval, coeval_res):
                lcpix = self.get_lc_distances_in_pixels(coeval_res)
                lcidx = int((lcpix.max() - lcd + 1 * un.pixel).to_value(un.pixel))
                return coeval.take(-lcidx + self.index_offset, axis=2, mode="wrap")

            def construct_lightcone(self, lcd, box):
                return box

            def construct_los_velocity_lightcone(self, lcd, velocity):
                return velocity

            def get_shape(self, simulation_options):
                return (
                    simulation_options.HII_DIM,
                    simulation_options.HII_DIM,
                    len(self.lc_distances),
                )

    return OldRectilinear


def test_old_style_lightconer_still_works(
    old_style_class, rectlcn, ic, cache, default_input_struct_lc, lc
):
    old = old_style_class(lc_distances=rectlcn.lc_distances, cosmo=rectlcn.cosmo)
    assert isinstance(old, LightconerProtocol)
    assert lcn._LIGHTCONERS["OldRectilinear"] is old_style_class

    old_lc = p21c.run_lightcone(
        lightconer=old,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        cache=cache,
        regenerate=False,
        include_dvdr_in_tau21=False,
    )
    # The lightconer of 21cmFAST gives the very same lightcone.
    np.testing.assert_array_equal(
        old_lc.lightcones["brightness_temp"], lc.lightcones["brightness_temp"]
    )


def test_old_style_lightconer_with_velocities(
    old_style_class, rectlcn, ic, cache, default_input_struct_lc
):
    """The velocity corrections give the same with the old and the new lightconer."""
    old = old_style_class(lc_distances=rectlcn.lc_distances, cosmo=rectlcn.cosmo)
    kw = {
        "initial_conditions": ic,
        "inputs": default_input_struct_lc,
        "cache": cache,
        "regenerate": False,
        "include_dvdr_in_tau21": True,
    }
    old_lc = p21c.run_lightcone(lightconer=old, **kw)
    new_lc = p21c.run_lightcone(lightconer=rectlcn, **kw)
    assert set(old_lc.lightcones) == {"brightness_temp", "los_velocity"}
    for name, values in new_lc.lightcones.items():
        np.testing.assert_array_equal(old_lc.lightcones[name], values, err_msg=name)


def test_old_style_validate_options_is_used(
    old_style_class, rectlcn, default_input_struct_lc
):
    """A v4.3 lightconer that overrides validate_options has it called by the driver."""
    with pytest.warns(deprecation.DeprecatedWarning, match="Picky is a subclass"):

        @attrs.define(kw_only=True, slots=False)
        class Picky(old_style_class):
            def validate_options(self, inputs, include_dvdr_in_tau21, apply_rsds, **kw):
                if include_dvdr_in_tau21:
                    raise ValueError("picky about velocities")
                return super().validate_options(
                    inputs, include_dvdr_in_tau21, apply_rsds, **kw
                )

    picky = Picky(lc_distances=rectlcn.lc_distances, cosmo=rectlcn.cosmo)
    kw = {"inputs": default_input_struct_lc, "apply_rsds": False}
    with pytest.raises(ValueError, match="picky about velocities"):
        prepare_lightconer(picky, include_dvdr_in_tau21=True, **kw)
    assert prepare_lightconer(picky, include_dvdr_in_tau21=False, **kw) is picky


def test_isinstance_of_old_base_class(rectlcn):
    assert isinstance(rectlcn, lcn.Lightconer)
    assert lcn._LIGHTCONERS["RectilinearLightconer"] is p21c.RectilinearLightconer


def test_deprecated_methods(rectlcn, default_input_struct_lc, ic):
    so = default_input_struct_lc.simulation_options
    with pytest.warns(deprecation.DeprecatedWarning, match="get_shape is deprecated"):
        shape = rectlcn.get_shape(so)
    assert shape == (so.HII_DIM, so.HII_DIM, len(rectlcn.lc_distances))

    with pytest.warns(
        deprecation.DeprecatedWarning, match="validate_options is deprecated"
    ):
        new = rectlcn.validate_options(
            default_input_struct_lc, include_dvdr_in_tau21=True, apply_rsds=False
        )
    assert new.quantities == ("brightness_temp", "los_velocity")

    box = np.arange(so.HII_DIM**3, dtype=float).reshape((so.HII_DIM,) * 3)
    lcd = rectlcn.get_lc_distances_in_pixels(so.cell_size)
    with pytest.warns(
        deprecation.DeprecatedWarning, match="coeval_subselect is deprecated"
    ):
        sub = rectlcn.coeval_subselect(lcd[3], box, so.cell_size)
    index = rectlcn.box_indices(lcd.value)[3]
    np.testing.assert_array_equal(sub, box.take(index, axis=2, mode="wrap"))

    with pytest.warns(
        deprecation.DeprecatedWarning, match="redshift_interpolation is deprecated"
    ):
        out = rectlcn.redshift_interpolation(1.5, box, 3 * box, 1.0, 2.0)
    np.testing.assert_allclose(out, 2 * box)


def test_deprecated_make_lightcone_slices(rectlcn, default_input_struct_lc):
    """The method of v4.3 gives what the session gives."""
    inputs = default_input_struct_lc
    rng = np.random.default_rng(3)
    zs = inputs.node_redshifts[2:4]
    coevals = [
        SimpleNamespace(
            redshift=z,
            simulation_options=inputs.simulation_options,
            cosmo_params=inputs.cosmo_params,
            brightness_temp=rng.normal(size=(35, 35, 35)),
        )
        for z in zs
    ]
    session = rectlcn.start(lcn.context_from_inputs(inputs, node_redshifts=zs))
    expected = [chunk for c in coevals for chunk in session.ingest(CoevalSnapshot(c))]
    assert len(expected) > 0

    with pytest.warns(
        deprecation.DeprecatedWarning, match="make_lightcone_slices is deprecated"
    ):
        # In either order of the coevals, as in v4.3.
        got = list(rectlcn.make_lightcone_slices(coevals[1], coevals[0]))
    assert len(got) == len(expected)
    for (q, idx, values), chunk in zip(got, expected, strict=True):
        assert q == chunk.quantity
        assert idx == chunk.index[-1]
        np.testing.assert_array_equal(values, chunk.values)


# ======================================================================================
# The angular lightconer, which is part of tuesday
# ======================================================================================
def test_angular_lightconer_needs_tuesday(monkeypatch):
    """Without tuesday, asking for the AngularLightconer says what to install."""
    monkeypatch.delitem(sys.modules, "py21cmfast.lightconers.angular", raising=False)
    for name in list(sys.modules):
        if name == "tuesday" or name.startswith("tuesday."):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "tuesday", None)

    with pytest.raises(ImportError, match="pip install tuesday-eor"):
        _ = p21c.AngularLightconer
    with pytest.raises(ImportError, match="pip install tuesday-eor"):
        _ = lcn.AngularLightconer

    # Everything else works without it.
    assert p21c.RectilinearLightconer is lcn.RectilinearLightconer


def test_angular_lightconer_from_tuesday():
    core = pytest.importorskip("tuesday.core.lightcones")
    assert issubclass(p21c.AngularLightconer, core.AngularLightconer)
    assert p21c.AngularLightconer is lcn.AngularLightconer
    assert p21c.AngularLightconer.default_cosmo is p21c.wrapper.inputs.Planck18

    lightconer = p21c.AngularLightconer.between_redshifts(
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2 * un.Mpc,
        latitude=np.zeros(8),
        longitude=np.linspace(0, 1, 8),
        quantities=("density", "los_velocity"),
    )
    assert isinstance(lightconer, LightconerProtocol)
    assert isinstance(lightconer, lcn.Lightconer)
    assert lightconer.required_fields() == (
        "density",
        "velocity_x",
        "velocity_y",
        "velocity_z",
    )


# ======================================================================================
# Smaller paths: errors, deprecated entry points and helpers
# ======================================================================================
def test_redshift_interpolation_mean_max():
    a, b = np.array([1.0, -1.0, 2.0]), np.array([3.0, 5.0, -2.0])
    np.testing.assert_allclose(lcn.redshift_interpolation(2, a, b, 1, 3), [2, 2, 0])
    # Where the two values have opposite signs, the larger is taken.
    np.testing.assert_allclose(
        lcn.redshift_interpolation(2, a, b, 1, 3, kind="mean_max"), [2, 5, 2]
    )


def test_first_los_index():
    from py21cmfast.drivers.lightcone import _first_los_index

    assert _first_los_index((..., 4), 10) == 4
    assert _first_los_index((..., slice(3, 7)), 10) == 3
    assert _first_los_index((..., slice(None, None, 2)), 10) == 0
    assert _first_los_index((..., np.array([5, 2, 8])), 10) == 2
    # Where the chunk is along the line of sight is not always known.
    assert _first_los_index(4, 10) is None
    assert _first_los_index((), 10) is None
    assert _first_los_index((..., None), 10) is None


def test_make_checkpoint_still_works(lc, tmp_path):
    """The method that the driver used in v4.3 writes slices from the far end."""
    fname = tmp_path / "checkpoint.h5"
    full = lc.lightcones["brightness_temp"]
    n = full.shape[-1]

    partial = attrs.evolve(
        lc,
        lightcones={"brightness_temp": np.zeros_like(full)},
        global_quantities={
            k: np.zeros_like(v) for k, v in lc.global_quantities.items()
        },
    )
    partial.save(fname)

    for lcidx, node in ((n - 5, 1), (n - 12, 2)):
        partial.lightcones["brightness_temp"][..., lcidx:] = full[..., lcidx:]
        for k, v in lc.global_quantities.items():
            partial.global_quantities[k][: node + 1] = v[: node + 1]
        partial.make_checkpoint(fname, lcidx=lcidx, node_index=node)
        assert partial._last_completed_lcidx == lcidx
        assert partial._last_completed_node == node

    on_disk = LightCone.from_file(fname)
    np.testing.assert_array_equal(
        on_disk.lightcones["brightness_temp"][..., n - 12 :], full[..., n - 12 :]
    )
    assert np.all(on_disk.lightcones["brightness_temp"][..., : n - 12] == 0)
    np.testing.assert_array_equal(
        on_disk.global_quantities["brightness_temp"][:3],
        lc.global_quantities["brightness_temp"][:3],
    )


def test_deprecated_make_lightcone_slices_without_slices(
    rectlcn, default_input_struct_lc
):
    """As in v4.3, two coevals with no slice between them give one empty item."""
    inputs = default_input_struct_lc
    coevals = [
        SimpleNamespace(
            redshift=z,
            simulation_options=inputs.simulation_options,
            cosmo_params=inputs.cosmo_params,
            brightness_temp=np.zeros((35, 35, 35)),
        )
        for z in (5.0, 5.1)
    ]
    with pytest.warns(deprecation.DeprecatedWarning, match="make_lightcone_slices"):
        (item,) = rectlcn.make_lightcone_slices(*coevals)
    quantity, index, values = item
    assert quantity is None
    assert values is None
    assert len(index) == 0


def test_deprecated_find_required_lightcone_limits(
    default_input_struct_lc, old_style_class
):
    from py21cmfast.rsds import find_required_lightcone_limits
    from py21cmfast.wrapper.classy_interface import run_classy

    inputs = default_input_struct_lc
    zs = sorted(inputs.node_redshifts)
    rect = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=zs[2],
        max_redshift=zs[-3],
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )
    classy = run_classy(output="vTk")
    expected = find_required_lightcone_limits(rect, classy_output=classy, inputs=inputs)

    with pytest.warns(
        deprecation.DeprecatedWarning, match="find_required_lightcone_limits"
    ):
        limits = rect.find_required_lightcone_limits(classy, inputs=inputs)
    assert limits == expected

    # The base class of v4.3 has the method too (and the same line of sight).
    old = old_style_class(lc_distances=rect.lc_distances, cosmo=rect.cosmo)
    assert old.find_required_lightcone_limits(classy, inputs=inputs) == expected


def test_old_style_lightconer_errors(old_style_class, rectlcn, default_input_struct_lc):
    """The base class of v4.3 checks the coevals, and can leave a field missing."""
    inputs = default_input_struct_lc
    old = old_style_class(
        lc_distances=rectlcn.lc_distances,
        cosmo=rectlcn.cosmo,
        quantities=("brightness_temp", "density"),
    )

    def coeval(z, **kw):
        fields = {
            "simulation_options": inputs.simulation_options,
            "cosmo_params": inputs.cosmo_params,
            "brightness_temp": np.zeros((35, 35, 35)),
            **kw,
        }
        return SimpleNamespace(redshift=z, **fields)

    other_options = inputs.simulation_options.clone(BOX_LEN=51)
    with pytest.raises(ValueError, match="same user parameters"):
        next(
            old.make_lightcone_slices(
                coeval(20), coeval(21, simulation_options=other_options)
            )
        )
    other_cosmo = inputs.cosmo_params.clone(SIGMA_8=0.9)
    with pytest.raises(ValueError, match="same cosmological parameters"):
        next(
            old.make_lightcone_slices(coeval(20), coeval(21, cosmo_params=other_cosmo))
        )

    # Through the protocol, a field that the snapshots do not have is an
    # AttributeError for the lightconer (as with a Coeval), whatever its name.
    zs = inputs.node_redshifts[2:4]
    session = old.start(lcn.context_from_inputs(inputs, node_redshifts=zs))
    assert list(session.ingest(CoevalSnapshot(coeval(zs[0])))) == []
    with pytest.raises(AttributeError, match="density"):
        list(session.ingest(CoevalSnapshot(coeval(zs[1]))))
    assert session.finish() == ()

    # The default sub-selection of the base class is the whole box.
    box = np.ones((2, 2, 2))
    assert lcn.Lightconer.coeval_subselect(old, 1.0, box, 1.0) is box


def test_angular_lightconer_needs_a_recent_tuesday(monkeypatch):
    """A tuesday without the lightconers is said to be too old."""
    core = pytest.importorskip("tuesday.core.lightcones")
    monkeypatch.delitem(sys.modules, "py21cmfast.lightconers.angular", raising=False)
    monkeypatch.delattr(core, "AngularLightconer")

    with pytest.raises(ImportError, match="too old to have it"):
        _ = lcn.AngularLightconer
