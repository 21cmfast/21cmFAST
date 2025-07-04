"""Tests of the RSDS module."""

import numpy as np
import pytest
from astropy import units

import py21cmfast as p21c
from py21cmfast import run_coeval
from py21cmfast.lightconers import RectilinearLightconer
from py21cmfast.rsds import apply_rsds, cloud_in_cell_los
from py21cmfast.wrapper.classy_interface import run_classy


class TestFindRequiredLightconeLimits:
    """Tests of the find_required_lightcone_limits method."""

    def setup_class(self):
        """Set up the RectilinearLightconer for testing."""
        node_redshifts = p21c.get_logspaced_redshifts(
            min_redshift=6.0, max_redshift=35.0, z_step_factor=1.02
        )
        self.inputs = p21c.InputParameters(
            random_seed=12345, node_redshifts=node_redshifts
        )

        self.lcner = RectilinearLightconer.between_redshifts(
            min_redshift=self.inputs.node_redshifts[-1] + 0.5,
            max_redshift=self.inputs.node_redshifts[0] - 0.5,
            resolution=self.inputs.simulation_options.cell_size,
            quantities=("brightness_temp",),
        )

    def test_limits_are_reasonable(self):
        """Test that the limits returned by find_required_lightcone_limits are reasonable."""
        classy = run_classy(
            inputs=self.inputs,
            output="vTk",
        )
        limits = self.lcner.find_required_lightcone_limits(classy, inputs=self.inputs)
        assert len(limits) == 2
        assert limits[0] < limits[1]
        assert limits[0] <= self.lcner.lc_distances.min()
        assert limits[1] >= self.lcner.lc_distances.max()
        assert limits[0] > self.lcner.lc_distances.min() - 2 * units.Mpc
        assert limits[1] < self.lcner.lc_distances.max() + 2 * units.Mpc


def test_coeval_rsds(ic, default_input_struct_ts, cache):
    """Test rsds on coeval boxes."""
    coeval = run_coeval(
        initial_conditions=ic,
        inputs=default_input_struct_ts,
        cache=cache,
    )
    box_rsd = coeval[0].compute_rsds()
    assert box_rsd.shape == coeval[0].brightness_temperature.brightness_temp.shape

    coeval = run_coeval(
        initial_conditions=ic,
        inputs=default_input_struct_ts.evolve_input_structs(SUBCELL_RSD=True),
        cache=cache,
        regenerate=True,
    )
    box_rsd = coeval[0].compute_rsds()
    assert box_rsd.shape == coeval[0].brightness_temperature.brightness_temp.shape


def test_bad_lightconer_inputs(default_input_struct_ts):
    lcner = RectilinearLightconer.between_redshifts(
        min_redshift=default_input_struct_ts.node_redshifts[-1],
        max_redshift=default_input_struct_ts.node_redshifts[0],
        resolution=default_input_struct_ts.simulation_options.cell_size,
        quantities=("brightness_temp",),
    )
    with pytest.raises(
        ValueError,
        match="The lightcone redshifts are not compatible with the given redshift.",
    ):
        p21c.run_lightcone(lightconer=lcner, inputs=default_input_struct_ts)

    lcner = RectilinearLightconer.between_redshifts(
        min_redshift=default_input_struct_ts.node_redshifts[-1],
        max_redshift=default_input_struct_ts.simulation_options.Z_HEAT_MAX,
        resolution=default_input_struct_ts.simulation_options.cell_size,
        quantities=("brightness_temp",),
    )
    with pytest.raises(
        ValueError, match="You have set SUBCELL_RSD to True with node redshifts between"
    ):
        p21c.run_lightcone(
            lightconer=lcner,
            inputs=default_input_struct_ts.evolve_input_structs(SUBCELL_RSD=True),
        )


def test_apply_rsds_sum():
    """Test that sum along LOS is perserved in cloud in cell for a periodic box."""
    nslices = 10
    nangles = 5
    rng = np.random.default_rng(12345)
    box_in = rng.random((nangles, nslices))
    los_displacement = rng.random((nangles, nslices)) * units.pixel
    distance = np.arange(nslices) * units.pixel
    box_out1 = apply_rsds(
        field=box_in.T,
        los_displacement=los_displacement.T,
        distance=distance,
        n_subcells=1,
        periodic=True,
    ).T
    box_out2 = apply_rsds(
        field=box_in.T,
        los_displacement=los_displacement.T,
        distance=distance,
        n_subcells=2,
        periodic=True,
    ).T

    sum_in = np.sum(box_in, axis=-1)
    sum1 = np.sum(box_out1, axis=-1)
    sum2 = np.sum(box_out2, axis=-1)

    assert np.max(np.abs(1.0 - sum1 / sum_in)) < 1e-3
    assert np.max(np.abs(1.0 - sum2 / sum_in)) < 1e-3


def test_apply_rsds_shift():
    """Test that cloud in cell results in a shifted box, for an integer velocity and a periodic box."""
    nslices = 10
    nangles = 5
    rng = np.random.default_rng(12345)
    v = int(rng.random())
    box_in = rng.random((nangles, nslices))
    los_displacement = v * np.ones_like(box_in) * units.pixel
    distance = np.arange(nslices) * units.pixel
    box_out1 = apply_rsds(
        field=box_in.T,
        los_displacement=los_displacement.T,
        distance=distance,
        n_subcells=1,
        periodic=True,
    ).T
    box_out2 = apply_rsds(
        field=box_in.T,
        los_displacement=los_displacement.T,
        distance=distance,
        n_subcells=2,
        periodic=True,
    ).T

    box_in_shifted = np.roll(box_in, v, axis=-1)

    assert np.max(np.abs(1.0 - box_out1 / box_in_shifted)) < 1e-3
    assert np.max(np.abs(1.0 - box_out2 / box_in_shifted)) < 1e-3


def test_cloud_in_cell():
    """Similar to above, but directly on cloud_in_cell, without numba."""
    nslices = 10
    nangles = 5
    rng = np.random.default_rng(12345)
    v = int(rng.random())
    box_in = rng.random((nangles, nslices))
    delta_los = v * np.ones_like(box_in) * units.pixel
    box_out = cloud_in_cell_los(
        field=box_in,
        delta_los=delta_los,
        periodic=True,
    )

    box_in_shifted = np.roll(box_in, v, axis=-1)

    assert np.max(np.abs(1.0 - box_out / box_in_shifted)) < 1e-3
