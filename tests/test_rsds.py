"""Tests of the RSDS module."""

import numpy as np
import pytest
from astropy import units

import py21cmfast as p21c
from py21cmfast import InputParameters, run_coeval
from py21cmfast.lightconers import RectilinearLightconer
from py21cmfast.rsds import apply_rsds, compute_rsds
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
        inputs=default_input_struct_ts.evolve_input_structs(APPLY_RSDS=True),
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
        ValueError, match="You have set APPLY_RSDS to True with node redshifts between"
    ):
        p21c.run_lightcone(
            lightconer=lcner,
            inputs=default_input_struct_ts.evolve_input_structs(APPLY_RSDS=True),
        )


class TestApplyRSDs:
    """Tests of the apply_rsds function."""

    @pytest.mark.parametrize("n_subcells", [1, 2, 4, 5])
    def test_mass_conservation(self, n_subcells):
        """Test that sum along LOS is perserved in cloud in cell for a periodic box."""
        nslices = 10
        nangles = 5
        rng = np.random.default_rng(12345)
        shape = (nslices, nangles)
        box_in = rng.random(shape)
        los_displacement = rng.random(shape) * units.pixel
        box_out = apply_rsds(
            field=box_in,
            los_displacement=los_displacement,
            n_subcells=n_subcells,
            periodic=True,
        )

        sum_in = np.sum(box_in, axis=0)
        sum1 = np.sum(box_out, axis=0)
        np.testing.assert_allclose(sum_in, sum1)

    @pytest.mark.parametrize("n_subcells", [1, 2])
    @pytest.mark.parametrize("velocity", [-10, -1, 0, 1, 10])
    def test_integer_shift(self, n_subcells: int, velocity: int):
        """Test that cloud in cell results in a shifted box, for an integer velocity and a periodic box."""
        nslices = 10
        nangles = 5
        rng = np.random.default_rng(12345)
        box_in = rng.random((nslices, nangles))
        los_displacement = velocity * np.ones_like(box_in) * units.pixel
        box_out = apply_rsds(
            field=box_in,
            los_displacement=los_displacement,
            n_subcells=n_subcells,
            periodic=True,
        )

        box_in_shifted = np.roll(box_in, velocity, axis=0)
        np.testing.assert_allclose(box_out, box_in_shifted)

    def test_bad_inputs(self):
        """Test that bad inputs raise good errors."""
        nslices = 1
        nangles = 5
        box_in = np.zeros((nslices, nangles))
        los_displacement = np.ones_like(box_in) * units.pixel

        with pytest.raises(ValueError, match="field must have at least 2 slices"):
            apply_rsds(
                field=box_in,
                los_displacement=los_displacement,
            )

    @pytest.mark.parametrize("n_subcells", [1, 2, 5])
    def test_non_periodic_large_displacement(self, n_subcells: int):
        """Test that a very large displacement results in all mass leaving the box."""
        nslices = 10
        nangles = 5
        box_in = np.ones((nslices, nangles))
        los_displacement = nslices * 2 * np.ones_like(box_in) * units.pixel

        box_out = apply_rsds(
            field=box_in,
            los_displacement=los_displacement,
            periodic=False,
            n_subcells=n_subcells,
        )
        np.testing.assert_allclose(box_out, 0)


class TestComputeRSDs:
    """Tests of the compute_rsds function.

    Since this function is pretty much just a wrapper around apply_rsds,
    we mostly test for case-behaviour (e.g. testing for errors raised),
    rather than for exactness of the output, which is done in TestApplyRSDs.
    """

    def setup_class(self):
        """Set up the arrays re-used throughout the class."""
        self.nslc = 10
        self.nang = 5

        self.bt3d = np.ones((self.nang, self.nang, self.nslc))
        self.bt2d = np.ones((self.nang, self.nslc))
        self.vel3d = np.ones_like(self.bt3d)
        self.vel2d = np.ones_like(self.bt2d)
        self.inputs = InputParameters.from_template(
            "simple", node_redshifts=[6, 7, 8, 40], random_seed=1
        )

    def test_bad_inputs(self):
        """Test that bad inputs raise good errors."""
        with pytest.raises(ValueError, match="tau_21 is not provided"):
            compute_rsds(
                brightness_temp=self.bt3d,
                los_velocity=self.vel3d,
                redshifts=6.0,
                inputs=self.inputs.evolve_input_structs(USE_TS_FLUCT=True),
                tau_21=None,
            )

    @pytest.mark.parametrize("periodic", [True, False, None])
    def test_that_coeval_is_treated_as_periodic(self, periodic: bool | None):
        """Test that cubic boxes are treated as periodic by default."""
        n = 10
        inputs = self.inputs.evolve_input_structs(
            HII_DIM=n, BOX_LEN=n, R_BUBBLE_MAX=n / 4
        )
        bt = np.ones((n, n, n))  # like a coeval

        hubble = inputs.cosmo_params.cosmo.H(8.0)
        vel = (
            np.ones_like(bt) * n * 2 * hubble.to_value(1 / units.s)
        )  # Move stuff twice the box length!

        # periodic is 'None' and gets interpreted as True since
        # bt is cubic.
        bt_out = compute_rsds(
            brightness_temp=bt,
            los_velocity=vel,
            redshifts=8.0,
            inputs=inputs,
            periodic=periodic,
            n_subcells=2,
        )

        # Since its periodic, we have mass conservation
        if periodic in [True, None]:
            # We don't really have mass conservation at the moment
            # because of the "linear" RSDs which change the amount of mass
            # TODO: this seems wrong...
            assert np.isclose(np.sum(bt_out), np.sum(bt), rtol=1e-1)
        else:
            assert np.allclose(bt_out, 0)

    def test_2d_ok(self):
        """Test that 2D brightness temp arrays are OK."""
        rng = np.random.default_rng(1019)
        nslc = 12
        bt3d = rng.uniform(-100, 30, size=(3, 3, nslc))
        vel = rng.uniform(-1.5, 1.5, size=bt3d.shape)

        box_out_3d = compute_rsds(
            brightness_temp=bt3d,
            los_velocity=vel,
            redshifts=8.0,
            inputs=self.inputs,
        )

        box_out_2d = compute_rsds(
            brightness_temp=bt3d.reshape((-1, nslc)),
            los_velocity=vel.reshape((-1, nslc)),
            redshifts=8.0,
            inputs=self.inputs,
        )

        np.testing.assert_allclose(box_out_3d.flatten(), box_out_2d.flatten())
