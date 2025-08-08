"""Tests of the RSDS module."""

import numpy as np
import pytest
from astropy import units

import py21cmfast as p21c
from py21cmfast import InputParameters, run_coeval
from py21cmfast.lightconers import AngularLightconer, RectilinearLightconer
from py21cmfast.rsds import apply_rsds, include_dvdr_in_tau21, rsds_shift
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
    box_rsd = coeval[0].apply_velocity_corrections()
    assert box_rsd.shape == coeval[0].brightness_temperature.brightness_temp.shape

    box_rsd = coeval[0].include_dvdr_in_tau21()
    assert box_rsd.shape == coeval[0].brightness_temperature.brightness_temp.shape

    box_rsd = coeval[0].apply_rsds()
    assert box_rsd.shape == coeval[0].brightness_temperature.brightness_temp.shape


def test_bad_coeval_inputs(default_input_struct, cache):
    coeval = run_coeval(
        inputs=default_input_struct.evolve_input_structs(KEEP_3D_VELOCITIES=False),
        out_redshifts=8.0,
        cache=cache,
    )
    with pytest.raises(ValueError, match="You asked for axis ="):
        coeval[0].apply_velocity_corrections(
            axis="x"
        )  # fails because KEEP_3D_VELOCITIES = False
    with pytest.raises(ValueError, match="You asked for axis ="):
        coeval[0].include_dvdr_in_tau21(
            axis="x"
        )  # fails because KEEP_3D_VELOCITIES = False
    with pytest.raises(ValueError, match="You asked for axis ="):
        coeval[0].apply_rsds(axis="x")  # fails because KEEP_3D_VELOCITIES = False
    with pytest.raises(ValueError, match="`axis` can only be `x`, `y` or `z`."):
        coeval[0].apply_velocity_corrections(axis="t")
    with pytest.raises(ValueError, match="`axis` can only be `x`, `y` or `z`."):
        coeval[0].include_dvdr_in_tau21(axis="t")
    with pytest.raises(ValueError, match="`axis` can only be `x`, `y` or `z`."):
        coeval[0].apply_rsds(axis="t")


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
        ValueError, match="You have set apply_rsds to True with node redshifts between"
    ):
        p21c.run_lightcone(
            lightconer=lcner, inputs=default_input_struct_ts, apply_rsds=True
        )


class TestRSDsShift:
    """Tests of the rsds_shift function."""

    @pytest.mark.parametrize("n_rsd_subcells", [1, 2, 4, 5])
    def test_mass_conservation(self, n_rsd_subcells):
        """Test that sum along LOS is perserved in cloud in cell for a periodic box."""
        nslices = 10
        nangles = 5
        rng = np.random.default_rng(12345)
        shape = (nslices, nangles)
        box_in = rng.random(shape)
        los_displacement = rng.random(shape) * units.pixel
        box_out = rsds_shift(
            field=box_in,
            los_displacement=los_displacement,
            n_rsd_subcells=n_rsd_subcells,
            periodic=True,
        )

        sum_in = np.sum(box_in, axis=0)
        sum1 = np.sum(box_out, axis=0)
        np.testing.assert_allclose(sum_in, sum1)

    @pytest.mark.parametrize("n_rsd_subcells", [1, 2])
    @pytest.mark.parametrize("velocity", [-10, -1, 0, 1, 10])
    def test_integer_shift(self, n_rsd_subcells: int, velocity: int):
        """Test that cloud in cell results in a shifted box, for an integer velocity and a periodic box."""
        nslices = 10
        nangles = 5
        rng = np.random.default_rng(12345)
        box_in = rng.random((nslices, nangles))
        los_displacement = velocity * np.ones_like(box_in) * units.pixel
        box_out = rsds_shift(
            field=box_in,
            los_displacement=los_displacement,
            n_rsd_subcells=n_rsd_subcells,
            periodic=True,
        )

        box_in_shifted = np.roll(box_in, velocity, axis=0)
        np.testing.assert_allclose(box_out, box_in_shifted)

    @pytest.mark.parametrize("n_rsd_subcells", [1, 2, 5])
    def test_non_periodic_large_displacement(self, n_rsd_subcells: int):
        """Test that a very large displacement results in all mass leaving the box."""
        nslices = 10
        nangles = 5
        box_in = np.ones((nslices, nangles))
        los_displacement = nslices * 2 * np.ones_like(box_in) * units.pixel

        box_out = rsds_shift(
            field=box_in,
            los_displacement=los_displacement,
            periodic=False,
            n_rsd_subcells=n_rsd_subcells,
        )
        np.testing.assert_allclose(box_out, 0)


class TestComputeRSDs:
    """Tests of include_dvdr_in_tau21 and apply_rsds functions.

    Since these functions are pretty much just a wrapper around rsds_shift,
    we mostly test for case-behaviour (e.g. testing for errors raised),
    rather than for exactness of the output, which is done in TestRSDsShift.
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

    @pytest.mark.parametrize("periodic", [True, False])
    def test_bad_inputs(self, periodic: bool | None):
        """Test that bad inputs raise good errors."""
        with pytest.raises(ValueError, match="tau_21 is not provided"):
            include_dvdr_in_tau21(
                brightness_temp=self.bt3d,
                los_velocity=self.vel3d,
                redshifts=6.0,
                inputs=self.inputs.evolve_input_structs(USE_TS_FLUCT=True),
                tau_21=None,
                periodic=periodic,
            )
        with pytest.raises(
            ValueError,
            match="Redshifts must be a float or array with the same size as number of LoS slices",
        ):
            include_dvdr_in_tau21(
                brightness_temp=self.bt3d,
                los_velocity=self.vel3d,
                redshifts=[6.0, 8.0],
                inputs=self.inputs,
                periodic=periodic,
            )
        with pytest.raises(
            ValueError,
            match="brightness_temp must be an array with the same shape as los_velocity",
        ):
            include_dvdr_in_tau21(
                brightness_temp=self.bt3d[:, :, :-1],
                los_velocity=self.vel3d,
                redshifts=6.0,
                inputs=self.inputs,
                periodic=periodic,
            )
        with pytest.raises(ValueError, match="field must have at least 2 slices"):
            apply_rsds(
                field=self.bt3d[:, :, 0].reshape(self.nang, self.nang, 1),
                los_velocity=self.vel3d[:, :, 0].reshape(self.nang, self.nang, 1),
                redshifts=6.0,
                inputs=self.inputs,
                periodic=periodic,
            )
        with pytest.raises(
            ValueError,
            match="Redshifts must be a float or array with the same size as number of LoS slices",
        ):
            apply_rsds(
                field=self.bt3d,
                los_velocity=self.vel3d,
                redshifts=[6.0, 8.0],
                inputs=self.inputs,
                periodic=periodic,
            )
        with pytest.raises(
            ValueError,
            match="field must be an array with the same shape as los_displacement",
        ):
            apply_rsds(
                field=self.bt3d[:, :, :-1],
                los_velocity=self.vel3d,
                redshifts=6.0,
                inputs=self.inputs,
                periodic=periodic,
            )
        with pytest.raises(ValueError, match="n_rsd_subcells must be an integer"):
            apply_rsds(
                field=self.bt3d,
                los_velocity=self.vel3d,
                redshifts=6.0,
                inputs=self.inputs,
                periodic=periodic,
                n_rsd_subcells=2.5,
            )

    @pytest.mark.parametrize("periodic", [True, False])
    def test_2d_ok(self, periodic: bool | None):
        """Test that 2D brightness temp arrays are OK."""
        rng = np.random.default_rng(1019)
        nslc = 12
        bt3d = rng.uniform(-100, 30, size=(3, 3, nslc))
        vel = rng.uniform(-1.5, 1.5, size=bt3d.shape)

        box_out_3d = apply_rsds(
            field=bt3d,
            los_velocity=vel,
            redshifts=8.0,
            inputs=self.inputs,
            periodic=periodic,
        )

        box_out_2d = apply_rsds(
            field=bt3d.reshape((-1, nslc)),
            los_velocity=vel.reshape((-1, nslc)),
            redshifts=8.0,
            inputs=self.inputs,
            periodic=periodic,
        )

        np.testing.assert_allclose(box_out_3d.flatten(), box_out_2d.flatten())


@pytest.mark.parametrize("lcner", ["rect", "ang"])
def test_new_rsd_lightcones(cache, lcner):
    """Test that new lightcones are added to output."""
    inputs = p21c.InputParameters(
        random_seed=12345,
        node_redshifts=p21c.get_logspaced_redshifts(
            min_redshift=34.0, max_redshift=35.0, z_step_factor=1.02
        ),
    ).evolve_input_structs(
        BOX_LEN=15,
        DIM=45,
        HII_DIM=10,
        N_THREADS=1,
        USE_TS_FLUCT=True,
        KEEP_3D_VELOCITIES=True,
    )
    if lcner == "rect":
        lightconer = RectilinearLightconer.between_redshifts(
            min_redshift=inputs.node_redshifts[-1] + 0.5,
            max_redshift=inputs.node_redshifts[0] - 0.5,
            resolution=inputs.simulation_options.cell_size,
            cosmo=inputs.cosmo_params.cosmo,
            quantities=("brightness_temp", "density"),
        )
    else:
        lightconer = AngularLightconer.like_rectilinear(
            max_redshift=inputs.node_redshifts[0] - 0.5,
            simulation_options=inputs.simulation_options,
            match_at_z=inputs.node_redshifts[-1] + 0.5,
            cosmo=inputs.cosmo_params.cosmo,
            quantities=("brightness_temp", "density"),
        )

    lightcone = p21c.run_lightcone(
        lightconer=lightconer,
        inputs=inputs,
        cache=cache,
        include_dvdr_in_tau21=True,
        apply_rsds=True,
    )
    assert "tau_21" in lightcone.lightcones
    assert "density_with_rsds" in lightcone.lightcones

    assert not np.allclose(
        lightcone.lightcones["brightness_temp"],
        lightcone.lightcones["brightness_temp_with_rsds"],
    )
    assert not np.allclose(
        lightcone.lightcones["density"], lightcone.lightcones["density_with_rsds"]
    )
    assert not np.allclose(lightcone.lightcones["tau_21"], 0)
