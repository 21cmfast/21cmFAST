"""Tests of the RSDS module."""

from astropy import units

import py21cmfast as p21c
from py21cmfast import rsds
from py21cmfast.lightconers import RectilinearLightconer
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
