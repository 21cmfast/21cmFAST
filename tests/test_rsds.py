"""Tests of the RSDS module."""

from py21cmfast import rsds
from py21cmfast.lightconers import RectilinearLightconer
from py21cmfast.wrapper.class_interface import run_classy


class TestFindRequiredLightconeLimits:
    """Tests of the find_required_lightcone_limits method."""

    def setup_class(self):
        """Set up the RectilinearLightconer for testing."""
        self.lcner = RectilinearLightconer.between_redshifts(
            min_redshift=6.0,
            max_redshift=7.0,
            resolution=2.0,  # Mpc
            quantities=("brightness_temp",),
        )

    def test_limits_are_reasonable(self):
        """Test that the limits returned by find_required_lightcone_limits are reasonable."""
        classy = run_classy(
            inputs=self.lcner.inputs,
            output="vPk",
        )
        limits = self.lcner.find_required_lightcone_limits(
            classy, inputs=self.lcner.inputs
        )
        assert len(limits) == 2
        assert limits[0] < limits[1]
        assert limits[0] <= self.lcner.lc_distances.min()
        assert limits[1] >= self.lcner.lc_distances.max()
        assert limits[0] > self.lcner.lc_distances.min() - 10  # Mpc
        assert limits[1] < self.lcner.lc_distances.max() + 10  # Mpc
