"""Test the Lightcone class."""

import numpy as np
import pytest
from astropy import units

from py21cmfast import InputParameters, LightCone
from py21cmfast.drivers.lightcone import _check_desired_arrays_exist


def mock_lightcone():
    inputs = InputParameters.from_template(
        "simple", node_redshifts=list(range(6, 40, 2)), random_seed=1
    ).evolve_input_structs(HII_DIM=200, BOX_LEN=200)
    rng = np.random.default_rng(123)

    dist = (
        np.arange(6000, 10000, inputs.simulation_options.cell_size.to_value("Mpc"))
        * units.Mpc
    )

    return LightCone(
        lightcone_distances=dist,
        inputs=inputs,
        lightcones={
            "brightness_temp": rng.uniform(
                -100,
                30,
                size=(
                    inputs.simulation_options.HII_DIM,
                    inputs.simulation_options.HII_DIM,
                    len(dist),
                ),
            )
        },
    )


class TestLightcone:
    """Test the LightCone object."""

    def test_invalid_quantity_raises(self):
        """Test that requesting an unknown lightcone quantity raises ValueError."""
        lc = mock_lightcone()
        with pytest.raises(ValueError, match="not computed"):
            _check_desired_arrays_exist(["not_a_real_quantity"], lc.inputs)

    def test_trim(self):
        """Test that trimming actually removes slices."""
        lc = mock_lightcone()

        newlc = lc.trim(7000 * units.Mpc, 9000 * units.Mpc)
        assert newlc.lightcone_distances.min() >= 7000 * units.Mpc
        assert newlc.lightcone_distances.max() <= 9000 * units.Mpc
