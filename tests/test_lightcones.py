"""Test the Lightcone class."""

import numpy as np
from astropy import units

from py21cmfast import InputParameters, LightCone


def mock_lightcone():
    inputs = InputParameters.from_template(
        "simple", node_redshifts=list(range(6, 40, 2)), random_seed=1
    ).evolve_input_structs(HII_DIM=300, BOX_LEN=300)
    rng = np.random.default_rng(123)

    dist = np.arange(6000, 10000, inputs.simulation_options.cell_size) * units.Mpc

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

    def test_trim(self):
        """Test that trimming actually removes slices."""
        lc = mock_lightcone()

        newlc = lc.trim(7000 * units.Mpc, 9000 * units.Mpc)
        assert newlc.lightcone_distances.min() >= 7000 * units.Mpc
        assert newlc.lightcone_distances.max() <= 9000 * units.Mpc
