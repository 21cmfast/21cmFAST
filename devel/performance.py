"""Simple script to run a lightcone for performance testing."""

import sys
import time
from pathlib import Path

import tyro

from py21cmfast import OutputCache, RectilinearLightconer, run_lightcone

here = Path(__file__).parent.resolve()

sys.path.append(str(here / "tests"))

from produce_integration_test_data import (  # noqa: E402
    OPTIONS_TESTRUNS,
    get_all_options_struct,
    get_lc_fields,
)


def get_at_it(
    case: str = "fixed_halogrids",
    sizefac: int = 1,
):
    """Run the lightcone."""
    t = time.time()

    z, kwargs = OPTIONS_TESTRUNS[case]
    inputs = get_all_options_struct(z, lc=True, **kwargs)["inputs"]

    inputs = inputs.evolve_input_structs(
        HII_DIM=inputs.simulation_options.HII_DIM * sizefac,
        BOX_LEN=inputs.simulation_options.BOX_LEN * sizefac,
        DIM=inputs.simulation_options.DIM * sizefac,
    )

    node_z = inputs.node_redshifts

    quantities = get_lc_fields(inputs)
    try:
        lcn = RectilinearLightconer.between_redshifts(
            min_redshift=node_z[-1] + 0.2,
            max_redshift=node_z[0] - 0.2,
            quantities=quantities,
            resolution=inputs.simulation_options.cell_size,
        )
    except AttributeError:
        lcn = RectilinearLightconer.with_equal_cdist_slices(
            min_redshift=node_z[-1] + 0.2,
            max_redshift=node_z[0] - 0.2,
            quantities=quantities,
            resolution=inputs.simulation_options.cell_size,
        )
    run_lightcone(
        lightconer=lcn,
        write=True,  # write so that perturbed fields and halos can be cached.
        regenerate=True,
        cache=OutputCache("."),
        inputs=inputs,
        progressbar=True,
    )
    print(f"Lightcone run took {time.time() - t:.2f} seconds.")


if __name__ == "__main__":
    tyro.cli(get_at_it, description="Run a lightcone for performance testing.")
    # Example usage: python -m tests.performance --case fixed_halogrids --sizefac 2
    # This will run the lightcone with double the resolution and box size of the fixed_halogrids case.
