"""Test configuration."""

import logging
import os
from pathlib import Path

import pytest

from py21cmfast import (
    AstroOptions,
    AstroParams,
    CosmoParams,
    InitialConditions,
    InputParameters,
    MatterOptions,
    OutputCache,
    PerturbedField,
    SimulationOptions,
    compute_initial_conditions,
    compute_ionization_field,
    config,
    get_logspaced_redshifts,
    perturb_field,
    run_lightcone,
)
from py21cmfast.io.caching import CacheConfig
from py21cmfast.lightconers import RectilinearLightconer


def pytest_addoption(parser):
    parser.addoption("--log-level-21", action="store", default="WARNING")


@pytest.fixture(scope="session")
def tmpdirec(tmp_path_factory):
    """Pytest fixture instantiating a new session-scope "data" folder.

    Parameters
    ----------
    tmpdir_factory :
        Pytest fixture for creating temporary directories.
    """
    return tmp_path_factory.mktemp("data")


def printdir(direc):
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 100

    print()
    print(f" Files In {direc} ".center(width, "="))
    for pth in direc.iterdir():
        print(f"\t {pth.name:<20}:\t\t {pth.stat().st_size / 1024**2:.3f} KB")
    print("=" * width)


@pytest.fixture(scope="module")
def module_direc(tmp_path_factory):
    original = config["direc"]
    direc = tmp_path_factory.mktemp("modtmp")

    config["direc"] = str(direc)

    yield direc

    printdir(direc)

    # Set direc back to original.
    config["direc"] = original


@pytest.fixture
def test_direc(tmp_path_factory):
    original = config["direc"]
    direc = tmp_path_factory.mktemp("testtmp")

    config["direc"] = str(direc)

    yield direc

    printdir(direc)

    # Set direc back to original.
    config["direc"] = original


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package(tmpdirec, request):
    # Set nice global defaults for testing purposes, to make runs faster
    # (can always be over-ridden per-test).

    # Set default global parameters for all tests
    # ------ #

    # we run small boxes often here, and R_max is often large, so we ignore this error
    config["ignore_R_BUBBLE_MAX_error"] = True

    log_level = request.config.getoption("--log-level-21") or logging.INFO
    logging.getLogger("py21cmfast").setLevel(log_level)

    yield

    printdir(tmpdirec)


# ======================================================================================
# Create a default set of boxes that can be used throughout.
# ======================================================================================


@pytest.fixture(scope="session")
def default_seed():
    return 12


@pytest.fixture(scope="session")
def default_simulation_options():
    return SimulationOptions(
        HII_DIM=35,
        DIM=70,
        BOX_LEN=50,
        ZPRIME_STEP_FACTOR=1.2,
    )


@pytest.fixture(scope="session")
def default_matter_options():
    return MatterOptions(
        KEEP_3D_VELOCITIES=True,
        USE_HALO_FIELD=False,
        HALO_STOCHASTICITY=False,
    )


@pytest.fixture(scope="session")
def default_cosmo_params():
    return CosmoParams()


@pytest.fixture(scope="session")
def default_astro_options():
    return AstroOptions(
        USE_EXP_FILTER=False,
        CELL_RECOMB=False,
        USE_UPPER_STELLAR_TURNOVER=False,
        INCLUDE_DVDR_IN_TAU21=False,
    )


@pytest.fixture(scope="session")
def default_astro_options_ts():
    return AstroOptions(
        USE_EXP_FILTER=False,
        CELL_RECOMB=False,
        USE_TS_FLUCT=True,
        USE_UPPER_STELLAR_TURNOVER=False,
    )


@pytest.fixture(scope="session")
def default_astro_params():
    return AstroParams.new()


@pytest.fixture(scope="session")
def default_input_struct(
    default_simulation_options,
    default_matter_options,
    default_cosmo_params,
    default_astro_params,
    default_astro_options,
    default_seed,
):
    return InputParameters(
        random_seed=default_seed,
        cosmo_params=default_cosmo_params,
        astro_params=default_astro_params,
        simulation_options=default_simulation_options,
        matter_options=default_matter_options,
        astro_options=default_astro_options,
        node_redshifts=(),
    )


@pytest.fixture(scope="session")
def default_input_struct_ts(redshift, default_input_struct, default_astro_options_ts):
    return default_input_struct.clone(
        astro_options=default_astro_options_ts,
        node_redshifts=get_logspaced_redshifts(
            min_redshift=redshift,
            max_redshift=default_input_struct.simulation_options.Z_HEAT_MAX,
            z_step_factor=default_input_struct.simulation_options.ZPRIME_STEP_FACTOR,
        ),
    )


@pytest.fixture(scope="session")
def default_input_struct_lc(lightcone_min_redshift, default_input_struct):
    return default_input_struct.clone(
        node_redshifts=get_logspaced_redshifts(
            min_redshift=lightcone_min_redshift,
            max_redshift=default_input_struct.simulation_options.Z_HEAT_MAX,
            z_step_factor=default_input_struct.simulation_options.ZPRIME_STEP_FACTOR,
        )
    )


@pytest.fixture(scope="session")
def cache(tmpdirec: Path):
    return OutputCache(tmpdirec)


@pytest.fixture(scope="session")
def ic(default_input_struct, cache) -> InitialConditions:
    return compute_initial_conditions(
        inputs=default_input_struct,
        write=True,
        cache=cache,
    )


@pytest.fixture(scope="session")
def redshift():
    """A default redshift to evaluate at. Not too high, not too low."""
    return 15


@pytest.fixture(scope="session")
def lightcone_min_redshift(redshift):
    return redshift + 0.1


@pytest.fixture(scope="session")
def max_redshift():
    """A default redshift to evaluate at. Not too high, not too low."""
    return 25


@pytest.fixture(scope="session")
def low_redshift():
    """A default redshift to evaluate at. Not too high, not too low."""
    return 8


@pytest.fixture(scope="session")
def perturbed_field(ic, redshift, cache):
    """A default PerturbedField."""
    return perturb_field(
        redshift=redshift,
        initial_conditions=ic,
        write=True,
        cache=cache,
    )


@pytest.fixture(scope="session")
def perturbed_field_lc(
    ic: InitialConditions,
    default_input_struct_lc: InputParameters,
    redshift: float,
    cache,
):
    """A default PerturbedField for a lightcone (which requires node_redshifts)."""
    return perturb_field(
        redshift=redshift,
        inputs=default_input_struct_lc,
        initial_conditions=ic,
        write=True,
        cache=cache,
    )


@pytest.fixture(scope="session")
def ionize_box(
    ic: InitialConditions,
    perturbed_field: PerturbedField,
    cache: OutputCache,
):
    """A default ionize_box."""
    return compute_ionization_field(
        initial_conditions=ic,
        perturbed_field=perturbed_field,
        write=True,
        cache=cache,
    )


@pytest.fixture(scope="session")
def rectlcn(
    lightcone_min_redshift,
    max_redshift,
    default_simulation_options,
    default_cosmo_params,
) -> RectilinearLightconer:
    return RectilinearLightconer.between_redshifts(
        min_redshift=lightcone_min_redshift,
        max_redshift=max_redshift,
        resolution=default_simulation_options.cell_size,
        cosmo=default_cosmo_params.cosmo,
    )


@pytest.fixture(scope="session")
def lc(rectlcn, ic, cache, default_input_struct_lc):
    return run_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        write=CacheConfig(),
        cache=cache,
    )
