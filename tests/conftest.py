import pytest

import logging
import os
from astropy import units as un

from py21cmfast import (
    AstroParams,
    CosmoParams,
    FlagOptions,
    InputParameters,
    UserParams,
    compute_initial_conditions,
    config,
    exhaust_lightcone,
    get_logspaced_redshifts,
    global_params,
    perturb_field,
    run_lightcone,
)
from py21cmfast.cache_tools import clear_cache
from py21cmfast.lightcones import RectilinearLightconer


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

    # Clear all cached items created.
    clear_cache(direc=str(direc))

    # Set direc back to original.
    config["direc"] = original


@pytest.fixture(scope="function")
def test_direc(tmp_path_factory):
    original = config["direc"]
    direc = tmp_path_factory.mktemp("testtmp")

    config["direc"] = str(direc)

    yield direc

    printdir(direc)
    # Clear all cached items created.
    clear_cache(direc=str(direc))

    # Set direc back to original.
    config["direc"] = original


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package(tmpdirec, request):
    # Set nice global defaults for testing purposes, to make runs faster
    # (can always be over-ridden per-test).

    # Set default global parameters for all tests
    # ------ #

    # Set default config parameters for all tests.
    config["direc"] = str(tmpdirec)
    config["regenerate"] = True
    config["write"] = False

    log_level = request.config.getoption("--log-level-21") or logging.INFO
    logging.getLogger("py21cmfast").setLevel(log_level)
    logging.getLogger("21cmFAST").setLevel(log_level)

    yield

    printdir(tmpdirec)

    clear_cache(direc=str(tmpdirec))


# ======================================================================================
# Create a default set of boxes that can be used throughout.
# ======================================================================================


@pytest.fixture(scope="session")
def default_seed():
    return 12


@pytest.fixture(scope="session")
def default_user_params():
    return UserParams(
        HII_DIM=35,
        DIM=70,
        BOX_LEN=50,
        KEEP_3D_VELOCITIES=True,
        ZPRIME_STEP_FACTOR=1.2,
    )


@pytest.fixture(scope="session")
def default_cosmo_params():
    return CosmoParams()


@pytest.fixture(scope="session")
def default_flag_options():
    return FlagOptions(
        USE_HALO_FIELD=False,
        USE_EXP_FILTER=False,
        CELL_RECOMB=False,
        HALO_STOCHASTICITY=False,
    )


@pytest.fixture(scope="session")
def default_flag_options_ts():
    return FlagOptions(
        USE_HALO_FIELD=False,
        USE_EXP_FILTER=False,
        CELL_RECOMB=False,
        HALO_STOCHASTICITY=False,
        USE_TS_FLUCT=True,
    )


@pytest.fixture(scope="session")
def default_astro_params():
    return AstroParams.new()


@pytest.fixture(scope="session")
def default_input_struct(
    default_user_params,
    default_cosmo_params,
    default_astro_params,
    default_flag_options,
    default_seed,
):
    return InputParameters(
        random_seed=default_seed,
        cosmo_params=default_cosmo_params,
        astro_params=default_astro_params,
        user_params=default_user_params,
        flag_options=default_flag_options,
        node_redshifts=[],
    )


@pytest.fixture(scope="session")
def default_input_struct_ts(
    redshift, max_redshift, default_input_struct, default_flag_options_ts
):
    return default_input_struct.clone(
        flag_options=default_flag_options_ts,
        node_redshifts=get_logspaced_redshifts(
            min_redshift=redshift,
            max_redshift=default_input_struct.user_params.Z_HEAT_MAX,
            z_step_factor=default_input_struct.user_params.ZPRIME_STEP_FACTOR,
        ),
    )


@pytest.fixture(scope="session")
def default_input_struct_lc(redshift, max_redshift, default_input_struct):
    return default_input_struct.clone(
        node_redshifts=get_logspaced_redshifts(
            min_redshift=redshift,
            max_redshift=max_redshift,
            z_step_factor=default_input_struct.user_params.ZPRIME_STEP_FACTOR,
        )
    )


@pytest.fixture(scope="session")
def ic(default_input_struct, default_seed, tmpdirec):
    return compute_initial_conditions(
        inputs=default_input_struct,
        write=True,
        direc=tmpdirec,
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
def perturbed_field(ic, default_input_struct, redshift):
    """A default perturb_field"""
    return perturb_field(
        redshift=redshift,
        inputs=default_input_struct,
        initial_conditions=ic,
        write=True,
    )


@pytest.fixture(scope="session")
def rectlcn(
    lightcone_min_redshift, max_redshift, default_user_params, default_cosmo_params
) -> RectilinearLightconer:
    return RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=lightcone_min_redshift,
        max_redshift=max_redshift,
        resolution=default_user_params.cell_size,
        cosmo=default_cosmo_params.cosmo,
    )


@pytest.fixture(scope="session")
def lc(rectlcn, ic, default_input_struct_lc):
    iz, z, coev, lc = exhaust_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        write=True,
    )
    return lc
