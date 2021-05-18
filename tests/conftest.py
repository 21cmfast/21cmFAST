import pytest

import logging
import os

from py21cmfast import UserParams, config, global_params, run_lightcone, wrapper
from py21cmfast.cache_tools import clear_cache


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
    original_zprime = global_params.ZPRIME_STEP_FACTOR

    # Set default global parameters for all tests
    global_params.ZPRIME_STEP_FACTOR = 1.2

    # Set default config parameters for all tests.
    config["direc"] = str(tmpdirec)
    config["regenerate"] = True
    config["write"] = False

    log_level = request.config.getoption("--log-level-21") or logging.INFO
    logging.getLogger("py21cmfast").setLevel(log_level)

    yield

    printdir(tmpdirec)

    clear_cache(direc=str(tmpdirec))
    global_params.ZPRIME_STEP_FACTOR = original_zprime


# ======================================================================================
# Create a default set of boxes that can be used throughout.
# ======================================================================================


@pytest.fixture(scope="session")
def default_user_params():
    return UserParams(HII_DIM=35, DIM=70, BOX_LEN=50)


@pytest.fixture(scope="session")
def ic(default_user_params, tmpdirec):
    return wrapper.initial_conditions(
        user_params=default_user_params, write=True, direc=tmpdirec, random_seed=12
    )


@pytest.fixture(scope="session")
def redshift():
    """A default redshift to evaluate at. Not too high, not too low."""
    return 15


@pytest.fixture(scope="session")
def max_redshift():
    """A default redshift to evaluate at. Not too high, not too low."""
    return 25


@pytest.fixture(scope="session")
def low_redshift():
    """A default redshift to evaluate at. Not too high, not too low."""
    return 8


@pytest.fixture(scope="session")
def perturb_field(ic, redshift):
    """A default perturb_field"""
    return wrapper.perturb_field(redshift=redshift, init_boxes=ic, write=True)


@pytest.fixture(scope="session")
def lc(perturb_field, max_redshift):
    return run_lightcone(perturb=perturb_field, max_redshift=max_redshift)
