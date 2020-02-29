import pytest

from py21cmfast import config
from py21cmfast import global_params


@pytest.fixture(scope="session")
def tmpdirec(tmp_path_factory):
    """Pytest fixture instantiating a new session-scope "data" folder.

    Parameters
    ----------
    tmpdir_factory :
        Pytest fixture for creating temporary directories.
    """
    return tmp_path_factory.mktemp("data")


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package(tmpdirec):
    # Set nice global defaults for testing purposes, to make runs faster
    # (can always be over-ridden per-test).
    original_zprime = global_params.ZPRIME_STEP_FACTOR

    # Set default global parameters for all tests
    global_params.ZPRIME_STEP_FACTOR = 1.2
    config["direc"] = tmpdirec.strpath

    # Set default config parameters for all tests.
    config["direc"] = str(tmpdirec)
    config["regenerate"] = True
    config["write"] = False

    yield

    global_params.ZPRIME_STEP_FACTOR = original_zprime
