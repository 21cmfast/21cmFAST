import pytest

from py21cmfast import global_params


@pytest.fixture(scope="session")
def tmpdirec(tmpdir_factory):
    """Pytest fixture instantiating a new session-scope "data" folder.

    Parameters
    ----------
    tmpdir_factory :
        Pytest fixture for creating temporary directories.
    """
    return tmpdir_factory.mktemp("data")


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package():
    # Set nice global defaults for testing purposes, to make runs faster
    # (can always be over-ridden per-test).
    original_zprime = global_params.ZPRIME_STEP_FACTOR
    global_params.ZPRIME_STEP_FACTOR = 1.2

    yield

    global_params.ZPRIME_STEP_FACTOR = original_zprime
