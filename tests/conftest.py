import pytest


@pytest.fixture(scope="session")
def tmpdirec(tmpdir_factory):
    """Pytest fixture instantiating a new session-scope "data" folder.

    Parameters
    ----------
    tmpdir_factory :
        Pytest fixture for creating temporary directories.
    """
    return tmpdir_factory.mktemp("data")
