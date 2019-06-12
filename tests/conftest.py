import pytest


@pytest.fixture(scope="session")
def tmpdirec(tmpdir_factory):
    return tmpdir_factory.mktemp("data")