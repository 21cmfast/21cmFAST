"""Test the universal configuration module."""

from pathlib import Path

import pytest
import yaml

import py21cmfast as p21
from py21cmfast._cfg import Config, ConfigurationError


@pytest.fixture(scope="module")
def cfgdir(tmp_path_factory):
    return tmp_path_factory.mktemp("config_test_dir")


def test_config_write(cfgdir):
    with p21.config.use(direc=str(cfgdir)):
        p21.config.write(cfgdir / "config.yml")

    with (cfgdir / "config.yml").open() as fl:
        new_config = yaml.load(fl, Loader=yaml.FullLoader)

    # Test adding new kind of string alias
    new_config["boxdir"] = new_config["direc"]
    del new_config["direc"]

    with (cfgdir / "config.yml").open("w") as fl:
        yaml.dump(new_config, fl)

    with pytest.raises(ConfigurationError):
        new_config = Config.load(cfgdir / "config.yml")


@pytest.fixture
def restore_backend():
    """Re-sync the C config_settings with the global config after a test.

    Constructing a new ``Config`` writes its values into the process-global C struct.
    """
    yield
    for k in p21.config._c_config_settings:
        p21.config._pass_to_backend(k, p21.config[k])


def test_config_write_paths(cfgdir):
    fname = cfgdir / "new_subdir" / "config_paths.yml"
    with p21.config.use(direc=str(cfgdir)):
        p21.config.write(fname)

        with fname.open() as fl:
            written = yaml.load(fl, Loader=yaml.FullLoader)

        assert written["direc"] == str(p21.config["direc"])
        assert written["wisdoms_path"] == str(p21.config["wisdoms_path"])
        assert written["external_table_path"] == str(p21.config["external_table_path"])


def test_config_write_without_fname():
    with pytest.raises(ValueError, match="No file name"):
        p21.config.write()


def test_config_load_roundtrip(cfgdir, restore_backend):
    fname = cfgdir / "config_roundtrip.yml"
    with p21.config.use(
        direc=str(cfgdir), HALO_CATALOG_MEM_FACTOR=3.5, safe_read=False
    ):
        p21.config.write(fname)
        expected = p21.config._as_dict()

    loaded = Config.load(fname)
    assert loaded.file_name == fname
    assert loaded["HALO_CATALOG_MEM_FACTOR"] == 3.5
    assert loaded["safe_read"] is False
    assert loaded["direc"] == cfgdir
    assert loaded._as_dict() == expected
    for k in ("direc", "wisdoms_path", "external_table_path"):
        assert isinstance(loaded[k], Path)


def test_config_load_rejects_python_tags(cfgdir):
    fname = cfgdir / "config_unsafe.yml"
    fname.write_text("direc: !!python/name:builtins.print\n")

    with pytest.raises(yaml.constructor.ConstructorError):
        Config.load(fname)


def test_config_load_missing(cfgdir, restore_backend):
    fname = cfgdir / "does_not_exist.yml"
    loaded = Config.load(fname)

    assert loaded.file_name == fname
    assert not fname.exists()
    assert loaded["direc"] == Path(Config._defaults["direc"]).expanduser().absolute()
    for k, v in Config._defaults.items():
        if k != "direc":
            assert loaded[k] == v
