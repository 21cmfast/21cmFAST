"""Test the universal configuration module."""

from pathlib import Path

import pytest
import yaml
from py21cmfast.c_21cmfast import ffi, lib

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


@pytest.mark.parametrize("key", ["direc", "wisdoms_path"])
def test_path_keys_expand_user(key):
    """Paths are expanded on setting, since the C backend cannot resolve "~"."""
    expected = Path("~/some/dir").expanduser()

    with p21.config.use(**{key: "~/some/dir"}):
        assert p21.config[key] == expected
        if key in p21.config._c_config_settings:
            backend_value = ffi.string(getattr(lib.config_settings, key)).decode()
            assert backend_value == str(expected)
