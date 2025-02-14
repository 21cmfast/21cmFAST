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

    with open(cfgdir / "config.yml") as fl:
        new_config = yaml.load(fl, Loader=yaml.FullLoader)

    # Test adding new kind of string alias
    new_config["boxdir"] = new_config["direc"]
    del new_config["direc"]

    with open(cfgdir / "config.yml", "w") as fl:
        yaml.dump(new_config, fl)

    with pytest.raises(ConfigurationError):
        new_config = Config.load(cfgdir / "config.yml")
