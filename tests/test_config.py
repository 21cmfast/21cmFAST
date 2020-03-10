from os import path

import pytest

import yaml

import py21cmfast as p21
from py21cmfast._cfg import Config


def test_config_context(tmpdirec):
    direc = tmpdirec.mkdir("config_test_dir")
    with p21.config.use(direc=direc, write=True):
        init = p21.initial_conditions(user_params={"HII_DIM": 30})

    assert path.exists(path.join(direc.strpath, init.filename))
    assert "config_test_dir" not in p21.config["direc"]


def test_config_write(tmpdirec):
    direc = tmpdirec.mkdir("config_write_dir")

    with p21.config.use(direc=direc):
        p21.config.write(path.join(direc, "config.yml"))

    with open(path.join(direc, "config.yml")) as fl:
        new_config = yaml.load(fl, Loader=yaml.FullLoader)

    # Test adding new kind of string alias
    new_config["boxdir"] = new_config["direc"]
    del new_config["direc"]

    with open(path.join(direc, "config.yml"), "w") as fl:
        yaml.dump(new_config, fl)

    with pytest.warns(UserWarning):
        new_config = Config.load(path.join(direc, "config.yml"))

    assert "boxdir" not in new_config
    assert "direc" in new_config

    with open(path.join(direc, "config.yml")) as fl:
        new_config = yaml.load(fl, Loader=yaml.FullLoader)

    assert "boxdir" not in new_config
    assert "direc" in new_config
