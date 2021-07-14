import pytest

import yaml
from click.testing import CliRunner

from py21cmfast import InitialConditions, cli, query_cache


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


@pytest.fixture(scope="module")
def cfg(default_user_params, tmpdirec):
    with open(tmpdirec / "cfg.yml", "w") as f:
        yaml.dump({"user_params": default_user_params.self}, f)
    return tmpdirec / "cfg.yml"


def test_init(module_direc, default_user_params, runner, cfg):
    # Run the CLI. There's no way to turn off writing from the CLI (since that
    # would be useless). We produce a *new* initial conditions box in a new
    # directory and check that it exists. It gets auto-deleted after.
    result = runner.invoke(
        cli.main,
        ["init", "--direc", str(module_direc), "--seed", "101010", "--config", cfg],
    )

    if result.exception:
        print(result.output)

    assert result.exit_code == 0

    ic = InitialConditions(user_params=default_user_params, random_seed=101010)
    assert ic.exists(direc=str(module_direc))


def test_init_param_override(module_direc, runner, cfg):
    # Run the CLI
    result = runner.invoke(
        cli.main,
        [
            "init",
            "--direc",
            str(module_direc),
            "--seed",
            "102030",
            "--config",
            cfg,
            "--",
            "HII_DIM",
            "37",
            "DIM=52",
            "--OMm",
            "0.33",
        ],
    )
    assert result.exit_code == 0

    boxes = [
        res[1]
        for res in query_cache(
            direc=str(module_direc), kind="InitialConditions", seed=102030
        )
    ]

    assert len(boxes) == 1

    box = boxes[0]

    assert box.user_params.HII_DIM == 37
    assert box.user_params.DIM == 52
    assert box.cosmo_params.OMm == 0.33
    assert box.cosmo_params.cosmo.Om0 == 0.33


def test_perturb(module_direc, runner, cfg):
    # Run the CLI
    result = runner.invoke(
        cli.main,
        [
            "perturb",
            "35",
            "--direc",
            str(module_direc),
            "--seed",
            "101010",
            "--config",
            cfg,
        ],
    )
    assert result.exit_code == 0


def test_spin(module_direc, runner, cfg):
    # Run the CLI
    result = runner.invoke(
        cli.main,
        [
            "spin",
            "34.9",
            "--direc",
            str(module_direc),
            "--seed",
            "101010",
            "--config",
            cfg,
        ],
    )
    print(result.output)
    assert result.exit_code == 0


def test_ionize(module_direc, runner, cfg):
    # Run the CLI
    result = runner.invoke(
        cli.main,
        [
            "ionize",
            "35",
            "--direc",
            str(module_direc),
            "--seed",
            "101010",
            "--config",
            cfg,
        ],
    )
    assert result.exit_code == 0


def test_coeval(module_direc, runner, cfg):
    # Run the CLI
    result = runner.invoke(
        cli.main,
        [
            "coeval",
            "35",
            "--direc",
            str(module_direc),
            "--seed",
            "101010",
            "--config",
            cfg,
        ],
    )
    assert result.exit_code == 0


def test_lightcone(module_direc, runner, cfg):
    # Run the CLI
    result = runner.invoke(
        cli.main,
        [
            "lightcone",
            "30",
            "--direc",
            str(module_direc),
            "--seed",
            "101010",
            "--config",
            cfg,
            "-X",
            "35",
        ],
    )
    assert result.exit_code == 0


def test_query(test_direc, runner, cfg):
    # Quickly run the default example once again.
    # Run the CLI
    result = runner.invoke(
        cli.main,
        ["init", "--direc", str(test_direc), "--seed", "101010", "--config", cfg],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        cli.main, ["query", "--direc", str(test_direc), "--seed", "101010"]
    )

    assert result.output.startswith("1 Data Sets Found:")
    assert "random_seed:101010" in result.output
    assert "InitialConditions(" in result.output
