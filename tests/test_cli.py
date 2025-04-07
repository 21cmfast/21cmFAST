"""Test CLI functionality."""

import pytest
import yaml
from click.testing import CliRunner

from py21cmfast import InitialConditions, cli
from py21cmfast.io.caching import OutputCache


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


@pytest.fixture(scope="module")
def cfg(
    default_cosmo_params,
    default_matter_params,
    default_matter_flags,
    default_astro_params,
    default_astro_flags,
    tmpdirec,
):
    with (tmpdirec / "cfg.yml").open("w") as f:
        yaml.dump({"matter_params": default_matter_params.asdict()}, f)
        yaml.dump({"matter_flags": default_matter_flags.asdict()}, f)
        yaml.dump({"cosmo_params": default_cosmo_params.asdict()}, f)
        yaml.dump({"astro_params": default_astro_params.asdict()}, f)
        yaml.dump({"astro_flags": default_astro_flags.asdict()}, f)
    return tmpdirec / "cfg.yml"


def test_init(module_direc, default_input_struct, runner, cfg):
    # Run the CLI. There's no way to turn off writing from the CLI (since that
    # would be useless). We produce a *new* initial conditions box in a new
    # directory and check that it exists. It gets auto-deleted after.
    result = runner.invoke(
        cli.main,
        [
            "init",
            "--direc",
            str(module_direc),
            "--seed",
            "101010",
            "--config",
            cfg,
            "--regen",
        ],
    )

    if result.exception:
        print(result.output)

    assert result.exit_code == 0

    ic = InitialConditions.new(
        inputs=default_input_struct.clone(random_seed=101010),
    )
    cache = OutputCache(module_direc)
    assert cache.find_existing(ic) is not None


# TODO: we could generate a single "prev" box in a temp cache directory to make these tests work
@pytest.mark.skip(
    reason="We have not replaced the recursive behaviour in the CLI tests"
)
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


@pytest.mark.skip(
    reason="We have not replaced the recursive behaviour in the CLI tests"
)
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


@pytest.mark.skip(
    reason="We have not replaced the recursive behaviour in the CLI tests"
)
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
            "--cache-dir",
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
