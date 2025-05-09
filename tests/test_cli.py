"""Test CLI functionality."""

import tomllib

import pytest
from click.testing import CliRunner

from py21cmfast import InitialConditions, cli
from py21cmfast.io.caching import OutputCache


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


@pytest.fixture(scope="module")
def cfg(
    default_cosmo_params,
    default_simulation_options,
    default_matter_options,
    default_astro_params,
    default_astro_options,
    tmpdirec,
):
    # NOTE:tomllib doesn't have dump?
    def toml_print(val):
        if isinstance(val, str):
            return f'"{val}"'
        elif isinstance(val, bool):
            return str(val).casefold()
        return val

    with (tmpdirec / "cfg.toml").open("w") as f:
        f.write("[CosmoParams]\n")
        [
            f.write(f"{k} = {toml_print(v)}\n")
            for k, v in default_cosmo_params.asdict().items()
        ]
        f.write("\n")
        f.write("[SimulationOptions]\n")
        [
            f.write(f"{k} = {toml_print(v)}\n")
            for k, v in default_simulation_options.asdict().items()
        ]
        f.write("\n")
        f.write("[MatterOptions]\n")
        [
            f.write(f"{k} = {toml_print(v)}\n")
            for k, v in default_matter_options.asdict().items()
        ]
        f.write("\n")
        f.write("[AstroParams]\n")
        [
            f.write(f"{k} = {toml_print(v)}\n")
            for k, v in default_astro_params.asdict().items()
        ]
        f.write("\n")
        f.write("[AstroOptions]\n")
        [
            f.write(f"{k} = {toml_print(v)}\n")
            for k, v in default_astro_options.asdict().items()
        ]
        f.write("\n")
    return tmpdirec / "cfg.toml"


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
