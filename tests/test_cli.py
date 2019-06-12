import os

import pytest
import yaml
from click.testing import CliRunner

from py21cmmc import UserParams, InitialConditions
from py21cmmc import cli
from py21cmmc._21cmfast.cache_tools import query_cache


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


@pytest.fixture(scope='module')
def user_params():
    return UserParams(HII_DIM=35, DIM=70, BOX_LEN=50)


@pytest.fixture(scope="module")
def cfg(user_params, tmpdirec):
    with open(os.path.join(tmpdirec.strpath, "cfg.yml"), 'w') as f:
        yaml.dump({"user_params": user_params.self}, f)
    return os.path.join(tmpdirec.strpath, "cfg.yml")


def test_init(tmpdirec, user_params, runner, cfg):
    # Run the CLI
    result = runner.invoke(cli.main,
                           ['init', '--direc', tmpdirec.strpath, '--seed', '101010', '--config', cfg])

    if result.exception:
        print(result.output)

    assert result.exit_code == 0

    ic = InitialConditions(user_params=user_params, random_seed=101010)
    assert ic.exists(direc=tmpdirec.strpath)


def test_init_param_override(tmpdirec, runner, cfg):
    # Run the CLI
    result = runner.invoke(cli.main,
                           ['init', '--direc', tmpdirec.strpath, '--seed', '102030', '--config', cfg, '--',
                            'HII_DIM', '37', 'DIM=52', '--OMm', '0.33'])
    assert result.exit_code == 0

    boxes = [res[1] for res in query_cache(direc=tmpdirec.strpath, kind="InitialConditions", seed=102030)]

    assert len(boxes) == 1

    box = boxes[0]

    assert box.user_params.HII_DIM == 37
    assert box.user_params.DIM == 52
    assert box.cosmo_params.OMm == 0.33
    assert box.cosmo_params.cosmo.Om0 == 0.33


def test_perturb(tmpdirec, runner, cfg):
    # Run the CLI
    result = runner.invoke(cli.main,
                           ['perturb', '35', '--direc', tmpdirec.strpath, '--seed', '101010', '--config', cfg])
    assert result.exit_code == 0


def test_spin(tmpdirec, runner, cfg):
    # Run the CLI
    result = runner.invoke(cli.main,
                           ['spin', '34.9', '--direc', tmpdirec.strpath, '--seed', '101010', '--config', cfg])
    assert result.exit_code == 0


def test_spin_heatmax(tmpdirec, runner, cfg):
    # Run the CLI
    result = runner.invoke(cli.main,
                           ['spin', '34.9', '--direc', tmpdirec.strpath, '--seed', '101010', '--config', cfg])
    assert result.exit_code == 0


def test_ionize(tmpdirec, runner, cfg):
    # Run the CLI
    result = runner.invoke(cli.main,
                            ['ionize', '35', '--direc', tmpdirec.strpath,
                             '--seed', '101010', '--config', cfg])
    assert result.exit_code == 0


def test_coeval(tmpdirec, runner, cfg):
    # Run the CLI
    result = runner.invoke(cli.main,
                           ['coeval', '35', '--direc', tmpdirec.strpath, '--seed', '101010', '--config', cfg])
    assert result.exit_code == 0


def test_lightcone(tmpdirec, runner, cfg):
    # Run the CLI
    result = runner.invoke(cli.main,
                           ['lightcone', '30', '--direc', tmpdirec.strpath, '--seed', '101010', '--config', cfg,
                            '-X', '35'])
    assert result.exit_code == 0


def test_query(tmpdirec, runner, cfg):
    # Run the CLI
    result = runner.invoke(cli.main, ['query', '--direc', tmpdirec.strpath, '--clear'])  # Clear everything in tmpdirec
    assert result.exit_code == 0

    # Quickly run the default example once again.
    # Run the CLI
    result = runner.invoke(cli.main,
                           ['init', '--direc', tmpdirec.strpath, '--seed', '101010', '--config', cfg])
    assert result.exit_code == 0

    result = runner.invoke(cli.main,
                           ['query', '--direc', tmpdirec.strpath, '--seed', '101010'])

    assert result.output.startswith("1 Data Sets Found:")
    assert 'random_seed:101010' in result.output
    assert "InitialConditions(" in result.output
