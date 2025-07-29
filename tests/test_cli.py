"""Test CLI functionality."""

import tempfile
import tomllib as toml
from pathlib import Path

import pytest
from rich.console import Console

from py21cmfast import Coeval, LightCone, cli
from py21cmfast.cli import Parameters, ParameterSelection, RunParams, _run_setup, app
from py21cmfast.io.h5 import read_output_struct
from py21cmfast.run_templates import create_params_from_template


class TestTemplateAvail:
    """Tests of the template avail command."""

    def test_that_it_prints(self, capsys):
        """Test that it prints out and contains known template names."""
        app("template avail")
        output = capsys.readouterr().out
        assert "simple" in output
        assert "Munoz21" in output


class TestTemplateCreate:
    """Test the `template create` command."""

    def test_create_without_explicit_params(self, tmp_path: Path):
        """Test creating from a template without overriding doesn't change anything."""
        app(f"template create --template simple --out {tmp_path / 'simple.toml'}")
        assert (tmp_path / "simple.toml").exists()

        p1 = create_params_from_template(tmp_path / "simple.toml")
        p2 = create_params_from_template("simple")

        assert all(v == p2[k] for k, v in p1.items())

    def test_create_with_explicit_params(self, tmp_path: Path):
        """Test that overriding params does change the inputs."""
        out = tmp_path / "simple_plus.toml"
        app(f"template create --template simple --hii-dim 37 --out {out}")
        assert out.exists()

        p1 = create_params_from_template(out)
        p2 = create_params_from_template("simple")

        assert p1["simulation_options"].HII_DIM == 37
        assert p2["simulation_options"].HII_DIM != 37

    def test_failure_with_both_template_and_file(self, tmp_path):
        """Test that providing both --template and --param-file errors."""
        new = tmp_path / "new.toml"
        app(f"template create --template simple --out {new}")

        # This should fail
        with pytest.raises(SystemExit):
            app(f"template create --param-file {new} --template simple --out here.toml")

    def test_default_minimal(self, tmp_path: Path):
        """Test that creating a minimal toml with no values works."""
        fl = tmp_path / "test.toml"
        app(f"template create --mode minimal --out {fl}")

        with fl.open("rb") as _fl:
            data = toml.load(_fl)

        # Since it was minimal, the file should have no values.
        assert all(len(v) == 0 for v in data.values())

    def test_non_existent_directory(self, tmp_path: Path):
        """Test that providing a non-existing directory to write to is OK."""
        out = tmp_path / "parent" / "config.toml"
        app(f"template create --template simple --out {out}")
        assert out.exists()


class TestTemplateShow:
    """Tests of the `template show` command."""

    def test_show_alias(self, capsys):
        """Test that showing an alias works."""
        app("template show EOS21")
        output = capsys.readouterr().out
        assert "from Munoz" in output


class TestRunSetup:
    """Tests of the _run_setup function."""

    def setup_class(self):
        """Make a default temp dir, and store a simple config TOML."""
        self.tmpdir = Path(tempfile.mkdtemp())

        self.simple = self.tmpdir / "simple.toml"

        # Create a full template in tmpdir
        app(f"template create --template simple --out {self.simple}")

    def test_unmodified_paramfile(self, capsys):
        """Test that running with an unmodified --param-file doesn't write the fullspec."""
        runp = RunParams(
            param_selection=ParameterSelection(param_file=self.simple),
            cachedir=self.tmpdir,
        )
        params = Parameters()  # don't modify the input
        _run_setup(runp, params)
        out = capsys.readouterr().out
        assert "Wrote full configuration" not in out

    def test_unmodified_template(self, capsys):
        """Test that an unmodified --template does write a simple fullspec TOML."""
        runp = RunParams(
            param_selection=ParameterSelection(template="simple"), cachedir=self.tmpdir
        )
        params = Parameters()  # don't modify the input
        _run_setup(runp, params)
        out = capsys.readouterr().out
        assert "Wrote full configuration" in out
        assert "simple.toml" in out

    def test_explicit_outcfg(self, capsys):
        """Test that directly modifying params and passing an explicit file works."""
        outcfg = self.tmpdir / "custom-name.toml"
        runp = RunParams(
            param_selection=ParameterSelection(template="simple"),
            outcfg=outcfg,
            cachedir=self.tmpdir,
        )
        params = Parameters(simulation_options=cli._SimulationOptions(HII_DIM=37))

        _run_setup(runp, params)
        out = capsys.readouterr().out
        assert "Wrote full configuration" in out
        assert f"{outcfg}" in out

        _run_setup(runp, Parameters())
        out = capsys.readouterr().out
        assert "Wrote full configuration" in out
        assert f"{outcfg}" not in out
        assert "simple.toml" in out

    def test_unknown_name(self, capsys):
        """Test that modifying params without an explicit file creates a random file."""
        runp = RunParams(
            param_selection=ParameterSelection(template="simple"), cachedir=self.tmpdir
        )
        params = Parameters(simulation_options=cli._SimulationOptions(HII_DIM=37))

        _run_setup(runp, params)
        out = capsys.readouterr().out
        assert "Wrote full configuration" in out
        assert "simple.toml" not in out  # got a random uuid


_small_box = (
    "--hii-dim 25 --dim 50 --box-len 50 --zprime-step-factor 1.2 --z-heat-max 20"
)


class TestRunICS:
    """Tests of the `run ics` command."""

    def test_basic_run(self, capsys, tmp_path: Path):
        """Test that a simple run creates an InitialConditions.h5 file."""
        app(
            f"run ics --template simple-small --cachedir {tmp_path}",
            console=Console(width=100),
        )
        output = capsys.readouterr().out
        assert "Saved initial conditions" in output

        outfile = Path(output.split("conditions to ")[-1].replace("\n", ""))
        assert outfile.exists()
        ics = read_output_struct(outfile)
        assert ics.simulation_options.HII_DIM == 32

    def test_warn_formatting(self, tmp_path, capsys):
        """Test that warnings are printed properly."""
        app(
            f"run ics --template simple-small --box-len 400 --zmin 5.0 --cachedir {tmp_path}"
        )
        out = capsys.readouterr().out
        assert "Resolution is likely too low" in out

    def test_regen(self, capsys, tmp_path):
        """Test that re-running the same box with --regen does actually re-run things."""
        app(
            f"run ics --template simple-small --cachedir {tmp_path}",
        )

        # Now run it again right away with regen
        app(
            f"run ics --template simple-small --cachedir {tmp_path} --regenerate",
        )
        out = capsys.readouterr().out
        assert "regeneration is requested. Overriding." in out

        # Run it without regen
        app(
            f"run ics --template simple-small --cachedir {tmp_path}",
        )
        out = capsys.readouterr().out
        assert "skipping computation" in out


class TestRunCoeval:
    """Tests of the `run coeval` command."""

    def test_basic_run(self, capsys, tmp_path: Path):
        """Test that a basic run through produces a coeval*.h5 file."""
        cfile = tmp_path / "coeval_z6.00.h5"
        app(
            f"run coeval --template simple-small --cachedir {tmp_path} "
            f"--redshifts 6.0 --out {cfile.parent}"
        )

        output = capsys.readouterr().out
        assert "Saved z=6.00 coeval box" in output

        assert cfile.exists()
        cv = Coeval.from_file(cfile)
        assert cv.redshift == 6.0

    def test_node_redshifts(self, capsys, tmp_path):
        """Test that having nodez in addition to --redshifts works."""
        # We have other node redshifts, but we don't do anything with them.
        app(
            f"run coeval --template Park19 {_small_box} --cachedir {tmp_path} "
            f"--no-save-all-redshifts "
            f"--redshifts 6.0 --out {tmp_path}"
        )
        cfile = tmp_path / "coeval_z6.00.h5"
        assert cfile.exists()

        # This time save everything....
        new = tmp_path / "new"
        new.mkdir()
        app(
            f"run coeval --template Park19 {_small_box} --cachedir {new} "
            f"--save-all-redshifts "
            f"--redshifts 6.0 --out {new}"
        )
        assert len(list(new.glob("coeval*.h5"))) > 1


class TestRunLightcone:
    """Test the `run lightcone` command."""

    def test_basic_run(self, capsys, tmp_path: Path):
        """Test that a basic run produces a lightcone.h5 file."""
        lcfile = tmp_path / "lightcone.h5"
        app(
            f"run lightcone --template simple-small --cachedir {tmp_path} "
            f"--redshift-range 6.0 12.0 --out {lcfile}"
        )

        output = capsys.readouterr().out
        assert "Saved Lightcone" in output

        assert lcfile.exists()
        LightCone.from_file(lcfile)

    def test_non_existent_path(self, tmp_path):
        """Test that a non-existent output path is OK."""
        lcfile = tmp_path / "new" / "lightcone.h5"
        app(
            f"run lightcone --template simple-small --cachedir {tmp_path} "
            f"--redshift-range 6.0 12.0 --out {lcfile}"
        )

        assert lcfile.exists()


class TestParamHelp:
    """Test the `run params` command."""

    def test_printing(self, capsys):
        """Test that the (stub) command prints a short useful message."""
        app("run params")
        assert "Usage: 21cmfast run params --help" in capsys.readouterr().out

    def test_full_help(self, capsys):
        """Test that the --help command prints out all the param help."""
        app("run params --help")
        out = capsys.readouterr().out

        assert "--hii-dim" in out
        assert "SimulationOptions" in out
        assert "--use-ts-fluct" in out


class TestPRFeature:
    """Test the `dev feature` command."""

    def test_simple_run_through(self, tmp_path: Path):
        """Test that a simple run-through produces the expected plots."""
        template = tmp_path / "small-simple.toml"
        app(f"template create --template simple-small --out {template}")
        app(
            f"dev feature --param-file {template} --redshift-range 6 12 --hmf PS --cachedir {tmp_path} --outdir {tmp_path}"
        )
        assert (tmp_path / "pr_feature_history.pdf").exists()
        assert (tmp_path / "pr_feature_power_history.pdf").exists()
        assert (tmp_path / "pr_feature_lightcone_2d_brightness_temp.pdf").exists()
