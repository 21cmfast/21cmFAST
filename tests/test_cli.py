"""Test CLI functionality."""

import tempfile
import tomllib as toml
from pathlib import Path

import h5py
import pytest
from rich.console import Console

from py21cmfast import Coeval, GlobalEvolution, LightCone, cli
from py21cmfast._templates import create_params_from_template
from py21cmfast.cli import Parameters, ParameterSelection, RunParams, _run_setup, app
from py21cmfast.io.h5 import load_high_level_simulation, read_output_struct


def app_noexit(*args, **kwargs):
    """Return the CLI app with SystemExit disabled for testing."""
    return app(*args, **kwargs, result_action="return_value")


class TestTemplateAvail:
    """Tests of the template avail command."""

    def test_that_it_prints(self, capsys):
        """Test that it prints out and contains known template names."""
        app_noexit("template avail")
        output = capsys.readouterr().out
        assert "simple" in output
        assert "Munoz21" in output


class TestTemplateCreate:
    """Test the `template create` command."""

    def test_create_without_explicit_params(self, tmp_path: Path):
        """Test creating from a template without overriding doesn't change anything."""
        app_noexit(
            f"template create --template simple --out {tmp_path / 'simple.toml'}"
        )
        assert (tmp_path / "simple.toml").exists()

        p1 = create_params_from_template(tmp_path / "simple.toml")
        p2 = create_params_from_template("simple")

        # A full TOML may include extra top-level keys (random_seed, node_redshifts)
        # that are not present in the minimal built-in template. Compare only the
        # struct params that are common to both.
        struct_keys = [k for k in p1 if k in p2]
        assert all(p1[k] == p2[k] for k in struct_keys)

    def test_create_with_explicit_params(self, tmp_path: Path):
        """Test that overriding params does change the inputs."""
        out = tmp_path / "simple_plus.toml"
        app_noexit(f"template create --template simple --hii-dim 37 --out {out}")
        assert out.exists()

        p1 = create_params_from_template(out)
        p2 = create_params_from_template("simple")

        assert p1["simulation_options"].HII_DIM == 37
        assert p2["simulation_options"].HII_DIM != 37

    def test_failure_with_both_template_and_file(self, tmp_path):
        """Test that providing both --template and --param-file errors."""
        new = tmp_path / "new.toml"
        app_noexit(f"template create --template simple --out {new}")

        # This should fail
        with pytest.raises(SystemExit):
            app_noexit(
                f"template create --param-file {new} --template simple --out here.toml"
            )

    def test_default_minimal(self, tmp_path: Path):
        """Test that creating a minimal toml with no values works."""
        fl = tmp_path / "test.toml"
        app_noexit(f"template create --mode minimal --out {fl}")

        with fl.open("rb") as _fl:
            data = toml.load(_fl)

        # Since it was minimal, the file should have no values.
        assert all(len(v) == 0 for v in data.values())

    def test_non_existent_directory(self, tmp_path: Path):
        """Test that providing a non-existing directory to write to is OK."""
        out = tmp_path / "parent" / "config.toml"
        app_noexit(f"template create --template simple --out {out}")
        assert out.exists()

    def test_node_redshifts_logspace_zstep(self, tmp_path: Path):
        """Test that --nodez.min/--nodez.max/--nodez.step embed logspaced node_redshifts."""
        out = tmp_path / "logspace.toml"
        app_noexit(
            f"template create --template simple --out {out} "
            "--nodez.min 5.0 --nodez.max 20.0 --nodez.step 1.05"
        )
        assert out.exists()

        p = create_params_from_template(out)
        nz = p["node_redshifts"]
        assert nz is not None
        assert len(nz) > 0
        assert min(nz) == pytest.approx(5.0, rel=1e-4)
        assert max(nz) >= 20.0

    def test_node_redshifts_logspace_nz(self, tmp_path: Path):
        """Test that --nodez.n overrides --nodez.step for logspaced node_redshifts."""
        out = tmp_path / "logspace_nz.toml"
        app_noexit(
            f"template create --template simple --out {out} "
            "--nodez.min 5.0 --nodez.max 20.0 --nodez.n 10"
        )
        assert out.exists()

        p = create_params_from_template(out)
        nz = p["node_redshifts"]
        assert nz is not None
        assert len(nz) == 10
        assert min(nz) == pytest.approx(5.0, rel=1e-4)
        assert max(nz) == pytest.approx(20.0, rel=1e-4)

    def test_node_redshifts_linear_zstep(self, tmp_path: Path):
        """Test that --nodez.spacing linear with --nodez.step produces linear node_redshifts."""
        out = tmp_path / "linear.toml"
        app_noexit(
            f"template create --template simple --out {out} "
            "--nodez.min 5.0 --nodez.max 20.0 --nodez.step 1.5 --nodez.spacing linear"
        )
        assert out.exists()

        p = create_params_from_template(out)
        nz = p["node_redshifts"]
        assert nz is not None
        assert len(nz) > 0
        assert min(nz) == pytest.approx(5.0, rel=1e-4)
        # Spacing should be roughly linear (constant differences)
        diffs = [nz[i] - nz[i + 1] for i in range(len(nz) - 1)]
        assert all(abs(d - diffs[0]) < 1e-8 for d in diffs)

    def test_node_redshifts_linear_nz(self, tmp_path: Path):
        """Test that --nodez.n with --nodez.spacing linear produces exactly nz nodes."""
        out = tmp_path / "linear_nz.toml"
        app_noexit(
            f"template create --template simple --out {out} "
            "--nodez.min 5.0 --nodez.max 20.0 --nodez.n 8 --nodez.spacing linear"
        )
        assert out.exists()

        p = create_params_from_template(out)
        nz = p["node_redshifts"]
        assert nz is not None
        assert len(nz) == 8
        assert min(nz) == pytest.approx(5.0, rel=1e-4)
        assert max(nz) == pytest.approx(20.0, rel=1e-4)

    def test_no_node_redshifts_without_z_args(self, tmp_path: Path):
        """Test that without z-related args, node_redshifts remain as template default."""
        out = tmp_path / "no_nz.toml"
        app_noexit(f"template create --template simple --out {out}")

        p = create_params_from_template(out)
        # simple template has no evolution, so default node_redshifts should be empty
        assert not p.get("node_redshifts")

    def test_random_seed_embedded(self, tmp_path: Path):
        """Test that --random-seed (and --seed alias) embed the seed in the template."""
        out = tmp_path / "seeded.toml"
        app_noexit(f"template create --template simple --out {out} --random-seed 1234")
        assert out.exists()

        p = create_params_from_template(out)
        assert p["random_seed"] == 1234

        # Test the --seed alias too
        out2 = tmp_path / "seeded2.toml"
        app_noexit(f"template create --template simple --out {out2} --seed 5678")
        p2 = create_params_from_template(out2)
        assert p2["random_seed"] == 5678


class TestTemplateShow:
    """Tests of the `template show` command."""

    def test_show_alias(self, capsys):
        """Test that showing an alias works."""
        app_noexit("template show EOS21")
        output = capsys.readouterr().out
        assert "from Munoz" in output


class TestRunSetup:
    """Tests of the _run_setup function."""

    def setup_class(self):
        """Make a default temp dir, and store a simple config TOML."""
        self.tmpdir = Path(tempfile.mkdtemp())

        self.simple = self.tmpdir / "simple.toml"

        # Create a full template in tmpdir
        app_noexit(f"template create --template simple --out {self.simple}")

    def test_unmodified_paramfile(self, capsys):
        """Test that running with an unmodified --param-file doesn't write the fullspec."""
        runp = RunParams(
            param_selection=ParameterSelection(param_file=[self.simple]),
            cachedir=self.tmpdir,
        )
        params = Parameters()  # don't modify the input
        _run_setup(runp, params)
        out = capsys.readouterr().out
        assert "Wrote full configuration" not in out

    def test_unmodified_template(self, capsys):
        """Test that an unmodified --template does write a simple fullspec TOML."""
        runp = RunParams(
            param_selection=ParameterSelection(template=["simple"]),
            cachedir=self.tmpdir,
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
            param_selection=ParameterSelection(template=["simple"]),
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
        assert f"{outcfg}" in out

    def test_unknown_name(self, capsys):
        """Test that modifying params without an explicit file creates a random file."""
        runp = RunParams(
            param_selection=ParameterSelection(template=["simple"]),
            cachedir=self.tmpdir,
        )
        params = Parameters(simulation_options=cli._SimulationOptions(HII_DIM=37))

        _run_setup(runp, params)
        out = capsys.readouterr().out
        assert "Wrote full configuration" in out
        assert "simple.toml" not in out  # got a random uuid


class TestRunICS:
    """Tests of the `run ics` command."""

    def test_basic_run(self, capsys, tmp_path: Path):
        """Test that a simple run creates an InitialConditions.h5 file."""
        app_noexit(
            f"run ics --template simple tiny --cachedir {tmp_path}",
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
        app_noexit(
            f"run ics --template simple tiny --box-len 400 --nodez.min 5.0 --cachedir {tmp_path}"
        )
        out = capsys.readouterr().out
        assert "Resolution is likely too low" in out

    def test_regen(self, capsys, tmp_path):
        """Test that re-running the same box with --regen does actually re-run things."""
        app_noexit(f"run ics --template simple tiny --cachedir {tmp_path}")

        # Now run it again right away with regen
        app_noexit(
            f"run ics --template simple tiny --cachedir {tmp_path} --regenerate",
        )
        out = capsys.readouterr().out
        assert "regeneration is requested. Overriding." in out

        # Run it without regen
        app_noexit(f"run ics --template simple tiny --cachedir {tmp_path}")
        out = capsys.readouterr().out
        assert "skipping computation" in out

    def test_passing_nodez_overwriting_template(self, capsys, tmp_path):
        """Test that passing nodez parameters does overwrite the template node redshifts."""
        app_noexit(
            f"template create --template latest tiny --nodez.min 5.0 --nodez.n 10 --out {tmp_path / 'latest.toml'}"
        )

        with (tmp_path / "latest.toml").open("rb") as f:
            data = toml.load(f)
        assert "node_redshifts" in data
        assert len(data["node_redshifts"]) == 10
        assert min(data["node_redshifts"]) == pytest.approx(5.0, rel=1e-4)

        # Run ICs again, this time overwriting the node redshift info with
        # --nodez.min and --nodez.n
        app_noexit(
            f"run ics --param-file {tmp_path / 'latest.toml'} --cachedir {tmp_path} "
            f"--nodez.min 10.0 --nodez.n 10"
        )

        out = capsys.readouterr().out
        outfile = out.split("conditions to ")[-1].replace("\n", "").strip()
        assert Path(outfile).exists()
        ics = read_output_struct(outfile)
        assert ics.inputs.node_redshifts is not None
        assert len(ics.inputs.node_redshifts) == 10
        assert min(ics.inputs.node_redshifts) == pytest.approx(10.0, rel=1e-4)


class TestRunCoeval:
    """Tests of the `run coeval` command."""

    def test_basic_run(self, capsys, tmp_path: Path):
        """Test that a basic run through produces a coeval*.h5 file."""
        cfile = tmp_path / "coeval_z6.00.h5"
        app_noexit(
            f"run coeval --template simple tiny --cachedir {tmp_path} "
            f"--redshifts 6.0 --out {cfile.parent}",
        )

        output = capsys.readouterr().out
        assert "Saved z=6.00 coeval box" in output

        assert cfile.exists()
        cv = Coeval.from_file(cfile)
        assert cv.redshift == 6.0

    def test_node_redshifts(self, capsys, tmp_path):
        """Test that having nodez in addition to --redshifts works."""
        # We have other node redshifts, but we don't do anything with them.
        app_noexit(
            f"run coeval --template Park19 tiny --zprime-step-factor 1.4 --z-heat-max 15 "
            f"--cachedir {tmp_path} "
            f"--no-save-all-redshifts "
            f"--redshifts 6.0 --out {tmp_path}",
        )
        cfile = tmp_path / "coeval_z6.00.h5"
        assert cfile.exists()

        # This time save everything....
        new = tmp_path / "new"
        new.mkdir()
        app_noexit(
            f"run coeval --template Park19 tiny --cachedir {new} "
            f"--save-all-redshifts "
            f"--redshifts 6.0 --out {new}",
        )
        assert len(list(new.glob("coeval*.h5"))) > 1


class TestRunLightcone:
    """Test the `run lightcone` command."""

    def test_basic_run(self, capsys, tmp_path: Path):
        """Test that a basic run produces a lightcone.h5 file."""
        lcfile = tmp_path / "lightcone.h5"
        app_noexit(
            f"run lightcone --template simple tiny --cachedir {tmp_path} "
            f"--redshift-range 6.0 12.0 --out {lcfile}",
        )

        output = capsys.readouterr().out
        assert "Saved Lightcone" in output

        assert lcfile.exists()
        LightCone.from_file(lcfile)

    def test_non_existent_path(self, tmp_path):
        """Test that a non-existent output path is OK."""
        lcfile = tmp_path / "new" / "lightcone.h5"
        app_noexit(
            f"run lightcone --template simple tiny --cachedir {tmp_path} "
            f"--redshift-range 6.0 12.0 --out {lcfile}",
        )

        assert lcfile.exists()


class TestParamHelp:
    """Test the `run params` command."""

    def test_printing(self, capsys):
        """Test that the (stub) command prints a short useful message."""
        app_noexit("run params")
        assert "Usage: 21cmfast run params --help" in capsys.readouterr().out

    def test_full_help(self, capsys):
        """Test that the --help command prints out all the param help."""
        app_noexit("run params --help")
        out = capsys.readouterr().out

        assert "--hii-dim" in out
        assert "SimulationOptions" in out
        assert "--use-ts-fluct" in out


class TestPRFeature:
    """Test the `dev feature` command."""

    def test_simple_run_through(self, tmp_path: Path):
        """Test that a simple run-through produces the expected plots."""
        template = tmp_path / "small-simple.toml"
        app_noexit(f"template create --template simple tiny --out {template}")
        app_noexit(
            f"dev feature --param-file {template} --redshift-range 6 12 --hmf PS --cachedir {tmp_path} --outdir {tmp_path}"
        )
        assert (tmp_path / "pr_feature_history.pdf").exists()
        assert (tmp_path / "pr_feature_power_history.pdf").exists()
        assert (tmp_path / "pr_feature_lightcone_2d_brightness_temp.pdf").exists()


class TestPredictStructSize:
    """Test the predict struct-size command."""

    def test_relevant_text_is_printed(self, capsys):
        """Test that running the size prediction CLI prints relevant text."""
        app_noexit("predict struct-size --template simple tiny --unit mb")
        # We just want to make sure it runs and prints something reasonable.
        # Detailed correctness is tested in management.py tests.

        out = capsys.readouterr().out
        assert "Output Struct Sizes" in out
        assert "InitialConditions" in out

    def test_cache_off(self, capsys):
        """Test that running with cache off affects the predicted sizes."""
        app_noexit("predict struct-size --template simple tiny --cache-config off")
        out_off = capsys.readouterr().out

        app_noexit("predict struct-size --template simple tiny --cache-config on")
        out_on = capsys.readouterr().out

        assert out_off != out_on
        assert "InitialConditions" in out_on
        assert "InitialConditions" not in out_off


class TestPredictTotalStorageSize:
    """Test the predict total storage-size command."""

    @pytest.mark.parametrize(
        "template",
        ["simple tiny", "Park19 small", "Munoz21 small", "latest-dhalos large"],
    )
    def test_relevant_text_is_printed(self, capsys, template: str):
        """Test that running the total storage size CLI prints relevant text."""
        app_noexit(f"predict storage-size --template {template} --unit gb")
        # We just want to make sure it runs and prints something reasonable.
        # Detailed correctness is tested in management.py tests.

        out = capsys.readouterr().out
        assert "Storage Sizes" in out

    @pytest.mark.parametrize(
        "template",
        ["simple tiny", "Park19 small", "Munoz21 small", "latest-dhalos large"],
    )
    def test_cache_off(self, capsys, template: str):
        """Test that running with cache off affects the predicted total storage size."""
        app_noexit(f"predict storage-size --template {template} --cache-config off")
        out_off = capsys.readouterr().out

        app_noexit("predict storage-size --template simple tiny --cache-config on")
        out_on = capsys.readouterr().out

        assert out_off != out_on
        assert "PerturbedField" in out_on
        assert "PerturbedField" not in out_off


class TestGlobalEvolution:
    """Tests of the global evolution CLI command."""

    def test_basic_run(self, capsys, tmp_path: Path):
        """Test that a basic run produces a lightcone.h5 file."""
        lcfile = tmp_path / "global-evolution.h5"
        app_noexit(
            f"run global --template simple --cachedir {tmp_path} "
            f"--zmin 12.0 --out {lcfile}",
        )

        output = capsys.readouterr().out
        assert "Saved Global Evolution" in output

        assert lcfile.exists()
        GlobalEvolution.from_file(lcfile)

    def test_non_existent_path(self, tmp_path):
        """Test that a non-existent output path is OK."""
        lcfile = tmp_path / "new" / "global-evolution.h5"
        app_noexit(
            f"run global --template simple tiny --cachedir {tmp_path} "
            f"--zmin 10.0 --out {lcfile}",
        )

        assert lcfile.exists()


class TestPlot:
    """Tests of the `plot` command and the `--plot` option on `run` commands."""

    def test_plot_lightcone(self, capsys, tmp_path: Path):
        """Test that `21cmfast plot` works on a saved lightcone."""
        lcfile = tmp_path / "lightcone.h5"
        app_noexit(
            f"run lightcone --template simple tiny --cachedir {tmp_path} "
            f"--redshift-range 6.0 12.0 --out {lcfile}",
        )

        # Without --plot, we should be told how to make one.
        assert "21cmfast plot" in capsys.readouterr().out

        app_noexit(f"plot {lcfile}")
        assert (tmp_path / "lightcone_summary.png").exists()

    def test_plot_explicit_out(self, tmp_path: Path):
        """Test that `--out` puts the plot where we asked for it."""
        lcfile = tmp_path / "lightcone.h5"
        app_noexit(
            f"run lightcone --template simple tiny --cachedir {tmp_path} "
            f"--redshift-range 6.0 12.0 --out {lcfile}",
        )

        out = tmp_path / "plots" / "mylc.png"
        app_noexit(f"plot {lcfile} --out {out}")
        assert out.exists()

    def test_run_lightcone_with_plot(self, tmp_path: Path):
        """Test that `run lightcone --plot` writes a plot next to the data."""
        lcfile = tmp_path / "lightcone.h5"
        app_noexit(
            f"run lightcone --template simple tiny --cachedir {tmp_path} "
            f"--redshift-range 6.0 12.0 --out {lcfile} --plot",
        )

        assert (tmp_path / "lightcone_summary.png").exists()

    def test_run_coeval_with_plot(self, tmp_path: Path):
        """Test that `run coeval --plot` writes a plot next to each coeval box."""
        app_noexit(
            f"run coeval --template simple tiny --cachedir {tmp_path} "
            f"--redshifts 7.0 --out {tmp_path} --plot",
        )

        assert (tmp_path / "coeval_z7.00.h5").exists()
        assert (tmp_path / "coeval_z7.00_summary.png").exists()

        # The same file can be re-plotted afterwards.
        out = tmp_path / "again.png"
        app_noexit(f"plot {tmp_path / 'coeval_z7.00.h5'} --out {out}")
        assert out.exists()

    def test_run_global_with_plot(self, tmp_path: Path):
        """Test that `run global --plot` writes a plot next to the data."""
        out = tmp_path / "global-evolution.h5"
        app_noexit(
            f"run global --template simple --cachedir {tmp_path} --zmin 12.0 "
            f"--out {out} --plot",
        )

        assert (tmp_path / "global-evolution_summary.png").exists()

    def test_plot_unknown_file(self, tmp_path: Path):
        """Test that a file that isn't a 21cmFAST output gives a nice error."""
        bad = tmp_path / "bad.h5"
        with h5py.File(bad, "w") as fl:
            fl.attrs["something_else"] = True

        with pytest.raises(ValueError, match="not a recognized 21cmFAST output"):
            load_high_level_simulation(bad)

    @pytest.mark.parametrize("flags", ["--show", "--plot --show"])
    def test_run_with_show(self, tmp_path: Path, monkeypatch, flags: str):
        """`--show` displays the plot; only `--plot` also writes it to file."""
        shown = []
        monkeypatch.setattr(cli.plt, "show", lambda *a, **kw: shown.append(True))

        out = tmp_path / "global-evolution.h5"
        app_noexit(
            f"run global --template simple --cachedir {tmp_path} --zmin 12.0 "
            f"--out {out} {flags}",
        )

        assert shown
        assert (tmp_path / "global-evolution_summary.png").exists() == (
            "--plot" in flags
        )

    def test_no_plot_gives_hint(self, capsys, tmp_path: Path):
        """Without --plot we should tell the user how to plot later."""
        out = tmp_path / "global-evolution.h5"
        app_noexit(
            f"run global --template simple --cachedir {tmp_path} --zmin 12.0 "
            f"--out {out}",
        )

        assert "21cmfast plot" in capsys.readouterr().out
        assert not list(tmp_path.glob("*.png"))

    def test_saved_plot_path_is_a_link(self, capsys, tmp_path: Path):
        """The saved-plot message carries a clickable file:// URL."""
        out = tmp_path / "global-evolution.h5"
        app_noexit(
            f"run global --template simple --cachedir {tmp_path} --zmin 12.0 "
            f"--out {out} --plot",
        )

        png = tmp_path / "global-evolution_summary.png"
        assert cli._as_url(png) == png.as_uri()
        assert "Saved summary plot" in capsys.readouterr().out


class TestCanShowPlots:
    """Tests of the auto-detection behind a bare (unspecified) --show."""

    def test_not_a_tty(self, monkeypatch):
        """Never show when stdout isn't a terminal -- plt.show() would block."""
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(cli.matplotlib, "get_backend", lambda: "TkAgg")
        assert cli._can_show_plots()

        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False, raising=False)
        assert not cli._can_show_plots()

    def test_non_gui_backend(self, monkeypatch):
        """Never show on a backend that can't open a window."""
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)

        for backend in ("agg", "Agg", "pdf", "svg", "template"):
            monkeypatch.setattr(cli.matplotlib, "get_backend", lambda b=backend: b)
            assert not cli._can_show_plots(), backend

    def test_auto_show_is_used_when_show_unset(self, tmp_path, monkeypatch):
        """A bare run consults _can_show_plots rather than defaulting to False."""
        shown = []
        monkeypatch.setattr(cli.plt, "show", lambda *a, **kw: shown.append(True))
        monkeypatch.setattr(cli, "_can_show_plots", lambda: True)

        out = tmp_path / "global-evolution.h5"
        app_noexit(
            f"run global --template simple --cachedir {tmp_path} --zmin 12.0 "
            f"--out {out}",
        )

        assert shown

    def test_explicit_no_show_beats_auto(self, tmp_path, monkeypatch):
        """--no-show wins even where we could have shown it."""
        shown = []
        monkeypatch.setattr(cli.plt, "show", lambda *a, **kw: shown.append(True))
        monkeypatch.setattr(cli, "_can_show_plots", lambda: True)

        out = tmp_path / "global-evolution.h5"
        app_noexit(
            f"run global --template simple --cachedir {tmp_path} --zmin 12.0 "
            f"--out {out} --no-show",
        )

        assert not shown
