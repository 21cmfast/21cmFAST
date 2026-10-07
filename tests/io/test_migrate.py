"""Tests of migrating files written by older versions of 21cmFAST."""

import warnings
from pathlib import Path

import h5py
import pytest

import py21cmfast as p21c
from py21cmfast.cli import app
from py21cmfast.io import compat, h5
from py21cmfast.io.caching import OutputCache
from py21cmfast.io.migrate import migrate_cache, migrate_high_level_file


@pytest.fixture(autouse=True)
def _ignore_input_warnings():
    """Ignore warnings about the (tiny, unusual) inputs of the legacy files."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*R_BUBBLE_MAX.*")
        warnings.filterwarnings("ignore", message=".*maximum halo mass.*")
        warnings.filterwarnings("ignore", message=".*V_CB_MODEL.*")
        yield


def _by_status(results) -> dict[str, list]:
    out = {}
    for res in results:
        out.setdefault(res.status, []).append(res)
    return out


def test_migrate_cache(legacy, tmp_path: Path):
    """All convertible files of a cache are migrated to where the current version expects."""
    legacy.write_cache(tmp_path / "old")
    results = migrate_cache(tmp_path / "old", tmp_path / "new")
    status = _by_status(results)

    assert set(status) <= {"migrated", "unconvertible", "exists"}
    # Some runs in the legacy cache share their inputs (and so their ICs), so the
    # second of these to be migrated finds the first.
    migrated = {r.destination for r in status["migrated"]}
    assert all(r.destination in migrated for r in status.get("exists", []))
    assert {r.kind for r in status["unconvertible"]} == {"RadiationFields"}
    assert all("XraySourceBox" in r.message for r in status["unconvertible"])
    assert all(r.version == legacy.version for r in results)

    cache = OutputCache(tmp_path / "new")
    for res in status["migrated"]:
        assert res.destination.is_relative_to(tmp_path / "new")
        with h5py.File(res.destination, "r") as fl:
            version = fl.attrs["21cmFAST-version"]
        assert version == p21c.__version__

        obj = h5.read_output_struct(res.destination)
        assert type(obj).__name__ == res.kind
        assert cache.find_existing(obj) == res.destination


def test_migrate_is_idempotent(legacy_v42, tmp_path: Path):
    """Migrating a cache in-place twice does nothing the second time."""
    legacy_v42.write_cache(tmp_path, run="nohalo")
    first = migrate_cache(tmp_path)
    assert {r.status for r in first} == {"migrated"}

    second = _by_status(migrate_cache(tmp_path))
    # The original files are kept, but their migrated version already exists.
    assert len(second["exists"]) == len(first)
    assert len(second["up-to-date"]) == len(first)


def test_overwrite(legacy_v42, tmp_path: Path):
    """Existing files are only replaced if asked for."""
    legacy_v42.write_cache(tmp_path / "old", run="nohalo")
    migrate_cache(tmp_path / "old", tmp_path / "new")
    results = migrate_cache(tmp_path / "old", tmp_path / "new", overwrite=True)
    assert {r.status for r in results} == {"migrated"}


def test_dry_run(legacy_v42, tmp_path: Path):
    """A dry run writes nothing."""
    legacy_v42.write_cache(tmp_path / "old", run="nohalo")
    results = migrate_cache(tmp_path / "old", tmp_path / "new", dry_run=True)
    assert {r.status for r in results} == {"migrated"}
    assert not (tmp_path / "new").exists()


def test_kinds(legacy_v42, tmp_path: Path):
    """Only the requested kinds are migrated (by current or old names)."""
    legacy_v42.write_cache(tmp_path / "old", run="halo")
    results = _by_status(
        migrate_cache(
            tmp_path / "old", tmp_path / "new", kinds=["InitialConditions", "HaloBox"]
        )
    )
    assert {r.kind for r in results["migrated"]} == {
        "InitialConditions",
        "EmissivityFields",
    }
    assert all(r.message == "kind not requested" for r in results["skipped"])


def test_non_cache_files_are_skipped(legacy_v42, tmp_path: Path):
    """Other files in the cache directory are left alone."""
    legacy_v42.write_coeval(tmp_path / "coeval.h5", "nohalo")
    (tmp_path / "junk.h5").write_text("not hdf5")
    results = migrate_cache(tmp_path)
    assert {r.status for r in results} == {"skipped"}


def test_unsupported_version(legacy_v42, tmp_path: Path):
    """Files from unsupported versions are reported as unconvertible."""
    path = legacy_v42.write_struct(tmp_path / "ics.h5", "nohalo:InitialConditions")
    with h5py.File(path, "a") as fl:
        fl["InitialConditions/OutputFields"].attrs["21cmFAST-version"] = "3.3.1"

    (res,) = migrate_cache(tmp_path)
    assert res.status == "unconvertible"
    assert "not supported" in res.message


def test_missing_arrays_are_not_migrated(legacy_v42, tmp_path: Path):
    """Boxes missing arrays required by the current version are not migrated."""
    path = legacy_v42.write_struct(tmp_path / "ts.h5", "nohalo:TsBox")
    with h5py.File(path, "a") as fl:
        del fl["TsBox/OutputFields/spin_temperature"]

    (res,) = migrate_cache(tmp_path)
    assert res.status == "unconvertible"
    assert "spin_temperature" in res.message
    assert any("spin_temperature" in w for w in res.warnings)


def test_not_a_directory(tmp_path: Path):
    """Caches must be directories."""
    with pytest.raises(NotADirectoryError):
        migrate_cache(tmp_path / "nope")


@pytest.mark.parametrize("kind", ["coeval", "lightcone"])
def test_migrate_high_level_file(legacy, tmp_path: Path, kind: str):
    """High-level outputs are migrated to the current format."""
    old = getattr(legacy, f"write_{kind}")(tmp_path / "old.h5", "halo")
    res = migrate_high_level_file(old, tmp_path / "new.h5")
    assert res.status == "migrated"
    assert res.version == legacy.version

    with warnings.catch_warnings():
        warnings.simplefilter("error", compat.CompatibilityWarning)
        new = h5.load_high_level_simulation(tmp_path / "new.h5")
    assert new == h5.load_high_level_simulation(old)

    # Don't overwrite unless asked.
    assert migrate_high_level_file(old, tmp_path / "new.h5").status == "exists"
    res = migrate_high_level_file(old, tmp_path / "new.h5", overwrite=True)
    assert res.status == "migrated"

    with pytest.raises(ValueError, match="same as the original"):
        migrate_high_level_file(old, old)


class TestMigrateCLI:
    """Tests of the `21cmfast migrate` command."""

    @staticmethod
    def run(*args):
        """Run the CLI without exiting."""
        return app(*args, result_action="return_value")

    def test_explain(self, capsys):
        """--explain prints the format changelog."""
        self.run("migrate --explain")
        out = capsys.readouterr().out
        assert "HaloBox -> EmissivityFields" in out

    def test_no_source(self):
        """A source is needed unless explaining."""
        with pytest.raises(ValueError, match="SOURCE"):
            self.run("migrate")

    def test_missing_source(self, tmp_path: Path):
        """A non-existent source is an error."""
        with pytest.raises(FileNotFoundError):
            self.run(f"migrate {tmp_path / 'nope'}")

    def test_cache(self, legacy_v42, tmp_path: Path, capsys):
        """A cache directory is migrated."""
        legacy_v42.write_cache(tmp_path / "old", run="halo")
        self.run(f"migrate {tmp_path / 'old'} --out {tmp_path / 'new'} --dry-run")
        out = capsys.readouterr().out
        assert "dry run" in out
        assert "XraySourceBox" in out
        assert not (tmp_path / "new").exists()

        self.run(f"migrate {tmp_path / 'old'} --out {tmp_path / 'new'}")
        out = capsys.readouterr().out
        assert "migrated" in out
        assert list((tmp_path / "new").rglob("EmissivityFields.h5"))

    def test_kind(self, legacy_v42, tmp_path: Path):
        """--kind restricts which kinds are migrated."""
        legacy_v42.write_cache(tmp_path / "old", run="halo")
        self.run(
            f"migrate {tmp_path / 'old'} --out {tmp_path / 'new'} "
            "--kind InitialConditions PerturbedField"
        )
        kinds = {p.stem for p in (tmp_path / "new").rglob("*.h5")}
        assert kinds == {"InitialConditions", "PerturbedField"}

    def test_file(self, legacy_v42, tmp_path: Path, capsys):
        """A high-level file is migrated, by default next to the original."""
        old = legacy_v42.write_lightcone(tmp_path / "lc.h5", "nohalo")
        self.run(f"migrate {old}")
        assert (tmp_path / "lc-migrated.h5").exists()
        assert "migrated" in capsys.readouterr().out

        with pytest.raises(ValueError, match="dry-run"):
            self.run(f"migrate {old} --dry-run")
