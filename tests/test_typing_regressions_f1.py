"""Regression tests for bugs found while adding type annotations to the I/O modules."""

from pathlib import Path

import deprecation
import h5py
import pytest

from py21cmfast import InputParameters, compute_initial_conditions
from py21cmfast._templates import load_template_file
from py21cmfast.io import _compat_framework, caching, h5


def test_write_outputs_to_group_with_path(tiny_inputs, tmp_path: Path):
    """Writing the fields of a struct to a file given by its path should work.

    The file used to be opened read-only, so that nothing could be written.
    """
    ics = compute_initial_conditions(inputs=tiny_inputs)
    path = tmp_path / "ics.h5"
    h5.write_outputs_to_group(ics, path)

    with h5py.File(path, "r") as fl:
        assert "OutputFields" in fl
        assert set(ics.arrays) <= set(fl["OutputFields"].keys())


def test_read_inputs_with_empty_node_redshifts(tmp_path: Path):
    """Node redshifts written as an empty dataset (for None) should be readable."""
    inputs = InputParameters(random_seed=1, node_redshifts=())
    path = tmp_path / "inputs.h5"
    with h5py.File(path, "w") as fl:
        h5._write_inputs_to_group(inputs, fl)
        del fl["InputParameters/node_redshifts"]
        fl["InputParameters/node_redshifts"] = h5py.Empty(None)

    assert h5.read_inputs(path).node_redshifts == ()


def test_load_template_file_missing_path():
    """A path that is neither a file nor a template should give a helpful error."""
    with pytest.raises(ValueError, match="not found on-disk or in native template"):
        load_template_file(Path("this-file-does-not-exist.toml"))


def test_oldest_supported_version_major_zero():
    """Versions with major version 0 (e.g. untagged checkouts) shouldn't crash."""
    assert (
        _compat_framework.oldest_supported_version("0.0.1.dev715+gee535b94b")
        == _compat_framework.FIRST_FORMAT_VERSION
    )
    assert (
        _compat_framework.support_status("4.2.0", current="0.0.1.dev1") == "supported"
    )


def test_run_cache_rejects_non_mapping_paths():
    """Redshift-dependent paths that are not a mapping used to be silently dropped."""
    with pytest.raises(TypeError, match="Expected a mapping"):
        caching.RunCache(PerturbedField=["a.h5", "b.h5"])

    rc = caching.RunCache(PerturbedField={"6.0": "a.h5"})
    assert rc.PerturbedField == {6.0: Path("a.h5")}


def test_run_cache_get_output_struct_of_non_redshift_kind():
    """Asking for a kind that isn't stored at each redshift should raise clearly."""
    rc = caching.RunCache(InitialConditions="ics.h5", PerturbedField={6.0: "pf.h5"})
    with pytest.raises(ValueError, match="no output structs of kind"):
        rc.get_output_struct_at_z("InitialConditions", z=6.0)
    with pytest.raises(ValueError, match="Must specify either z or index"):
        rc.get_output_struct_at_z("PerturbedField")


@pytest.mark.parametrize("alias", ["halobox", "xray_source_box"])
def test_cache_config_deprecated_alias_warning_location(alias: str):
    """Deprecation warnings for aliases should point at the caller."""
    with pytest.warns(deprecation.DeprecatedWarning) as record:
        caching.CacheConfig(**{alias: False})
    assert record[0].filename == __file__

    with pytest.warns(deprecation.DeprecatedWarning) as record:
        caching.CacheConfig().update(**{alias: False})
    assert record[0].filename == __file__
