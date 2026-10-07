"""Fixtures for writing files with the layout of older versions of 21cmFAST.

The layouts are described by the manifests in ``tests/test_data/legacy_formats``,
which were extracted from files written by those versions (see
``devel/legacy_formats``). Arrays are filled with random numbers.
"""

import json
import zlib
from pathlib import Path

import h5py
import numpy as np
import pytest

LEGACY_DIR = Path(__file__).parent.parent / "test_data" / "legacy_formats"
LEGACY_VERSIONS = sorted(p.stem for p in LEGACY_DIR.glob("*.json"))


def _write_dict(group: h5py.Group, dct: dict):
    """Write a dict to attrs, with sub-dicts as sub-groups (as 21cmFAST does)."""
    for k, v in dct.items():
        if isinstance(v, dict):
            _write_dict(group.create_group(k), v)
        else:
            group.attrs[k] = v


def _write_inputs(group: h5py.Group, inputs: dict):
    grp = group.create_group("InputParameters")
    inputs = dict(inputs)
    node_redshifts = inputs.pop("node_redshifts")
    grp["node_redshifts"] = (
        h5py.Empty(None) if node_redshifts is None else np.array(node_redshifts)
    )
    _write_dict(grp, inputs)


class LegacyFiles:
    """Writes files with the layout of an old version of 21cmFAST."""

    def __init__(self, version: str):
        with (LEGACY_DIR / f"{version}.json").open() as fl:
            self.manifest = json.load(fl)
        self.version = self.manifest["version"]
        self.rng = np.random.default_rng(1)

    @property
    def structs(self) -> list[str]:
        """The keys (``run:kind``) of the cache files in the manifest."""
        return list(self.manifest["cache"])

    def inputs(self, key: str) -> dict:
        """Return the raw input parameters of a cache file in the manifest."""
        return self.manifest["inputs"][self.manifest["cache"][key]["inputs"]]

    def array_data(self, key: str) -> dict[str, np.ndarray]:
        """Return the (random, but fixed) array data for a struct in the manifest."""
        out = {}
        for name, (shape, dtype) in sorted(
            self.manifest["cache"][key]["arrays"].items()
        ):
            rng = np.random.default_rng(zlib.crc32(f"{key}:{name}".encode()))
            out[name] = rng.uniform(0.1, 1, size=shape).astype(dtype)
        return out

    def write_struct_group(self, group: h5py.Group, key: str):
        """Write a struct from the manifest into a group of a file."""
        desc = self.manifest["cache"][key]
        kind = key.split(":")[1]
        grp = group.create_group(kind)
        if desc["redshift"] is not None:
            grp.attrs["redshift"] = desc["redshift"]
        _write_inputs(grp, self.inputs(key))

        fields = grp.create_group("OutputFields")
        _write_dict(fields, desc["primitives"])
        for name, data in self.array_data(key).items():
            fields[name] = data

    def write_struct(self, path: Path, key: str) -> Path:
        """Write a cache file for a struct in the manifest."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as fl:
            fl.attrs["21cmFAST-version"] = self.version
            self.write_struct_group(fl, key)
        return path

    def write_cache(self, direc: Path, run: str | None = None) -> list[Path]:
        """Write all cache files (of a given run) into a directory.

        The files are not written at the paths that the old version would have used
        (which we can't compute), but that doesn't matter for migration.
        """
        return [
            self.write_struct(direc / key.replace(":", "/") / "file.h5", key)
            for key in self.structs
            if run is None or key.startswith(f"{run}:")
        ]

    def write_coeval(self, path: Path, run: str) -> Path:
        """Write a coeval file for a run in the manifest."""
        desc = self.manifest["coeval"][run]
        with h5py.File(path, "w") as fl:
            _write_dict(fl, desc["attrs"])
            fl.create_group("photon_nonconservation_data")
            for kind in desc["structs"]:
                self.write_struct_group(fl, f"{run}:{kind}")
        return path

    def write_lightcone(self, path: Path, run: str) -> Path:
        """Write a lightcone file for a run in the manifest."""
        desc = self.manifest["lightcone"][run]
        with h5py.File(path, "w") as fl:
            _write_dict(fl, desc["attrs"])
            _write_inputs(fl, self.manifest["inputs"][desc["inputs"]])
            fl.create_group("photon_nonconservation_data")
            for grpname in ("lightcones", "global_quantities"):
                grp = fl.create_group(grpname)
                for name, shape in desc[grpname].items():
                    grp[name] = self.rng.uniform(0.1, 1, size=shape)
            fl["lightcone_distances"] = np.linspace(1000, 1100, desc["n_distances"])
        return path


@pytest.fixture(params=LEGACY_VERSIONS)
def legacy(request) -> LegacyFiles:
    """Writers of files of each legacy version of 21cmFAST."""
    return LegacyFiles(request.param)


@pytest.fixture
def legacy_v42() -> LegacyFiles:
    """Writers of files of 21cmFAST v4.2."""
    return LegacyFiles("v4.2")
