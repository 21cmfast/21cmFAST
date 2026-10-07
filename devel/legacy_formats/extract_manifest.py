"""Extract a compact JSON description of the files written by an old 21cmFAST version.

The tests in ``tests/io/test_compat.py`` use these "manifests" to write small files
that have exactly the layout (input parameters, fields, shapes and dtypes) of files
written by old versions of 21cmFAST, without having to store the (large) files
themselves in the repository.

Usage: python extract_manifest.py <outdir-of-gen.py> <manifest.json>

where ``<outdir-of-gen.py>`` is the output directory of ``gen.py`` (and
``gen_class.py`` and ``gen_phc.py``), run with the old version. This script only needs
h5py and numpy, and can be run with any version of 21cmFAST installed.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

# Halo catalogs are trimmed to this many halos, to keep test files small.
N_HALOS = 5


def _jsonify(val):
    if isinstance(val, np.ndarray):
        return [_jsonify(v) for v in val.tolist()]
    if isinstance(val, bytes):
        return val.decode()
    if isinstance(val, np.bool_ | bool):
        return bool(val)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, list):
        return [_jsonify(v) for v in val]
    return val


def _group_to_dict(grp: h5py.Group) -> dict:
    out = {k: _jsonify(v) for k, v in grp.attrs.items()}
    for k, v in grp.items():
        if isinstance(v, h5py.Group):
            out[k] = _group_to_dict(v)
        elif v.shape is None:
            out[k] = None
        else:
            out[k] = _jsonify(v[()])
    return out


def _input_index(inputs: dict, manifest: dict) -> int:
    """Store each distinct set of inputs only once, returning its index."""
    if inputs not in manifest["inputs"]:
        manifest["inputs"].append(inputs)
    return manifest["inputs"].index(inputs)


def _describe_struct(grp: h5py.Group, manifest: dict) -> dict:
    fields = grp["OutputFields"]
    prims = {k: _jsonify(v) for k, v in fields.attrs.items()}
    arrays = {k: [list(v.shape), str(v.dtype)] for k, v in fields.items()}
    if "n_halos" in prims:
        prims["n_halos"] = prims["buffer_size"] = N_HALOS
        arrays = {k: [[N_HALOS, *shp[1:]], dt] for k, (shp, dt) in arrays.items()}

    return {
        "redshift": _jsonify(grp.attrs.get("redshift")),
        "primitives": prims,
        "arrays": arrays,
        "inputs": _input_index(_group_to_dict(grp["InputParameters"]), manifest),
    }


def main(direc: Path, outfile: Path):
    """Write the manifest of the outputs in `direc` to `outfile`."""
    manifest = {"inputs": [], "cache": {}, "coeval": {}, "lightcone": {}}

    for run in sorted(direc.glob("cache_*")):
        for fl in sorted(run.glob("**/*.h5")):
            with h5py.File(fl, "r") as f:
                manifest["version"] = _jsonify(f.attrs["21cmFAST-version"])
                (kind,) = f.keys()
                manifest["cache"][f"{run.name[6:]}:{kind}"] = _describe_struct(
                    f[kind], manifest
                )

    for fl in sorted(direc.glob("coeval_*.h5")):
        with h5py.File(fl, "r") as f:
            manifest["coeval"][fl.stem[7:]] = {
                "attrs": {k: _jsonify(v) for k, v in f.attrs.items()},
                "structs": sorted(k for k in f if k != "photon_nonconservation_data"),
            }

    for fl in sorted(direc.glob("lightcone_*.h5")):
        with h5py.File(fl, "r") as f:
            manifest["lightcone"][fl.stem[10:]] = {
                "attrs": {k: _jsonify(v) for k, v in f.attrs.items()},
                "inputs": _input_index(_group_to_dict(f["InputParameters"]), manifest),
                "lightcones": {k: list(v.shape) for k, v in f["lightcones"].items()},
                "global_quantities": {
                    k: list(v.shape) for k, v in f["global_quantities"].items()
                },
                "n_distances": len(f["lightcone_distances"]),
            }

    with outfile.open("w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
