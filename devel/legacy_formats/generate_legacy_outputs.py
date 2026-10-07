"""Generate tiny sample outputs with an old version of 21cmFAST.

This script is run with an *old* version of 21cmFAST installed (in its own virtual
environment), to produce example outputs of that version, which are then summarised
by ``extract_manifest.py`` into ``tests/test_data/legacy_formats/<version>.json``.
It works for v4.0.0, v4.1.1 and v4.2 (which share the same high-level API). To add a
new version (e.g. after a release that changes the file format), install it with::

    git archive <tag> | tar -x -C src-<tag>
    cd src-<tag> && git init && git add -A && git commit -m init && cd ..
    uv venv venv-<tag> --python 3.12
    CC=gcc SETUPTOOLS_SCM_PRETEND_VERSION=<version> uv pip install \
        --python venv-<tag>/bin/python ./src-<tag> h5py

(the ``git init`` is required so that the package data is included), then run::

    venv-<tag>/bin/python generate_legacy_outputs.py out-<tag>
    python extract_manifest.py out-<tag> ../../tests/test_data/legacy_formats/<tag>.json

Usage: python generate_legacy_outputs.py <outdir>
"""

import shutil
import sys
import time
from pathlib import Path

import py21cmfast as p21c

out = Path(sys.argv[1]).absolute()
out.mkdir(parents=True, exist_ok=True)
print("py21cmfast version:", p21c.__version__)

SIZE_KW = {
    "HII_DIM": 10,
    "HIRES_TO_LOWRES_FACTOR": 2,  # DIM = 20
    "BOX_LEN": 20.0,
    "ZPRIME_STEP_FACTOR": 1.3,
    "Z_HEAT_MAX": 15.0,
    "R_BUBBLE_MAX": 6.0,
}


def make_inputs(templates):
    """Create tiny inputs from the given templates."""
    inputs = p21c.InputParameters.from_template(templates, random_seed=1234, **SIZE_KW)
    return inputs.with_logspaced_redshifts(zmin=7.0)


def run_all(label, templates, coeval_z, do_lightcone):
    """Write a full cache, a coeval and (optionally) a lightcone for the templates."""
    t0 = time.time()
    inputs = make_inputs(templates)
    print(f"[{label}] node_redshifts:", inputs.node_redshifts)
    print(f"[{label}] SOURCE_MODEL:", inputs.matter_options.SOURCE_MODEL)

    # templates
    p21c.write_template(inputs, out / f"template_{label}_full.toml")
    try:
        p21c.write_template(
            inputs, out / f"template_{label}_minimal.toml", mode="minimal"
        )
    except Exception as e:  # noqa
        print("minimal template failed:", e)

    # (a) cache dir with everything cached
    cdir = out / f"cache_{label}"
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir()
    cache = p21c.OutputCache(cdir)
    coevals = p21c.run_coeval(
        inputs=inputs,
        out_redshifts=[coeval_z],
        write=p21c.CacheConfig(),
        cache=cache,
        regenerate=True,
    )
    coeval = coevals[0]
    print(
        f"[{label}] coeval z={coeval.redshift}, structs: {list(coeval.output_structs)}"
    )

    # (b) Coeval .save()
    cpath = out / f"coeval_{label}.h5"
    if cpath.exists():
        cpath.unlink()
    coeval.save(cpath)
    # round-trip check with the same version
    c2 = p21c.Coeval.from_file(cpath)
    print(f"[{label}] coeval roundtrip OK:", c2.redshift)

    # (c) Lightcone .save()
    if do_lightcone:
        lcn = p21c.RectilinearLightconer.between_redshifts(
            min_redshift=7.2,
            max_redshift=9.0,
            resolution=inputs.simulation_options.cell_size,
            cosmo=inputs.cosmo_params.cosmo,
            quantities=("brightness_temp", "neutral_fraction", "density"),
        )
        lc = p21c.run_lightcone(
            lightconer=lcn,
            inputs=inputs,
            write=p21c.CacheConfig.off(),
            cache=p21c.OutputCache(out / f"cache_lc_tmp_{label}"),
            regenerate=True,
        )
        lpath = out / f"lightcone_{label}.h5"
        if lpath.exists():
            lpath.unlink()
        lc.save(lpath)
        lc2 = p21c.LightCone.from_file(lpath)
        print(f"[{label}] lightcone roundtrip OK:", lc2.shape)
        shutil.rmtree(out / f"cache_lc_tmp_{label}", ignore_errors=True)
    print(f"[{label}] done in {time.time() - t0:.1f}s")


def run_class():
    """IC and PF with CLASS, relative velocities and explicit SIGMA_8.

    This exercises the cosmo_tables, the SIGMA_8 attribute and the lowres_vcb field.
    """
    cdir = out / "cache_class"
    shutil.rmtree(cdir, ignore_errors=True)
    cdir.mkdir(parents=True)

    inputs = p21c.InputParameters.from_template(
        ["simple", "tiny"],
        random_seed=42,
        HII_DIM=10,
        HIRES_TO_LOWRES_FACTOR=2,
        BOX_LEN=20.0,
        R_BUBBLE_MAX=6.0,
        POWER_SPECTRUM="CLASS",
        USE_RELATIVE_VELOCITIES=True,
        SIGMA_8=0.81,
    )
    cache = p21c.OutputCache(cdir)
    ic = p21c.compute_initial_conditions(
        inputs=inputs, cache=cache, write=True, regenerate=True
    )
    pf = p21c.perturb_field(
        redshift=8.0, initial_conditions=ic, cache=cache, write=True, regenerate=True
    )
    p21c.write_template(inputs, out / "template_class_full.toml")
    print("ok", list(ic.arrays), list(pf.arrays))


def run_perturbed_halos():
    """Write a PerturbedHaloCatalog, which the high-level drivers never cache."""
    cdir = out / "cache_phc"
    shutil.rmtree(cdir, ignore_errors=True)
    cdir.mkdir()

    inputs = p21c.InputParameters.from_template(
        ["latest-discrete", "tiny"],
        random_seed=1234,
        HII_DIM=10,
        HIRES_TO_LOWRES_FACTOR=2,
        BOX_LEN=20.0,
        ZPRIME_STEP_FACTOR=1.3,
        Z_HEAT_MAX=15.0,
        R_BUBBLE_MAX=6.0,
    ).with_logspaced_redshifts(zmin=7.0)

    cache = p21c.OutputCache(cdir)
    kw = {"cache": cache, "write": True, "regenerate": True}
    ic = p21c.compute_initial_conditions(inputs=inputs, **kw)
    hc = p21c.determine_halo_catalog(
        redshift=10.0, initial_conditions=ic, inputs=inputs, **kw
    )
    phc = p21c.perturb_halo_catalog(
        initial_conditions=ic, halo_catalog=hc, inputs=inputs, **kw
    )
    print("wrote", cache.get_path(phc), list(phc.arrays))


# Halo-based: latest-discrete (CHMF-SAMPLER, USE_TS_FLUCT, inhomogeneous recombinations)
run_all("halo", ["latest-discrete", "tiny"], coeval_z=10.0, do_lightcone=True)
# Not halo-based: latest (E-INTEGRAL), USE_TS_FLUCT
run_all("nohalo", ["latest", "tiny"], coeval_z=10.0, do_lightcone=True)
run_class()
run_perturbed_halos()
