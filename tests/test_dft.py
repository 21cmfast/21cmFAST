"""Tests of the C-level FFT wrappers."""

import os
import time

import pytest
from py21cmfast.c_21cmfast import lib

import py21cmfast as p21c


def _cpu_to_wall_ratio(dim: int, n_threads: int, n_repeat: int) -> float:
    """Return the ratio of process CPU time to wall time while running the FFTs.

    With N threads fully busy the ratio should approach N, while a serial FFT gives ~1.
    """
    wall0, cpu0 = time.perf_counter(), time.process_time()
    status = lib.test_dft_cube(dim, n_threads, n_repeat)
    wall1, cpu1 = time.perf_counter(), time.process_time()
    assert status == 0
    return (cpu1 - cpu0) / (wall1 - wall0)


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs at least two cores")
def test_dft_uses_multiple_threads():
    """FFTs requested with 2 threads should keep two (and only two) cores busy."""
    ratio = _cpu_to_wall_ratio(dim=128, n_threads=2, n_repeat=20)
    assert ratio > 1.4, f"CPU/wall time ratio {ratio:.2f} suggests FFTs ran serially"
    assert ratio < 2.5, f"CPU/wall time ratio {ratio:.2f} suggests too many threads ran"


def test_fftw_wisdom_is_reused(tmp_path):
    """Wisdom created by CreateFFTWWisdoms should be importable later in the process.

    FFTW rejects wisdom whose planner configuration differs from the current one, in
    which case CreateFFTWWisdoms silently re-plans (with FFTW_PATIENT) and overwrites
    the files. Running the ICs in between checks that the FFTW cleanup done by the
    compute functions leaves the planner in a compatible state.
    """
    inputs = p21c.InputParameters.from_template(
        "simple", random_seed=1, node_redshifts=[]
    ).evolve_input_structs(
        HII_DIM=16, DIM=32, BOX_LEN=32, N_THREADS=2, USE_FFTW_WISDOM=True
    )
    with p21c.config.use(wisdoms_path=tmp_path):
        lib.Broadcast_struct_global_all(
            inputs.simulation_options._cstruct,
            inputs.matter_options._cstruct,
            inputs.cosmo_params._cstruct,
            inputs.astro_params._cstruct,
            inputs.astro_options._cstruct,
            inputs.cosmo_tables._cstruct,
        )
        assert lib.CreateFFTWWisdoms() == 0
        wisdoms = {f.name: f.read_bytes() for f in tmp_path.iterdir()}
        assert len(wisdoms) == 4

        p21c.compute_initial_conditions(inputs=inputs, write=False)
        assert lib.CreateFFTWWisdoms() == 0
        assert {f.name: f.read_bytes() for f in tmp_path.iterdir()} == wisdoms
