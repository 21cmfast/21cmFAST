"""Tests of the C-level FFT wrappers."""

from py21cmfast.c_21cmfast import lib

import py21cmfast as p21c
from py21cmfast.drivers._global_initialization import (
    _GlobalInitManagerSingleton,
    c_state,
)


def test_fftw_wisdom_is_reused(tmp_path):
    """Wisdom created by CreateFFTWWisdoms should be importable later in the process.

    FFTW rejects wisdom whose planner configuration differs from the current one, in
    which case CreateFFTWWisdoms silently re-plans (with FFTW_PATIENT) and overwrites
    the files. Running the ICs in between checks that the FFTW cleanup done by the
    compute functions leaves the planner in a compatible state.
    """
    inputs = p21c.InputParameters.from_template(
        "simple", random_seed=1, node_redshifts=()
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


def test_fftw_wisdom_is_saved_on_broadcast(tmp_path):
    """Broadcasting inputs with USE_FFTW_WISDOM saves the wisdom into wisdoms_path.

    FFTW cannot save into a directory that doesn't exist, and fails silently if so,
    which would mean the (slow) wisdom creation is repeated in every run.
    """
    inputs = p21c.InputParameters.from_template(
        "simple", random_seed=1, node_redshifts=()
    ).evolve_input_structs(
        HII_DIM=16, DIM=32, BOX_LEN=32, N_THREADS=2, USE_FFTW_WISDOM=True
    )
    wisdoms_path = tmp_path / "not" / "yet" / "created"
    _GlobalInitManagerSingleton.free()
    with (
        p21c.config.use(wisdoms_path=wisdoms_path),
        c_state(inputs, broadcast_inputs=True),
    ):
        pass

    assert len(list(wisdoms_path.iterdir())) == 4


def test_fftw_wisdom_is_recreated_for_new_wisdoms_path(tmp_path):
    """Changing wisdoms_path creates wisdom there, even if the inputs are unchanged."""
    inputs = p21c.InputParameters.from_template(
        "simple", random_seed=1, node_redshifts=()
    ).evolve_input_structs(
        HII_DIM=16, DIM=32, BOX_LEN=32, N_THREADS=2, USE_FFTW_WISDOM=True
    )
    _GlobalInitManagerSingleton.free()
    for name in ("first", "second"):
        with (
            p21c.config.use(wisdoms_path=tmp_path / name),
            c_state(inputs, broadcast_inputs=True),
        ):
            pass

        assert len(list((tmp_path / name).iterdir())) == 4
