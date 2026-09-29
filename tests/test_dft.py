"""Tests of the C-level FFT wrappers."""

from py21cmfast.c_21cmfast import lib

import py21cmfast as p21c


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
