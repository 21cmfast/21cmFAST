"""Regression tests for DFT/FFTW wisdom path handling."""

import py21cmfast as p21c
from py21cmfast.c_21cmfast import lib
from py21cmfast.drivers._global_initialization import _GlobalInitManagerSingleton


def _small_inputs():
    return p21c.InputParameters(
        random_seed=1,
        simulation_options=p21c.SimulationOptions(HII_DIM=4, DIM=8, BOX_LEN=10),
    )


def test_create_fftw_wisdoms_normal_path(tmp_path):
    """CreateFFTWWisdoms() succeeds with a valid short wisdom path."""
    wisdom_dir = tmp_path / "wisdoms"
    wisdom_dir.mkdir()

    _GlobalInitManagerSingleton.init(inputs=_small_inputs(), broadcast_inputs=True)
    with p21c.config.use(wisdoms_path=str(wisdom_dir)):
        status = lib.CreateFFTWWisdoms()

    assert status == 0


def test_create_fftw_wisdoms_long_path_no_crash():
    """CreateFFTWWisdoms() does not crash when wisdoms_path exceeds the buffer size.

    Regression: before the fix, 4 sprintf() calls in CreateFFTWWisdoms() would overflow
    the 500-byte wisdom_filename buffer with a path longer than ~460 characters. After
    the fix (snprintf + truncation check), the function must return without UB.
    """
    oversized_path = "a" * 600  # well beyond the 500-byte wisdom_filename buffer

    _GlobalInitManagerSingleton.init(inputs=_small_inputs(), broadcast_inputs=True)
    with p21c.config.use(wisdoms_path=oversized_path):
        status = lib.CreateFFTWWisdoms()

    # Must not segfault — reaching this line confirms the buffer overflow fix works.
    assert isinstance(status, int)
