"""Various tests of the compute_initial_conditions() function and InitialConditions class."""

from multiprocessing import cpu_count

import numpy as np
import pytest

import py21cmfast as p21c


@pytest.fixture(scope="module")
def ic_hires(default_input_struct) -> p21c.InitialConditions:
    return p21c.compute_initial_conditions(
        inputs=default_input_struct.evolve_input_structs(PERTURB_ON_HIGH_RES=True),
    )


@pytest.fixture(scope="module")
def ic_from_ic(default_input_struct, ic) -> p21c.InitialConditions:
    """Make initial conditions, given the hires density of an initial condition instance."""
    return p21c.compute_initial_conditions(
        inputs=default_input_struct, initial_density=ic.hires_density
    )


@pytest.fixture(scope="module")
def single_pxl_array_mean_zero(default_input_struct) -> np.ndarray:
    """Make a single pixel array, with mean zero."""
    dim = default_input_struct.simulation_options.DIM
    array = -np.ones((dim,) * 3)
    array[dim // 2, dim // 2, dim // 2] = array[0, 0, 0] - array.sum()
    return array


@pytest.fixture(scope="module")
def ic_from_array(
    default_input_struct, single_pxl_array_mean_zero
) -> p21c.InitialConditions:
    """Make initial conditions, given the hires density array of a single pixel."""
    return p21c.compute_initial_conditions(
        inputs=default_input_struct, initial_density=single_pxl_array_mean_zero
    )


def test_box_shape(ic_hires: p21c.InitialConditions, ic: p21c.InitialConditions):
    """Test basic properties of the InitialConditions struct."""
    shape = (35, 35, 35)
    hires_shape = tuple(2 * s for s in shape)
    # test common fields
    for box in (ic, ic_hires):
        assert box.lowres_density.shape == shape
        assert box.hires_density.shape == hires_shape
        assert box.hires_vx_2LPT.shape == hires_shape
        assert box.hires_vy_2LPT.shape == hires_shape
        assert box.hires_vz_2LPT.shape == hires_shape
        assert box.lowres_vcb is None
        assert box.cosmo_params == p21c.CosmoParams()

    # test hires only fields

    assert ic_hires.hires_vx.shape == hires_shape
    assert ic_hires.hires_vy.shape == hires_shape
    assert ic_hires.hires_vz.shape == hires_shape

    assert ic_hires.lowres_vx is None
    assert ic_hires.lowres_vy is None
    assert ic_hires.lowres_vz is None

    assert ic_hires.lowres_vx_2LPT is None
    assert ic_hires.lowres_vy_2LPT is None
    assert ic_hires.lowres_vz_2LPT is None

    # test lowres only fields

    assert ic.lowres_vx.shape == shape
    assert ic.lowres_vy.shape == shape
    assert ic.lowres_vz.shape == shape

    assert ic.lowres_vx_2LPT.shape == shape
    assert ic.lowres_vy_2LPT.shape == shape
    assert ic.lowres_vz_2LPT.shape == shape

    assert ic.hires_vx is None
    assert ic.hires_vy is None
    assert ic.hires_vz is None


def test_modified_cosmo(
    ic: p21c.InitialConditions, default_input_struct: p21c.InputParameters, cache
):
    """Test using a modified cosmology."""
    inputs = default_input_struct.evolve_input_structs(SIGMA_8=0.9)
    ic2 = p21c.compute_initial_conditions(inputs=inputs, cache=cache)

    assert ic2.cosmo_params != ic.cosmo_params
    assert ic2.cosmo_params == inputs.cosmo_params
    assert ic2.cosmo_params.SIGMA_8 == inputs.cosmo_params.SIGMA_8


def test_transfer_function(
    ic: p21c.InitialConditions, default_input_struct: p21c.InputParameters, cache
):
    """Test using a modified transfer function."""
    inputs = default_input_struct.evolve_input_structs(POWER_SPECTRUM="CLASS")
    ic2 = p21c.compute_initial_conditions(inputs=inputs, cache=cache)
    hrd2 = ic2.hires_density
    hrd = ic.hires_density

    rmsnew = np.sqrt(np.mean(hrd2**2))
    rmsdelta = np.sqrt(np.mean((hrd2 - hrd) ** 2))
    assert rmsdelta < rmsnew
    assert rmsnew > 0.0
    assert not np.allclose(hrd2, hrd)


def test_relvels():
    """Test for relative velocity initial conditions."""
    inputs = (
        p21c.InputParameters(random_seed=1)
        .with_logspaced_redshifts()
        .evolve_input_structs(
            HII_DIM=100,
            DIM=300,
            BOX_LEN=300,
            POWER_SPECTRUM="CLASS",
            V_CB_MODEL="FLUCTS",
            USE_MCGS=True,
            USE_TS_FLUCT=True,
            RECOMB_MODEL="inhomogeneous",
            R_BUBBLE_MAX=50.0,
            M_TURN_STELLAR_FEEDBACK=5.0,
            K_MAX_FOR_CLASS=1.0,
            N_THREADS=cpu_count(),  # To make this one a bit faster.
        )
    )
    ic = p21c.compute_initial_conditions(inputs=inputs)

    vcbrms_lowres = np.sqrt(np.mean(ic.lowres_vcb**2))
    vcbavg_lowres = np.mean(ic.lowres_vcb)

    # we test the lowres box
    # rms should be about 30 km/s for LCDM, so we check it is finite and not far off
    # the average should be 0.92*vrms, since it follows a maxwell boltzmann
    assert vcbrms_lowres > 20.0
    assert vcbrms_lowres < 40.0
    assert vcbavg_lowres < 0.97 * vcbrms_lowres
    assert vcbavg_lowres > 0.88 * vcbrms_lowres


@pytest.mark.parametrize(
    "name",
    [
        "lowres_density",
        "hires_density",
        "lowres_vx",
        "lowres_vy",
        "lowres_vz",
        "lowres_vx_2LPT",
        "lowres_vy_2LPT",
        "lowres_vz_2LPT",
    ],
)
def test_initial_density_array(
    ic: p21c.InitialConditions,
    ic_from_ic: p21c.InitialConditions,
    ic_from_array: p21c.InitialConditions,
    single_pxl_array_mean_zero: np.ndarray,
    name: str,
):
    """Test the functionality with the initial_density argument."""
    # Test that the hires_density arrays are exactly the same (by definition)
    assert np.all(ic_from_ic.hires_density == ic.hires_density)

    # Test that the other arrays are close (numerical differences exist due to FFT-IFFT)
    np.testing.assert_allclose(
        getattr(ic, name), getattr(ic_from_ic, name), atol=1e-5, rtol=0.0
    )

    # Test the array we use actually has mean zero
    assert single_pxl_array_mean_zero.mean() == 0.0

    # Test that the hires_density array is exactly our single pixel array input
    assert np.all(ic_from_array.hires_density == single_pxl_array_mean_zero)

    # Test that the arrays are different between the original ic and the ic we got from array
    assert not np.allclose(
        getattr(ic, name), getattr(ic_from_array, name), atol=1e-5, rtol=0.0
    )


def test_bad_initial_density_array(
    default_input_struct: p21c.InitialConditions, single_pxl_array_mean_zero: np.ndarray
):
    """Test bad/weird initial_density array."""
    # Run initial conditions with hires density box that has non-zero mean, just to throw the relevant warning
    with pytest.warns(UserWarning, match="initial_density has mean 1.0"):
        ic_non_zero = p21c.compute_initial_conditions(
            inputs=default_input_struct,
            initial_density=np.ones_like(single_pxl_array_mean_zero),
        )
    assert isinstance(ic_non_zero, p21c.InitialConditions)

    with pytest.raises(
        ValueError,
        match="The shape of your high resolution initial_density is not consistent with inputs!",
    ):
        p21c.compute_initial_conditions(
            inputs=default_input_struct,
            initial_density=np.zeros((single_pxl_array_mean_zero.shape[0] * 2,) * 3),
        )


# ======================================================================================
# Fixed and paired initial conditions
# ======================================================================================


@pytest.fixture(scope="module")
def ic_fixed(default_input_struct) -> p21c.InitialConditions:
    """Initial conditions with fixed mode amplitudes, same seed as ``ic``."""
    return p21c.compute_initial_conditions(
        inputs=default_input_struct.evolve_input_structs(FIX_IC_AMPLITUDES=True),
    )


@pytest.fixture(scope="module")
def ic_paired(default_input_struct) -> p21c.InitialConditions:
    """The paired partner of ``ic`` (all phases shifted by pi)."""
    return p21c.compute_initial_conditions(
        inputs=default_input_struct.evolve_input_structs(INVERT_IC_PHASES=True),
    )


@pytest.fixture(scope="module")
def ic_fixed_paired(default_input_struct) -> p21c.InitialConditions:
    """The paired partner of ``ic_fixed``."""
    return p21c.compute_initial_conditions(
        inputs=default_input_struct.evolve_input_structs(
            FIX_IC_AMPLITUDES=True, INVERT_IC_PHASES=True
        ),
    )


def _fourier_modes(field: np.ndarray, box_len: float) -> np.ndarray:
    """Get the Fourier modes of a real field, normalised as in the C code.

    ``InitialConditions.c`` draws the modes ``box`` and sets the density to
    ``FFT^-1(box) / VOLUME``, so ``box = rfftn(density) * VOLUME / N_cells``.
    """
    volume = box_len**3
    return np.fft.rfftn(field.astype(np.float64)) * volume / field.size


def _kmag(shape: tuple[int, int, int], box_len: float) -> np.ndarray:
    kx = 2 * np.pi * np.fft.fftfreq(shape[0], d=box_len / shape[0])
    ky = 2 * np.pi * np.fft.fftfreq(shape[1], d=box_len / shape[1])
    kz = 2 * np.pi * np.fft.rfftfreq(shape[2], d=box_len / shape[2])
    return np.sqrt(
        kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    )


def test_fixed_amplitudes(
    ic_fixed: p21c.InitialConditions, default_input_struct: p21c.InputParameters
):
    """Every mode of a fixed field has |delta_k|^2 = VOLUME * P(k)."""
    box_len = default_input_struct.simulation_options.BOX_LEN
    modes = _fourier_modes(ic_fixed.hires_density, box_len)
    kmag = _kmag(ic_fixed.hires_density.shape, box_len)
    power = p21c.wrapper.cfuncs.get_matter_power_values(
        inputs=default_input_struct, k_values=kmag[kmag > 0]
    )

    # All modes but the zero mode, including the real-valued Nyquist corners.
    np.testing.assert_allclose(
        np.abs(modes[kmag > 0]) ** 2, box_len**3 * power, rtol=1e-4, atol=0
    )


def test_fixed_keeps_phases(
    ic: p21c.InitialConditions,
    ic_fixed: p21c.InitialConditions,
    default_input_struct: p21c.InputParameters,
):
    """Fixing changes the amplitudes of the modes, but not their phases."""
    box_len = default_input_struct.simulation_options.BOX_LEN
    nonzero = _kmag(ic.hires_density.shape, box_len) > 0
    modes = _fourier_modes(ic.hires_density, box_len)[nonzero]
    modes_fixed = _fourier_modes(ic_fixed.hires_density, box_len)[nonzero]

    np.testing.assert_allclose(
        modes_fixed / np.abs(modes_fixed), modes / np.abs(modes), rtol=0, atol=1e-3
    )
    assert not np.allclose(np.abs(modes_fixed), np.abs(modes), rtol=0.1)


@pytest.mark.parametrize("fixed", [False, True])
def test_paired_ics(
    ic: p21c.InitialConditions,
    ic_paired: p21c.InitialConditions,
    ic_fixed: p21c.InitialConditions,
    ic_fixed_paired: p21c.InitialConditions,
    fixed: bool,
):
    """First-order fields of the paired partner change sign, 2LPT terms do not."""
    original, paired = (ic_fixed, ic_fixed_paired) if fixed else (ic, ic_paired)

    for name in (
        "hires_density",
        "lowres_density",
        "lowres_vx",
        "lowres_vy",
        "lowres_vz",
    ):
        orig = getattr(original, name)
        np.testing.assert_allclose(
            getattr(paired, name),
            -orig,
            rtol=0,
            atol=1e-6 * np.abs(orig).max(),
            err_msg=name,
        )

    for name in ("lowres_vx_2LPT", "lowres_vy_2LPT", "lowres_vz_2LPT"):
        orig = getattr(original, name)
        assert np.abs(orig).max() > 0
        np.testing.assert_allclose(
            getattr(paired, name),
            orig,
            rtol=0,
            atol=1e-6 * np.abs(orig).max(),
            err_msg=name,
        )


def test_paired_evolved_density(
    ic: p21c.InitialConditions, ic_paired: p21c.InitialConditions
):
    """In the evolved density of a pair, the first-order term cancels in the mean.

    That is what cancels the leading-order cosmic variance of pair-averaged statistics.
    The second-order (2LPT) terms are the same in both members, so the pair mean is
    not zero, but it no longer correlates with the linear field.
    """
    density = [
        p21c.perturb_field(redshift=6.0, initial_conditions=ics).density
        for ics in (ic, ic_paired)
    ]
    linear = ic.lowres_density.ravel()
    pair_mean = (density[0] + density[1]) / 2

    def corr(x: np.ndarray) -> float:
        return np.corrcoef(x.ravel(), linear)[0, 1]

    assert corr(density[0]) > 0.5
    assert corr(density[1]) < -0.5
    assert abs(corr(pair_mean)) < 0.05
    assert np.std(pair_mean) > 0.1 * np.std(density[0])


@pytest.mark.filterwarnings("ignore:USE_MCGS is False but V_CB_MODEL:UserWarning")
def test_paired_relative_velocity():
    """The relative velocity |v_cb| is second order: pairing leaves it unchanged."""
    inputs = p21c.InputParameters(random_seed=3).evolve_input_structs(
        HII_DIM=20,
        DIM=40,
        BOX_LEN=40,
        V_CB_MODEL="FLUCTS",
        POWER_SPECTRUM="CLASS",
        K_MAX_FOR_CLASS=1.0,
        SOURCE_MODEL="E-INTEGRAL",
        USE_EXP_FILTER=False,
        CELL_RECOMB=False,
        USE_UPPER_STELLAR_TURNOVER=False,
        USE_METALLICITY=False,
    )
    ic_vcb = p21c.compute_initial_conditions(inputs=inputs)
    ic_vcb_paired = p21c.compute_initial_conditions(
        inputs=inputs.evolve_input_structs(INVERT_IC_PHASES=True)
    )

    np.testing.assert_allclose(
        ic_vcb_paired.lowres_density, -ic_vcb.lowres_density, rtol=0, atol=1e-6
    )
    assert ic_vcb.lowres_vcb.mean() > 0
    np.testing.assert_allclose(
        ic_vcb_paired.lowres_vcb, ic_vcb.lowres_vcb, rtol=1e-6, atol=0
    )


@pytest.mark.parametrize(
    "options",
    [
        {"FIX_IC_AMPLITUDES": True},
        {"INVERT_IC_PHASES": True},
        {"FIX_IC_AMPLITUDES": True, "INVERT_IC_PHASES": True},
    ],
)
def test_fixed_paired_in_cache_hash(
    ic: p21c.InitialConditions, default_input_struct: p21c.InputParameters, options
):
    """Fixed or paired ICs are never confused with the default ones in the cache."""
    inputs = default_input_struct.evolve_input_structs(**options)
    other = p21c.InitialConditions.new(inputs=inputs)
    cache = p21c.OutputCache(".")

    assert not ic._inputs_compatible_with(other)
    assert cache.get_filename(ic) != cache.get_filename(other)


@pytest.mark.parametrize("option", ["FIX_IC_AMPLITUDES", "INVERT_IC_PHASES"])
def test_fixed_paired_with_initial_density(
    default_input_struct: p21c.InputParameters,
    single_pxl_array_mean_zero: np.ndarray,
    option: str,
):
    """The options only apply to fields drawn from the random seed."""
    with pytest.raises(ValueError, match="only apply to initial conditions drawn"):
        p21c.compute_initial_conditions(
            inputs=default_input_struct.evolve_input_structs(**{option: True}),
            initial_density=single_pxl_array_mean_zero,
        )
