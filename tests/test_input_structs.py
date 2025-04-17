"""Unit tests for input structures."""

import pickle
import tomllib
import warnings
from pathlib import Path

import pytest

from py21cmfast import (
    AstroOptions,
    AstroParams,
    CosmoParams,
    InputParameters,
    IonizedBox,
    MatterOptions,
    SimulationOptions,
    __version__,
    config,
)


@pytest.fixture(scope="module")
def c():
    return CosmoParams(SIGMA_8=0.8)


def test_diff(c):
    """Ensure that the python dict has all fields."""
    d = CosmoParams(SIGMA_8=0.9)

    assert c is not d
    assert c != d
    assert hash(c) != hash(d)


def test_constructed_the_same(c):
    c2 = CosmoParams(SIGMA_8=0.8)

    assert c is not c2
    assert c == c2
    assert hash(c) == hash(c2)


def test_bad_construction(c):
    with pytest.raises(TypeError):
        CosmoParams(c, c)

    with pytest.raises(TypeError):
        CosmoParams(SimulationOptions())

    with pytest.raises(TypeError):
        CosmoParams(1)

    with pytest.raises(TypeError):
        CosmoParams.new(None, SIGMA_8=0.8, bad_param=1)


def test_constructed_from_itself(c):
    c3 = CosmoParams.new(c)

    assert c == c3
    assert c is not c3


def test_altered_construction(c):
    c3 = CosmoParams.new(c, SIGMA_8=0.7)

    assert c != c3
    assert c3.SIGMA_8 == 0.7


def test_dynamic_variables():
    u = SimulationOptions()
    assert u.DIM == 3 * u.HII_DIM

    u = u.clone(DIM=200)

    assert u.DIM == 200


def test_clone():
    u = SimulationOptions()
    v = u.clone()
    assert u == v
    assert u is not v


def test_repr(c):
    assert "SIGMA_8=0.8" in repr(c)


def test_pickle(c):
    # Make sure we can pickle/unpickle it.
    c4 = pickle.dumps(c)
    c4 = pickle.loads(c4)
    assert c == c4

    # Make sure the c data gets loaded fine.
    assert c4.cstruct.SIGMA_8 == c.cstruct.SIGMA_8


def test_self(c):
    c5 = CosmoParams.new(c)
    assert c5 == c
    assert c5.asdict() == c.asdict()
    assert c5.cdict == c.cdict
    assert (
        c5.cdict != c5.asdict()
    )  # not the same because the former doesn't include dynamic parameters.
    assert c5 == c


def test_c_structures(c):
    # See if the C structures are behaving correctly
    c2 = CosmoParams(SIGMA_8=0.8)

    assert c is not c2


def test_mmin():
    fo = AstroOptions(USE_MASS_DEPENDENT_ZETA=True)
    assert fo.M_MIN_in_Mass


def test_validation():
    with pytest.raises(ValueError, match="R_BUBBLE_MAX is larger than BOX_LEN"):
        InputParameters(
            cosmo_params=CosmoParams(),
            astro_params=AstroParams(R_BUBBLE_MAX=100),
            simulation_options=SimulationOptions(BOX_LEN=50),
            matter_options=MatterOptions(),
            astro_options=AstroOptions(),
            random_seed=1,
        )

    with (
        config.use(ignore_R_BUBBLE_MAX_error=False),
        pytest.raises(ValueError, match="Your R_BUBBLE_MAX is > BOX_LEN/3"),
    ):
        InputParameters(
            cosmo_params=CosmoParams(),
            astro_params=AstroParams(R_BUBBLE_MAX=20),
            simulation_options=SimulationOptions(BOX_LEN=50),
            matter_options=MatterOptions(),
            astro_options=AstroOptions(USE_EXP_FILTER=False, HII_FILTER="sharp-k"),
            random_seed=1,
        )

    msg = r"This is non\-standard \(but allowed\), and usually occurs upon manual update of INHOMO_RECO"
    with pytest.warns(UserWarning, match=msg):
        InputParameters(
            cosmo_params=CosmoParams(),
            astro_params=AstroParams(R_BUBBLE_MAX=10),
            simulation_options=SimulationOptions(BOX_LEN=50),
            matter_options=MatterOptions(),
            astro_options=AstroOptions(INHOMO_RECO=True),
            random_seed=1,
        )

    with pytest.warns(
        UserWarning, match="USE_MINI_HALOS needs USE_RELATIVE_VELOCITIES"
    ):
        InputParameters(
            cosmo_params=CosmoParams(),
            astro_params=AstroParams(),
            simulation_options=SimulationOptions(),
            matter_options=MatterOptions(USE_RELATIVE_VELOCITIES=False),
            astro_options=AstroOptions(
                USE_MINI_HALOS=True, INHOMO_RECO=True, USE_TS_FLUCT=True
            ),
            random_seed=1,
        )
    with pytest.raises(
        ValueError,
        match="You have set USE_MASS_DEPENDENT_ZETA to False but USE_HALO_FIELD is True!",
    ):
        InputParameters(
            cosmo_params=CosmoParams(),
            astro_params=AstroParams(),
            simulation_options=SimulationOptions(),
            matter_options=MatterOptions(USE_HALO_FIELD=True),
            astro_options=AstroOptions(USE_MASS_DEPENDENT_ZETA=False),
            random_seed=1,
        )
    with pytest.raises(
        ValueError, match="USE_HALO_FIELD is not compatible with the redshift-based"
    ):
        InputParameters(
            cosmo_params=CosmoParams(),
            astro_params=AstroParams(),
            simulation_options=SimulationOptions(),
            matter_options=MatterOptions(USE_HALO_FIELD=True),
            astro_options=AstroOptions(PHOTON_CONS_TYPE="z-photoncons"),
            random_seed=1,
        )
    with pytest.raises(
        ValueError,
        match="USE_EXP_FILTER is not compatible with USE_HALO_FIELD == False",
    ):
        InputParameters(
            cosmo_params=CosmoParams(),
            astro_params=AstroParams(),
            simulation_options=SimulationOptions(),
            matter_options=MatterOptions(
                USE_HALO_FIELD=False, HALO_STOCHASTICITY=False
            ),
            astro_options=AstroOptions(
                USE_EXP_FILTER=True, USE_UPPER_STELLAR_TURNOVER=False
            ),
            random_seed=1,
        )

    with pytest.raises(
        NotImplementedError,
        match="USE_UPPER_STELLAR_TURNOVER is not yet implemented for when USE_HALO_FIELD is False",
    ):
        InputParameters(
            cosmo_params=CosmoParams(),
            astro_params=AstroParams(),
            simulation_options=SimulationOptions(),
            matter_options=MatterOptions(
                USE_HALO_FIELD=False, HALO_STOCHASTICITY=False
            ),
            astro_options=AstroOptions(
                USE_UPPER_STELLAR_TURNOVER=True, USE_EXP_FILTER=False
            ),
            random_seed=1,
        )

    with pytest.warns(
        UserWarning, match="You are setting M_TURN > 8 when USE_MINI_HALOS=True."
    ):
        InputParameters(
            cosmo_params=CosmoParams(),
            astro_params=AstroParams(M_TURN=10),
            simulation_options=SimulationOptions(),
            matter_options=MatterOptions(),
            astro_options=AstroOptions(
                USE_MINI_HALOS=True, USE_TS_FLUCT=True, INHOMO_RECO=True
            ),
            random_seed=1,
        )

    with pytest.warns(
        UserWarning,
        match="Resolution is likely too low for accurate evolved density fields",
    ):
        InputParameters(
            cosmo_params=CosmoParams(),
            astro_params=AstroParams(),
            simulation_options=SimulationOptions(BOX_LEN=50, DIM=20),
            matter_options=MatterOptions(),
            astro_options=AstroOptions(),
            random_seed=1,
        )


def test_simulation_options():
    up = SimulationOptions()
    up_non_cubic = SimulationOptions(NON_CUBIC_FACTOR=1.5)

    assert up_non_cubic.tot_fft_num_pixels == 1.5 * up.tot_fft_num_pixels
    assert up_non_cubic.HII_tot_num_pixels == up.HII_tot_num_pixels * 1.5

    with pytest.raises(
        ValueError,
        match="NON_CUBIC_FACTOR \\* DIM and NON_CUBIC_FACTOR \\* HII_DIM must be integers",
    ):
        up = SimulationOptions(NON_CUBIC_FACTOR=1.1047642)

    assert up.cell_size / up.cell_size_hires == up.DIM / up.HII_DIM


def test_matter_options():
    msg = r"The halo sampler enabled with HALO_STOCHASTICITY requires the use of HMF interpolation tables."
    with pytest.raises(ValueError, match=msg):
        MatterOptions(
            USE_HALO_FIELD=True,
            HALO_STOCHASTICITY=True,
            USE_INTERPOLATION_TABLES="sigma-interpolation",
        )

    msg = r"HALO_STOCHASTICITY is True but USE_HALO_FIELD is False"
    with pytest.raises(ValueError, match=msg):
        MatterOptions(USE_HALO_FIELD=False, HALO_STOCHASTICITY=True)

    msg = r"Can only use 'CLASS' power spectrum with relative velocities"
    with pytest.raises(ValueError, match=msg):
        MatterOptions(USE_RELATIVE_VELOCITIES=True, POWER_SPECTRUM="EH")

    msg = r"The conditional mass functions requied for the halo field"
    with pytest.raises(NotImplementedError, match=msg):
        MatterOptions(USE_HALO_FIELD=True, HMF="WATSON")


# Testing all the AstroOptions dependencies, including emitted warnings
def test_astro_options():
    with pytest.raises(
        ValueError,
        match="The SUBCELL_RSD flag is only effective if APPLY_RSDS is True.",
    ):
        AstroOptions(SUBCELL_RSD=True, APPLY_RSDS=False)

    with pytest.raises(
        ValueError,
        match="You have set USE_MINI_HALOS to True but USE_MASS_DEPENDENT_ZETA is False!",
    ):
        AstroOptions(
            USE_MASS_DEPENDENT_ZETA=False,
            USE_MINI_HALOS=True,
            INHOMO_RECO=True,
            USE_TS_FLUCT=True,
        )
    with pytest.raises(
        ValueError,
        match="M_MIN_in_Mass must be true if USE_MASS_DEPENDENT_ZETA is true.",
    ):
        AstroOptions(USE_MASS_DEPENDENT_ZETA=True, M_MIN_in_Mass=False)

    with pytest.raises(
        ValueError,
        match="You have set USE_MINI_HALOS to True but INHOMO_RECO is False!",
    ):
        AstroOptions(USE_MINI_HALOS=True, USE_TS_FLUCT=True, INHOMO_RECO=False)

    with pytest.raises(
        ValueError,
        match="You have set USE_MINI_HALOS to True but USE_TS_FLUCT is False!",
    ):
        AstroOptions(USE_MINI_HALOS=True, INHOMO_RECO=True, USE_TS_FLUCT=False)

    msg = r"USE_MINI_HALOS is not compatible with the redshift-based"
    with pytest.raises(ValueError, match=msg):
        AstroOptions(
            PHOTON_CONS_TYPE="z-photoncons",
            USE_MINI_HALOS=True,
            INHOMO_RECO=True,
            USE_TS_FLUCT=True,
        )

    with pytest.raises(
        ValueError, match="USE_EXP_FILTER is True but CELL_RECOMB is False"
    ):
        AstroOptions(USE_EXP_FILTER=True, CELL_RECOMB=False)

    with pytest.raises(
        ValueError,
        match="USE_EXP_FILTER can only be used with a real-space tophat HII_FILTER==0",
    ):
        AstroOptions(USE_EXP_FILTER=True, HII_FILTER="sharp-k")


def test_inputstruct_init(default_seed):
    default_struct = InputParameters(random_seed=default_seed)
    altered_struct = default_struct.evolve_input_structs(BOX_LEN=30)

    assert default_struct.cosmo_params == CosmoParams.new()
    assert default_struct.simulation_options == SimulationOptions.new()
    assert default_struct.matter_options == MatterOptions.new()
    assert default_struct.astro_params == AstroParams.new()
    assert default_struct.astro_options == AstroOptions.new()
    assert altered_struct.simulation_options.BOX_LEN == 30


def test_native_template_loading(default_seed):
    template_path = Path(__file__).parent.parent / "src/py21cmfast/templates/"
    with (template_path / "manifest.toml").open("rb") as f:
        manifest = tomllib.load(f)

        # check all files and all aliases work
        for manf_entry in manifest["templates"]:
            for alias in manf_entry["aliases"]:
                assert isinstance(
                    InputParameters.from_template(alias, random_seed=default_seed),
                    InputParameters,
                )
