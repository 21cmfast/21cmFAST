"""Unit tests for input structures."""

import pickle
from itertools import chain
from typing import Any, ClassVar

import pytest

from py21cmfast import (
    AstroOptions,
    AstroParams,
    CosmoParams,
    InputParameters,
    MatterOptions,
    SimulationOptions,
    config,
)
from py21cmfast import _templates as tmpl
from py21cmfast.input_serialization import (
    deserialize_inputs,
    prepare_inputs_for_serialization,
)

_TEMPLATES = tmpl.list_templates()
_ALL_ALIASES = list(chain.from_iterable(t["aliases"] for t in _TEMPLATES))


class TestInputStructSubclasses:
    """Tests of the InputStruct object and its subclasses."""

    def setup_class(self):
        """Set up basic objects to have fun with."""
        self.cosmo = CosmoParams(SIGMA_8=0.8)
        self.cosmo2 = CosmoParams(SIGMA_8=0.9)

    def test_hash(self):
        """Test that hashes are unequal beween different objects."""
        assert hash(self.cosmo) != hash(self.cosmo2)

    def test_equality(self):
        """Ensure that the python dict has all fields."""
        assert self.cosmo == self.cosmo
        assert self.cosmo != self.cosmo2

    def test_identity(self):
        """Check that the 'is' operator works as expected."""
        assert self.cosmo is self.cosmo
        assert self.cosmo is not self.cosmo2

        c = CosmoParams(SIGMA_8=0.8)
        assert self.cosmo is not c

    def test_bad_construction(self):
        """Test basic errors of construction via __init__."""
        with pytest.raises(TypeError):
            CosmoParams(self.cosmo, self.cosmo)

        with pytest.raises(TypeError):
            CosmoParams(SimulationOptions())

        with pytest.raises(TypeError):
            CosmoParams(1)

        with pytest.raises(TypeError):
            CosmoParams.new(None, SIGMA_8=0.8, bad_param=1)

    def test_constructed_from_itself(self):
        """Test that constructing an object from itself returns a new equivalent one."""
        c3 = CosmoParams.new(self.cosmo)

        assert self.cosmo == c3
        assert self.cosmo is not c3

    def test_altered_construction(self):
        """Test that altering params upon construction alters the output object."""
        c3 = CosmoParams.new(self.cosmo, SIGMA_8=0.7)

        assert self.cosmo != c3
        assert c3.SIGMA_8 == 0.7

    def test_clone(self):
        """Test that the .clone() method returns a distinct but equivalent clone."""
        u = SimulationOptions()
        v = u.clone()
        assert u == v
        assert u is not v

    def test_repr(self):
        """Test that repr works."""
        assert "SIGMA_8=0.8" in repr(self.cosmo)

    def test_pickle(self):
        """Test that pickling works (important for paralellization)."""
        # Make sure we can pickle/unpickle it.
        c4 = pickle.dumps(self.cosmo)
        c4 = pickle.loads(c4)
        assert self.cosmo == c4

        # Make sure the c data gets loaded fine.
        assert c4.cstruct.SIGMA_8 == self.cosmo.cstruct.SIGMA_8

    def test_asdict(self):
        """Test the asdict() method works."""
        c5 = CosmoParams.new(self.cosmo)
        assert c5 == self.cosmo
        assert c5.asdict() == self.cosmo.asdict()
        assert c5.cdict == self.cosmo.cdict
        assert (
            c5.cdict != c5.asdict()
        )  # not the same because the former doesn't include dynamic parameters.
        assert c5 == self.cosmo


class TestCosmoParams:
    """Tests of CosmoParams."""

    sigma_8 = 1.0
    A_s = 3.0e-9

    def test_defaults(self):
        """Test defaults."""
        cosmo_params = CosmoParams()
        assert cosmo_params.SIGMA_8 == cosmo_params._DEFAULT_SIGMA_8
        assert cosmo_params.A_s == cosmo_params._DEFAULT_A_s

    def test_sigma8(self):
        """Test defaults with sigma8."""
        cosmo_params = CosmoParams(SIGMA_8=self.sigma_8)
        assert self.sigma_8 == cosmo_params.SIGMA_8
        assert cosmo_params.A_s != cosmo_params._DEFAULT_A_s

    def test_A_s(self):
        """Test defaults with A_s."""
        cosmo_params = CosmoParams(A_s=self.A_s)
        assert cosmo_params.SIGMA_8 != cosmo_params._DEFAULT_SIGMA_8
        assert cosmo_params.A_s == self.A_s

    def test_bad_input(self):
        """Test bad inputs."""
        with pytest.raises(ValueError, match="Cannot set both SIGMA_8 and A_s!"):
            CosmoParams(SIGMA_8=self.sigma_8, A_s=self.A_s)


class TestAstroOptions:
    """Tests of AstroOptions."""

    def test_mmin(self):
        """Test that use_mass_dep_zeta sets M_MIN_in_mass."""
        fo = AstroOptions(USE_MASS_DEPENDENT_ZETA=True)
        assert fo.M_MIN_in_Mass

    # Testing all the AstroOptions dependencies, including emitted warnings
    def test_bad_inputs(self):
        """Test possible exceptions when creating the object."""
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


class TestSimulationOptions:
    """Test the SimulationOptions class."""

    def test_non_cubic(self):
        """Test setting non_cubic_factor."""
        up = SimulationOptions()
        up_non_cubic = SimulationOptions(NON_CUBIC_FACTOR=1.5)

        assert up_non_cubic.tot_fft_num_pixels == 1.5 * up.tot_fft_num_pixels
        assert up_non_cubic.HII_tot_num_pixels == up.HII_tot_num_pixels * 1.5
        assert up.cell_size / up.cell_size_hires == up.DIM / up.HII_DIM

    def test_bad_non_cubic_value(self):
        """Test exceptions when setting non_cubic_factor with bad values."""
        with pytest.raises(
            ValueError,
            match="NON_CUBIC_FACTOR \\* DIM and NON_CUBIC_FACTOR \\* HII_DIM must be integers",
        ):
            SimulationOptions(NON_CUBIC_FACTOR=1.1047642)

    def check_attributes_dim(self, s, checks):
        """Check for attributes relating to DIM.

        This is used in several following tests.
        """
        assert checks[0] == s.HIRES_TO_LOWRES_FACTOR
        assert checks[1] == s._HIRES_TO_LOWRES_FACTOR
        assert checks[2] == s.DIM
        assert checks[3] == s._DIM

    @pytest.mark.parametrize(
        ("options", "expected"),
        [
            pytest.param({}, [3, None, 300, None], id="default"),
            pytest.param({"HIRES_TO_LOWRES_FACTOR": 2}, (2, 2, 200, None), id="direct"),
            pytest.param({"DIM": 200}, (2, None, 200, 200), id="explicit"),
        ],
    )
    def test_dim_setting_direct(self, options: dict, expected: tuple[float | None]):
        """Tests of setting DIM vs. HIRES_TO_LOWRES_FACTOR."""
        # By default, uses ratio of 3:
        s = SimulationOptions(HII_DIM=100, **options)
        self.check_attributes_dim(s, expected)

    def test_dim_setting_exceptions(self):
        """Test that exceptions are raised when bad options are passed for DIM."""
        # Exception: explicitly setting both
        with pytest.raises(
            ValueError, match="Cannot set both DIM and HIRES_TO_LOWRES_FACTOR"
        ):
            SimulationOptions(HII_DIM=100, DIM=200, HIRES_TO_LOWRES_FACTOR=2)

        # Exception: Evolve from ratio --> explicit
        with pytest.raises(
            ValueError, match="Cannot set both DIM and HIRES_TO_LOWRES_FACTOR"
        ):
            SimulationOptions(HII_DIM=100, HIRES_TO_LOWRES_FACTOR=2).clone(DIM=200)

        # Exception: Evolve from explicit -> ratio
        with pytest.raises(
            ValueError, match="Cannot set both DIM and HIRES_TO_LOWRES_FACTOR"
        ):
            SimulationOptions(HII_DIM=100, DIM=200).clone(HIRES_TO_LOWRES_FACTOR=3)

    @pytest.mark.parametrize(
        ("direct", "evolved", "expected"),
        [
            pytest.param(
                {}, {"HIRES_TO_LOWRES_FACTOR": 4}, [4, 4, 400, None], id="unset->ratio"
            ),
            pytest.param({}, {"DIM": 300}, [3, None, 300, 300], id="unset->explicit"),
            pytest.param({}, {}, [3, None, 300, None], id="unset->unset"),
            pytest.param(
                {"DIM": 200}, {"DIM": 300}, [3, None, 300, 300], id="explicit->explicit"
            ),
            pytest.param(
                {"HIRES_TO_LOWRES_FACTOR": 2},
                {"HIRES_TO_LOWRES_FACTOR": 3},
                [3, 3, 300, None],
                id="ratio->ratio",
            ),
        ],
    )
    def test_dim_setting_evolve(self, direct, evolved, expected):
        """Test for correct final output after creation then *evolution*."""
        # Evolve from unset -> ratio
        s = SimulationOptions(HII_DIM=100, **direct).clone(**evolved)
        self.check_attributes_dim(s, expected)

    @pytest.mark.parametrize(
        ("direct", "evolved", "expected"),
        [
            pytest.param(
                {}, {"HIRES_TO_LOWRES_FACTOR": 4}, [4, 4, 400, None], id="unset->ratio"
            ),
            pytest.param({}, {"DIM": 300}, [3, None, 300, 300], id="unset->explicit"),
            pytest.param({}, {}, [3, None, 300, None], id="unset->unset"),
            pytest.param(
                {"DIM": 200}, {"DIM": 300}, [3, None, 300, 300], id="explicit->explicit"
            ),
            pytest.param(
                {"HIRES_TO_LOWRES_FACTOR": 2},
                {"HIRES_TO_LOWRES_FACTOR": 3},
                [3, 3, 300, None],
                id="ratio->ratio",
            ),
        ],
    )
    @pytest.mark.parametrize("mode", ["full", "minimal"])
    def test_dim_setting_serialization(
        self,
        direct: dict[str, Any],
        evolved: dict[str, Any],
        expected: tuple[float | None],
        mode: str,
    ):
        """Tests of setting DIM via explicit or ratios, using deserialization.

        This tests for what happens when you write either a TOML file
        or an HDF5 file, then read it in and want to evolve it.
        """
        # To/From TOML
        s = SimulationOptions(HII_DIM=100, **direct)
        inputs = InputParameters(random_seed=1, simulation_options=s)
        dct = prepare_inputs_for_serialization(inputs, mode=mode)
        new = InputParameters(random_seed=1, **deserialize_inputs(dct))

        assert new == inputs

        # Now evolve
        new = new.evolve_input_structs(**evolved)
        self.check_attributes_dim(new.simulation_options, expected)


class TestMatterOptions:
    """Tests of the MatterOptions class."""

    def test_bad_inputs(self):
        """Test that exceptions are raised for bad inputs."""
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


class TestInputParameters:
    """Tests of the InputParameters class."""

    EXCEPTION_CASES: ClassVar = [
        (
            ValueError,
            "R_BUBBLE_MAX is larger than BOX_LEN",
            {
                "astro_params": AstroParams(R_BUBBLE_MAX=100),
                "simulation_options": SimulationOptions(BOX_LEN=50),
            },
        ),
        (
            ValueError,
            "Your R_BUBBLE_MAX is > BOX_LEN/3",
            {
                "astro_params": AstroParams(R_BUBBLE_MAX=20),
                "simulation_options": SimulationOptions(BOX_LEN=50),
                "astro_options": AstroOptions(
                    USE_EXP_FILTER=False, HII_FILTER="sharp-k"
                ),
            },
        ),
        (
            ValueError,
            "You have set USE_MASS_DEPENDENT_ZETA to False but USE_HALO_FIELD is True!",
            {
                "matter_options": MatterOptions(USE_HALO_FIELD=True),
                "astro_options": AstroOptions(USE_MASS_DEPENDENT_ZETA=False),
            },
        ),
        (
            ValueError,
            "USE_HALO_FIELD is not compatible with the redshift-based",
            {
                "matter_options": MatterOptions(USE_HALO_FIELD=True),
                "astro_options": AstroOptions(PHOTON_CONS_TYPE="z-photoncons"),
            },
        ),
        (
            ValueError,
            "USE_EXP_FILTER is not compatible with USE_HALO_FIELD == False",
            {
                "matter_options": MatterOptions(
                    USE_HALO_FIELD=False, HALO_STOCHASTICITY=False
                ),
                "astro_options": AstroOptions(
                    USE_EXP_FILTER=True, USE_UPPER_STELLAR_TURNOVER=False
                ),
            },
        ),
        (
            NotImplementedError,
            "USE_UPPER_STELLAR_TURNOVER is not yet implemented for when USE_HALO_FIELD is False",
            {
                "matter_options": MatterOptions(
                    USE_HALO_FIELD=False, HALO_STOCHASTICITY=False
                ),
                "astro_options": AstroOptions(
                    USE_UPPER_STELLAR_TURNOVER=True, USE_EXP_FILTER=False
                ),
            },
        ),
    ]

    WARNINGS_CASES: ClassVar = [
        (
            "You are setting M_TURN > 8 when USE_MINI_HALOS=True.",
            {
                "astro_params": AstroParams(M_TURN=10),
                "astro_options": AstroOptions(
                    USE_MINI_HALOS=True, USE_TS_FLUCT=True, INHOMO_RECO=True
                ),
            },
        ),
        (
            "Resolution is likely too low for accurate evolved density fields",
            {"simulation_options": SimulationOptions(BOX_LEN=50, DIM=20)},
        ),
        (
            r"This is non\-standard \(but allowed\), and usually occurs upon manual update of INHOMO_RECO",
            {
                "astro_params": AstroParams(R_BUBBLE_MAX=10),
                "simulation_options": SimulationOptions(BOX_LEN=50),
                "astro_options": AstroOptions(INHOMO_RECO=True),
            },
        ),
        (
            "USE_MINI_HALOS needs USE_RELATIVE_VELOCITIES",
            {
                "matter_options": MatterOptions(USE_RELATIVE_VELOCITIES=False),
                "astro_options": AstroOptions(
                    USE_MINI_HALOS=True, INHOMO_RECO=True, USE_TS_FLUCT=True
                ),
            },
        ),
    ]

    def setup_class(self):
        """Create a default InputParameters."""
        self.default = InputParameters(random_seed=1)
        self.default_sigma8 = InputParameters(
            random_seed=1, cosmo_params=CosmoParams(SIGMA_8=1.0)
        )
        self.default_A_s = InputParameters(
            random_seed=1, cosmo_params=CosmoParams(A_s=3.0e-9)
        )

    @pytest.mark.parametrize(("exc", "msg", "kw"), EXCEPTION_CASES)
    def test_validation_exceptions(self, exc, msg, kw):
        """Test various exceptions that can happen on validation."""
        with config.use(ignore_R_BUBBLE_MAX_error=False), pytest.raises(exc, match=msg):
            InputParameters(random_seed=1, **kw)

    @pytest.mark.parametrize(("msg", "kw"), WARNINGS_CASES)
    def test_validation_warnings(self, msg, kw):
        """Test various warnings that can happen on validation."""
        with pytest.warns(UserWarning, match=msg):
            InputParameters(random_seed=1, **kw)

    def test_default(self):
        """Test the default object is, well, default."""
        assert self.default.cosmo_params == CosmoParams.new()
        assert self.default.simulation_options == SimulationOptions.new()
        assert self.default.matter_options == MatterOptions.new()
        assert self.default.astro_params == AstroParams.new()
        assert self.default.astro_options == AstroOptions.new()

    def test_evolve(self):
        """Test that evolve_input_structs does what it says."""
        # Test defaults
        altered_struct = self.default.evolve_input_structs(BOX_LEN=30)
        assert altered_struct.simulation_options.BOX_LEN == 30

        altered_struct = self.default.evolve_input_structs(SIGMA_8=1.0)
        assert altered_struct.cosmo_params.SIGMA_8 == 1.0

        altered_struct = self.default.evolve_input_structs(A_s=3.0e-9)
        assert altered_struct.cosmo_params.A_s == 3.0e-9

        # Test defaults with kwargs
        altered_struct = self.default_sigma8.evolve_input_structs(SIGMA_8=1.0)
        assert altered_struct.cosmo_params.SIGMA_8 == 1.0

        altered_struct = self.default_A_s.evolve_input_structs(A_s=3.0e-9)
        assert altered_struct.cosmo_params.A_s == 3.0e-9

        with pytest.raises(ValueError, match="Cannot set both SIGMA_8 and A_s!"):
            self.default_sigma8.evolve_input_structs(A_s=3.0e-9)

        with pytest.raises(ValueError, match="Cannot set both SIGMA_8 and A_s!"):
            self.default_A_s.evolve_input_structs(SIGMA_8=1.0)

    @pytest.mark.parametrize("template", _ALL_ALIASES)
    def test_from_template(self, template):
        """Test that creation from a template works for all templates."""
        inputs = InputParameters.from_template(template, random_seed=1)
        assert isinstance(inputs, InputParameters)

    def test_bad_input(self):
        """Test that passing a non-existent parameter to evolve raises."""
        with pytest.raises(
            TypeError,
            match="BAD_INPUT is not a valid keyword input.",
        ):
            InputParameters(random_seed=0).evolve_input_structs(BAD_INPUT=True)
