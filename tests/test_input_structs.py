"""Unit tests for input structures."""

import pickle
from itertools import chain
from typing import Any, ClassVar

import deprecation
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
from py21cmfast.wrapper.inputs import CosmoTables, Table1D

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
        assert c4._cstruct.hlittle == self.cosmo._cstruct.hlittle

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


class TestCosmoTables:
    """Tests of CosmoTables."""

    def setup_class(self):
        """Set up basic objects to have fun with."""
        self.size = 3
        self.x_values = [1, 2, 3]
        self.y_values = [4, 5, 6]
        self.transfer_density = (
            Table1D(
                size=self.size,
                x_values=self.x_values,
                y_values=self.y_values,
            ),
        )
        self.transfer_vcb = Table1D(
            size=self.size,
            x_values=self.x_values,
            y_values=self.y_values,
        )
        self.cosmo_tables = CosmoTables(
            transfer_density=self.transfer_density, transfer_vcb=self.transfer_vcb
        )

    def test_constructed_from_itself(self):
        """Test that constructing an object from itself returns a new equivalent one."""
        cosmo_tables2 = CosmoTables.new(self.cosmo_tables)
        cosmo_tables3 = CosmoTables.new(
            {
                "transfer_density": self.cosmo_tables.transfer_density,
                "transfer_vcb": self.cosmo_tables.transfer_vcb,
            }
        )

        assert self.cosmo_tables == cosmo_tables2
        assert self.cosmo_tables == cosmo_tables3
        assert self.cosmo_tables is not cosmo_tables2
        assert self.cosmo_tables is not cosmo_tables3

    def test_none_input_to_new(self):
        """Test None input to a new CosmoTables."""
        cosmo_tables2 = CosmoTables.new()
        cosmo_tables3 = CosmoTables.new()

        assert self.cosmo_tables != cosmo_tables2
        assert cosmo_tables2 == cosmo_tables3
        assert cosmo_tables2 is not cosmo_tables3

    def test_bad_input(self):
        """Test bad inputs."""
        with pytest.raises(ValueError, match="Cannot instantiate"):
            CosmoParams.new("")


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


class TestAstroParams:
    """Tests of AstroParams."""

    def test_fixed_vavg_deprecated_warning(self):
        """Test that using FIXED_VAVG=True shows deprecation warning."""
        fixed_vavg = 1.0  # dummy value for testing
        with pytest.warns(
            deprecation.DeprecatedWarning, match="FIXED_VAVG is deprecated"
        ):
            astro_params = AstroParams(FIXED_VAVG=fixed_vavg)
        assert fixed_vavg == astro_params.FIXED_VAVG
        assert fixed_vavg == astro_params.V_CB_AVG_DEBUG

    @deprecation.fail_if_not_removed
    def test_fixed_vavg_is_removed(self):
        """Fails when the removed_in version is reached, reminding you to delete FIXED_VAVG."""
        AstroParams(FIXED_VAVG=1.0)

    def test_mturn_deprecated_warning(self):
        """Test that using a non-None value for M_TURN shows deprecation warning."""
        mturn = 8.7  # dummy value for testing
        with pytest.warns(deprecation.DeprecatedWarning, match="M_TURN is deprecated"):
            astro_params = AstroParams(M_TURN=mturn)
        assert mturn == astro_params.M_TURN
        assert mturn == astro_params.M_TURN_STELLAR_FEEDBACK

    @deprecation.fail_if_not_removed
    def test_mturn_is_removed(self):
        """Fails when the removed_in version is reached, reminding you to delete M_TURN."""
        AstroParams(M_TURN=8.7)


class TestAstroOptions:
    """Tests of AstroOptions."""

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"PHOTON_CONS_TYPE": "z-photoncons"}, {"PHOTON_CONS_TYPE": 1}),
            ({"PHOTON_CONS_TYPE": "alpha-photoncons"}, {"PHOTON_CONS_TYPE": 2}),
            ({"PHOTON_CONS_TYPE": "f-photoncons"}, {"PHOTON_CONS_TYPE": 3}),
            (
                {"USE_EXP_FILTER": False, "HII_FILTER": "gaussian"},
                {"HII_FILTER": 2},
            ),
            (
                {"USE_EXP_FILTER": False, "HEAT_FILTER": "sharp-k"},
                {"HEAT_FILTER": 1},
            ),
            (
                {"INTEGRATION_METHOD_ATOMIC": "GSL-QAG"},
                {"INTEGRATION_METHOD_ATOMIC": 0},
            ),
            (
                {"INTEGRATION_METHOD_MINI": "GAMMA-APPROX"},
                {"INTEGRATION_METHOD_MINI": 2},
            ),
        ],
    )
    def test_enum_options_to_cdict(
        self, kwargs: dict[str, Any], expected: dict[str, int]
    ):
        """Check enum-like option fields are mapped to their expected integer values."""
        options = AstroOptions(**kwargs)
        for key, val in expected.items():
            assert options.cdict[key] == val

    # Testing all the AstroOptions dependencies, including emitted warnings
    def test_bad_inputs(self):
        """Test possible exceptions when creating the object."""
        with pytest.raises(
            ValueError,
            match="You have set USE_MINI_HALOS to True but RECOMB_MODEL is 'none'!",
        ):
            AstroOptions(USE_MINI_HALOS=True, USE_TS_FLUCT=True, RECOMB_MODEL="none")

        with pytest.raises(
            ValueError,
            match="You have set USE_MINI_HALOS to True but USE_TS_FLUCT is False!",
        ):
            AstroOptions(
                USE_MINI_HALOS=True, RECOMB_MODEL="inhomogeneous", USE_TS_FLUCT=False
            )

        msg = r"USE_MINI_HALOS is not compatible with the redshift-based"
        with pytest.raises(ValueError, match=msg):
            AstroOptions(
                PHOTON_CONS_TYPE="z-photoncons",
                USE_MINI_HALOS=True,
                RECOMB_MODEL="inhomogeneous",
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

    @pytest.mark.parametrize("use_mini_halos", [True, False])
    def test_reionization_feedback_model_default(self, use_mini_halos):
        """Test that REIONIZATION_FEEDBACK_MODEL defaults to the correct option based on USE_MINI_HALOS."""
        opts = AstroOptions(
            USE_MINI_HALOS=use_mini_halos,
            RECOMB_MODEL="inhomogeneous",
            USE_TS_FLUCT=True,
        )
        correct_model = "BOTH" if use_mini_halos else "NONE"
        assert correct_model == opts.REIONIZATION_FEEDBACK_MODEL

    def test_reionization_feedback_model_choices_valid(self):
        """Test that only valid REIONIZATION_FEEDBACK_MODEL choices are accepted."""
        with pytest.raises(
            ValueError, match="REIONIZATION_FEEDBACK_MODEL must be one of"
        ):
            AstroOptions(REIONIZATION_FEEDBACK_MODEL="invalid")

    @pytest.mark.parametrize("recomb_model", ["none", "homogeneous", "inhomogeneous"])
    def test_recomb_model_basic(self, recomb_model):
        """Test basic RECOMB_MODEL usage without INHOMO_RECO."""
        opts_none = AstroOptions(RECOMB_MODEL=recomb_model)
        assert recomb_model == opts_none.RECOMB_MODEL
        assert opts_none.INHOMO_RECO is False if recomb_model == "none" else True

    def test_inhomo_reco_deprecated_warning(self):
        """Test that using INHOMO_RECO=True shows deprecation warning."""
        with pytest.warns(
            deprecation.DeprecatedWarning, match="INHOMO_RECO is deprecated"
        ):
            opts = AstroOptions(INHOMO_RECO=True)
        assert opts.RECOMB_MODEL == "inhomogeneous"
        assert opts.INHOMO_RECO is True

    @deprecation.fail_if_not_removed
    def test_inhomo_reco_is_removed(self):
        """Fails when the removed_in version is reached, reminding you to delete INHOMO_RECO."""
        AstroOptions(INHOMO_RECO=True)

    @pytest.mark.parametrize("kwargs", [{}, {"INHOMO_RECO": False}])
    def test_inhomo_reco_false_sets_none(self, kwargs):
        """Test that INHOMO_RECO=False (or not provided) sets RECOMB_MODEL='none'."""
        opts = AstroOptions(**kwargs)
        assert opts.RECOMB_MODEL == "none"
        assert opts.INHOMO_RECO is False

    @pytest.mark.parametrize("recomb_model", ["none", "homogeneous", "inhomogeneous"])
    def test_recomb_model_conflict(self, recomb_model):
        """Test error when INHOMO_RECO=False conflicts with RECOMB_MODEL!='none'."""
        inhomo_reco_wrong = recomb_model == "none"
        with pytest.raises(
            ValueError,
            match=f"RECOMB_MODEL is set to '{recomb_model}' but INHOMO_RECO is {inhomo_reco_wrong}",
        ):
            AstroOptions(INHOMO_RECO=inhomo_reco_wrong, RECOMB_MODEL=recomb_model)

    def test_recomb_model_choices_valid(self):
        """Test that only valid RECOMB_MODEL choices are accepted."""
        with pytest.raises(ValueError, match="RECOMB_MODEL must be one of"):
            AstroOptions(RECOMB_MODEL="invalid")

    def test_recomb_model_homo_with_cell_recomb_false(self):
        """Test error when RECOMB_MODEL is 'homogeneous' but CELL_RECOMB is False."""
        with pytest.raises(
            ValueError,
            match="CELL_RECOMB cannot be False when RECOMB_MODEL is 'homogeneous'!",
        ):
            AstroOptions(
                RECOMB_MODEL="homogeneous", CELL_RECOMB=False, USE_EXP_FILTER=False
            )


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

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"HMF": "PS"}, {"HMF": 0}),
            ({"HMF": "DELOS", "SOURCE_MODEL": "E-INTEGRAL"}, {"HMF": 4}),
            ({"POWER_SPECTRUM": "CLASS"}, {"POWER_SPECTRUM": 5}),
            (
                {
                    "USE_INTERPOLATION_TABLES": "sigma-interpolation",
                    "SOURCE_MODEL": "E-INTEGRAL",
                },
                {"USE_INTERPOLATION_TABLES": 1},
            ),
            ({"SAMPLE_METHOD": "PARTITION"}, {"SAMPLE_METHOD": 2}),
            ({"FILTER": "gaussian"}, {"FILTER": 2}),
            ({"HALO_FILTER": "sharp-k"}, {"HALO_FILTER": 1}),
            ({"PERTURB_ALGORITHM": "LINEAR"}, {"PERTURB_ALGORITHM": 0}),
            ({"SOURCE_MODEL": "DEXM-ESF"}, {"SOURCE_MODEL": 3}),
        ],
    )
    def test_enum_options_to_cdict(
        self, kwargs: dict[str, Any], expected: dict[str, int]
    ):
        """Check enum-like option fields are mapped to their expected integer values."""
        options = MatterOptions(**kwargs)
        for key, val in expected.items():
            assert options.cdict[key] == val

    def test_bad_inputs(self):
        """Test that exceptions are raised for bad inputs."""
        msg = r"SOURCE_MODEL settings using the halo sampler require the use of HMF interpolation tables."
        with pytest.raises(ValueError, match=msg):
            MatterOptions(
                SOURCE_MODEL="CHMF-SAMPLER",
                USE_INTERPOLATION_TABLES="sigma-interpolation",
            )

        msg = r"When using V_CB_MODEL='FLUCTS', you must use POWER_SPECTRUM = 'CLASS'!"
        with pytest.raises(ValueError, match=msg):
            MatterOptions(V_CB_MODEL="FLUCTS", POWER_SPECTRUM="EH")

        msg = r"The conditional mass functions requied for the discrete halo field"
        with pytest.raises(NotImplementedError, match=msg):
            MatterOptions(SOURCE_MODEL="CHMF-SAMPLER", HMF="WATSON")

    @pytest.mark.parametrize("v_cb_model", ["NONE", "AVG-AUTO", "FLUCTS", "AVG-DEBUG"])
    def test_v_cb_model_basic(self, v_cb_model):
        """Test basic V_CB_MODEL usage without USE_RELATIVE_VELOCITIES."""
        opts_none = MatterOptions(V_CB_MODEL=v_cb_model)
        assert v_cb_model == opts_none.V_CB_MODEL
        assert (
            opts_none.USE_RELATIVE_VELOCITIES is False if v_cb_model == "NONE" else True
        )

    def test_use_relative_velocities_deprecated_warning(self):
        """Test that using USE_RELATIVE_VELOCITIES=True shows deprecation warning."""
        with pytest.warns(
            deprecation.DeprecatedWarning, match="USE_RELATIVE_VELOCITIES is deprecated"
        ):
            opts = MatterOptions(USE_RELATIVE_VELOCITIES=True)
        assert opts.V_CB_MODEL == "FLUCTS"
        assert opts.USE_RELATIVE_VELOCITIES is True

    @deprecation.fail_if_not_removed
    def test_use_relative_velocities_is_removed(self):
        """Fails when the removed_in version is reached, reminding you to delete USE_RELATIVE_VELOCITIES."""
        MatterOptions(USE_RELATIVE_VELOCITIES=True)

    @pytest.mark.parametrize("kwargs", [{}, {"USE_RELATIVE_VELOCITIES": False}])
    def test_use_relative_velocities_false_sets_none(self, kwargs):
        """Test that USE_RELATIVE_VELOCITIES=False (or not provided) sets V_CB_MODEL='NONE'."""
        opts = MatterOptions(**kwargs)
        assert opts.V_CB_MODEL == "NONE"
        assert opts.USE_RELATIVE_VELOCITIES is False

    @pytest.mark.parametrize("v_cb_model", ["NONE", "AVG-AUTO", "FLUCTS", "AVG-DEBUG"])
    def test_v_cb_model_conflict(self, v_cb_model):
        """Test error when USE_RELATIVE_VELOCITIES=False conflicts with V_CB_MODEL!='NONE'."""
        use_relative_veclocities_wrong = v_cb_model == "NONE"
        with pytest.raises(
            ValueError,
            match=f"V_CB_MODEL is set to '{v_cb_model}' but USE_RELATIVE_VELOCITIES is {use_relative_veclocities_wrong}",
        ):
            MatterOptions(
                USE_RELATIVE_VELOCITIES=use_relative_veclocities_wrong,
                V_CB_MODEL=v_cb_model,
            )

    def test_v_cb_model_choices_valid(self):
        """Test that only valid V_CB_MODEL choices are accepted."""
        with pytest.raises(ValueError, match="V_CB_MODEL must be one of"):
            MatterOptions(V_CB_MODEL="invalid")


class TestInputParameters:
    """Tests of the InputParameters class."""

    EXCEPTION_CASES: ClassVar = [
        (
            ValueError,
            "SOURCE_MODEL == 'CONST-ION-EFF' is not compatible with USE_MINI_HALOS=True",
            {
                "matter_options": MatterOptions(SOURCE_MODEL="CONST-ION-EFF"),
                "astro_options": AstroOptions(
                    USE_MINI_HALOS=True,
                    RECOMB_MODEL="inhomogeneous",
                    USE_TS_FLUCT=True,
                    USE_EXP_FILTER=False,
                    USE_UPPER_STELLAR_TURNOVER=False,
                ),
            },
        ),
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
            "is not compatible with the redshift-based",
            {
                "matter_options": MatterOptions(SOURCE_MODEL="CHMF-SAMPLER"),
                "astro_options": AstroOptions(PHOTON_CONS_TYPE="z-photoncons"),
            },
        ),
        (
            ValueError,
            "LYA_MULTIPLE_SCATTERING is not compatible with SOURCE_MODEL == E-INTEGRAL",
            {
                "matter_options": MatterOptions(
                    SOURCE_MODEL="E-INTEGRAL",
                ),
                "astro_options": AstroOptions(
                    LYA_MULTIPLE_SCATTERING=True,
                    USE_EXP_FILTER=False,
                    USE_UPPER_STELLAR_TURNOVER=False,
                ),
            },
        ),
        (
            ValueError,
            "USE_EXP_FILTER is not compatible with SOURCE_MODEL == E-INTEGRAL",
            {
                "matter_options": MatterOptions(
                    SOURCE_MODEL="E-INTEGRAL",
                ),
                "astro_options": AstroOptions(
                    USE_EXP_FILTER=True, USE_UPPER_STELLAR_TURNOVER=False
                ),
            },
        ),
        (
            NotImplementedError,
            "USE_UPPER_STELLAR_TURNOVER is not yet implemented for SOURCE_MODEL",
            {
                "matter_options": MatterOptions(
                    SOURCE_MODEL="L-INTEGRAL",
                ),
                "astro_options": AstroOptions(
                    USE_UPPER_STELLAR_TURNOVER=True, USE_EXP_FILTER=False
                ),
            },
        ),
    ]

    WARNINGS_CASES: ClassVar = [
        (
            "You are setting M_TURN_STELLAR_FEEDBACK > 8 when USE_MINI_HALOS=True.",
            {
                "astro_params": AstroParams(M_TURN_STELLAR_FEEDBACK=10),
                "astro_options": AstroOptions(
                    USE_MINI_HALOS=True, USE_TS_FLUCT=True, RECOMB_MODEL="inhomogeneous"
                ),
            },
        ),
        (
            "Resolution is likely too low for accurate evolved density fields",
            {"simulation_options": SimulationOptions(BOX_LEN=50, DIM=20)},
        ),
        (
            r"This is non\-standard \(but allowed\), and usually occurs upon manual update of RECOMB_MODEL or R_BUBBLE_MAX",
            {
                "astro_params": AstroParams(R_BUBBLE_MAX=10),
                "simulation_options": SimulationOptions(BOX_LEN=50),
                "astro_options": AstroOptions(RECOMB_MODEL="inhomogeneous"),
            },
        ),
        (
            "USE_MINI_HALOS needs a non-trivial V_CB_MODEL",
            {
                "matter_options": MatterOptions(V_CB_MODEL="NONE"),
                "astro_options": AstroOptions(
                    USE_MINI_HALOS=True, RECOMB_MODEL="inhomogeneous", USE_TS_FLUCT=True
                ),
            },
        ),
        (
            "USE_MINI_HALOS is False but V_CB_MODEL != 'NONE'",
            {
                "matter_options": MatterOptions(V_CB_MODEL="FLUCTS"),
                "astro_options": AstroOptions(
                    USE_MINI_HALOS=False,
                ),
            },
        ),
        (
            "REIONIZATION_FEEDBACK_MODEL is set to 'BOTH' but USE_MINI_HALOS is False! ",
            {
                "astro_options": AstroOptions(
                    USE_MINI_HALOS=False,
                    REIONIZATION_FEEDBACK_MODEL="BOTH",
                )
            },
        ),
        (
            "REIONIZATION_FEEDBACK_MODEL is set to 'MCG' but USE_MINI_HALOS is False! ",
            {
                "astro_options": AstroOptions(
                    USE_MINI_HALOS=False,
                    REIONIZATION_FEEDBACK_MODEL="MCG",
                )
            },
        ),
    ]

    def setup_class(self):
        """Create a default InputParameters."""
        # In test_evolve, we inspect the content of cosmo_tables after evolving the input structs, which causes CLASS to run multiple times,
        # when POWER_SPECTRUM is set to "CLASS". To reduce the time of the test, we set K_MAX_FOR_CLASS to a small value.
        self.simulation_options_default = SimulationOptions.new(K_MAX_FOR_CLASS=0.01)
        self.default = InputParameters(
            random_seed=1, simulation_options=self.simulation_options_default
        )
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

    @pytest.mark.parametrize("fix_vcb_avg", [True, False])
    def test_fix_vcb_avg_deprecated_warning(self, fix_vcb_avg):
        """Test that using FIX_VCB_AVG=True shows deprecation warning."""
        v_cb_model = "AVG-DEBUG" if fix_vcb_avg else "NONE"
        with pytest.warns(
            deprecation.DeprecatedWarning, match="FIX_VCB_AVG is deprecated"
        ):
            inputs = InputParameters(
                random_seed=1,
                astro_options=AstroOptions(FIX_VCB_AVG=fix_vcb_avg),
                matter_options=MatterOptions(V_CB_MODEL=v_cb_model),
            )
        assert v_cb_model == inputs.matter_options.V_CB_MODEL
        assert fix_vcb_avg == inputs.astro_options.FIX_VCB_AVG

    @deprecation.fail_if_not_removed
    def test_fix_vcb_avg_is_removed(self):
        """Fails when the removed_in version is reached, reminding you to delete FIX_VCB_AVG."""
        InputParameters(
            random_seed=1,
            astro_options=AstroOptions(FIX_VCB_AVG=True),
            matter_options=MatterOptions(V_CB_MODEL="AVG-DEBUG"),
        )

    @pytest.mark.parametrize("fix_vcb_avg", [True, False])
    def test_fix_vcb_avg_conflict(self, fix_vcb_avg):
        """Test error when FIX_VCB_AVG conflicts with V_CB_MODEL."""
        v_cb_model_wrong = "NONE" if fix_vcb_avg else "AVG-DEBUG"
        with pytest.raises(
            ValueError,
            match=f"FIX_VCB_AVG={fix_vcb_avg} is not compatible with ",
        ):
            InputParameters(
                random_seed=1,
                astro_options=AstroOptions(FIX_VCB_AVG=fix_vcb_avg),
                matter_options=MatterOptions(V_CB_MODEL=v_cb_model_wrong),
            )

    def test_default(self):
        """Test the default object is, well, default."""
        assert self.default.cosmo_params == CosmoParams.new()
        assert self.default.simulation_options == SimulationOptions.new(
            K_MAX_FOR_CLASS=0.01
        )
        assert self.default.matter_options == MatterOptions.new()
        assert self.default.astro_params == AstroParams.new()
        assert self.default.astro_options == AstroOptions.new()

    def test_evolve(self):
        """Test that evolve_input_structs does what it says."""
        altered_struct = self.default.evolve_input_structs(BOX_LEN=100)
        assert altered_struct.simulation_options.BOX_LEN == 100

        assert self.default.cosmo_tables.ps_norm == self.default.cosmo_params.SIGMA_8
        assert self.default.cosmo_tables.USE_SIGMA_8

        altered_struct_CLASS = self.default.evolve_input_structs(POWER_SPECTRUM="CLASS")
        altered_struct2 = self.default.evolve_input_structs(OMm=0.5)
        altered_struct3 = altered_struct2.evolve_input_structs(POWER_SPECTRUM="EH")

        assert self.default.cosmo_tables.transfer_density is None
        assert altered_struct_CLASS.cosmo_tables.transfer_density is not None
        assert altered_struct_CLASS.cosmo_tables != altered_struct2.cosmo_tables
        assert altered_struct3.cosmo_tables.transfer_density is None
        assert (
            altered_struct_CLASS.cosmo_tables.ps_norm
            == altered_struct_CLASS.cosmo_params.SIGMA_8
        )
        assert altered_struct_CLASS.cosmo_tables.USE_SIGMA_8

        altered_struct = self.default.evolve_input_structs(SIGMA_8=1.0)
        assert altered_struct.cosmo_params.SIGMA_8 == 1.0

        altered_struct = self.default.evolve_input_structs(A_s=3.0e-9)
        assert altered_struct.cosmo_params.A_s == 3.0e-9
        # Even though we work with A_s, transfer function is EH, so we still use sigma8 at the backend
        assert (
            altered_struct.cosmo_tables.ps_norm == altered_struct.cosmo_params.SIGMA_8
        )
        assert altered_struct.cosmo_tables.USE_SIGMA_8

        # Test defaults with kwargs
        altered_struct = self.default_sigma8.evolve_input_structs(SIGMA_8=1.0)
        assert altered_struct.cosmo_params.SIGMA_8 == 1.0
        assert (
            altered_struct.cosmo_tables.ps_norm == altered_struct.cosmo_params.SIGMA_8
        )
        assert altered_struct.cosmo_tables.USE_SIGMA_8

        altered_struct = self.default_A_s.evolve_input_structs(A_s=3.0e-9)
        assert altered_struct.cosmo_params.A_s == 3.0e-9
        # Even though we work with A_s, transfer function is EH, so we still use sigma8 at the backend
        assert (
            altered_struct.cosmo_tables.ps_norm == altered_struct.cosmo_params.SIGMA_8
        )
        assert altered_struct.cosmo_tables.USE_SIGMA_8

        # Check that we can set A_s over the CLASS inputs and that we actually use it at the backend
        altered_struct = altered_struct_CLASS.evolve_input_structs(A_s=3.0e-9)
        assert altered_struct.cosmo_tables.ps_norm == altered_struct.cosmo_params.A_s
        assert not altered_struct.cosmo_tables.USE_SIGMA_8
        # Now check that if we switch to sigma8, we use it at the backend!
        altered_struct = altered_struct_CLASS.evolve_input_structs(
            SIGMA_8=1.0, A_s=None
        )
        assert (
            altered_struct.cosmo_tables.ps_norm == altered_struct.cosmo_params.SIGMA_8
        )
        assert altered_struct.cosmo_tables.USE_SIGMA_8
        # Last check, switching agaim from sigma8 to A_s (but this time SIGMA_8 is not None)
        altered_struct = altered_struct_CLASS.evolve_input_structs(
            A_s=3.0e-9, SIGMA_8=None
        )
        assert altered_struct.cosmo_tables.ps_norm == altered_struct.cosmo_params.A_s
        assert not altered_struct.cosmo_tables.USE_SIGMA_8

        # Check that we cannot work with both parameters
        with pytest.raises(ValueError, match="Cannot set both SIGMA_8 and A_s!"):
            self.default_sigma8.evolve_input_structs(A_s=3.0e-9)

        with pytest.raises(ValueError, match="Cannot set both SIGMA_8 and A_s!"):
            self.default_A_s.evolve_input_structs(SIGMA_8=1.0)

        # Check that we can change normalization parameter if we set the other parameter to None
        altered_struct = self.default_sigma8.evolve_input_structs(
            A_s=3.0e-9, SIGMA_8=None
        )
        assert altered_struct.cosmo_params.A_s == 3.0e-9
        # Even though we work with A_s, transfer function is EH, so we still use sigma8 at the backend
        assert (
            altered_struct.cosmo_tables.ps_norm == altered_struct.cosmo_params.SIGMA_8
        )
        assert altered_struct.cosmo_tables.USE_SIGMA_8
        altered_struct = self.default_A_s.evolve_input_structs(SIGMA_8=1.0, A_s=None)
        assert altered_struct.cosmo_params.SIGMA_8 == 1.0
        assert (
            altered_struct.cosmo_tables.ps_norm == altered_struct.cosmo_params.SIGMA_8
        )
        assert altered_struct.cosmo_tables.USE_SIGMA_8

    @pytest.mark.parametrize("template", _ALL_ALIASES)
    def test_from_template(self, template):
        """Test that creation from a template works for all templates."""
        inputs = InputParameters.from_template(template, random_seed=1)
        assert isinstance(inputs, InputParameters)

    def test_bad_input(self):
        """Test that passing a non-existent parameter to evolve raises."""
        with pytest.raises(
            TypeError,
            match="BAD_INPUT is not a valid keyword input",
        ):
            InputParameters(random_seed=0).evolve_input_structs(BAD_INPUT=True)

    def test_halomass_ranges(self):
        """Test that passing a non-existent parameter to evolve raises."""
        with pytest.raises(
            ValueError,
            match="There is a gap/overlap in the halo mass ranges",
        ):
            # These cells are ~6e7 Msun, with 1e8 minimum sampler mass this leaves a gap
            self.default.evolve_input_structs(BOX_LEN=30.0)

        with pytest.warns(
            UserWarning,
            match="The maximum halo mass",
        ):
            # The cell size is ~1e11 Msun
            self.default.evolve_input_structs(
                SOURCE_MODEL="L-INTEGRAL",
                USE_UPPER_STELLAR_TURNOVER=False,
            )

    def test_linear_node_redshifts(self):
        """Test that with_linear_redshifts works as expected."""
        with pytest.raises(ValueError, match=r"Either `nz` or `step` must be provided"):
            InputParameters(random_seed=1).with_linear_redshifts()

    def test_zstep_factor_raises_warning(self):
        """Test that using zstep_factor raises a warning."""
        with pytest.warns(
            DeprecationWarning,
            match=r"The `zstep_factor` argument is deprecated and will be removed in a future version. Please use `step` instead.",
        ):
            InputParameters(random_seed=1).with_logspaced_redshifts(
                zstep_factor=0.5, zmin=5, zmax=15
            )
