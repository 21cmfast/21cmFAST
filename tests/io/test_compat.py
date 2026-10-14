"""Tests of reading files written by older versions of 21cmFAST."""

import itertools
import warnings
from pathlib import Path

import attrs
import h5py
import numpy as np
import pytest
from packaging.version import Version

import py21cmfast as p21c
from py21cmfast.io import _compat, h5, migrate
from py21cmfast.io import _compat_framework as framework
from py21cmfast.wrapper import outputs as ostruct
from py21cmfast.wrapper._utils import snake_to_camel
from py21cmfast.wrapper.inputs import InputStruct

# The (tiny, unusual) inputs of the legacy test files trigger these warnings.
pytestmark = [
    pytest.mark.filterwarnings("ignore:.*R_BUBBLE_MAX.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*maximum halo mass.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*V_CB_MODEL.*:UserWarning"),
]

RHOCRIT_OMB_FACTOR = 2.775e11  # rho_crit / h^2 in Msun/Mpc^3, approximately


def _raw_inputs(version="4.2", **structs) -> framework.RawInputs:
    return framework.RawInputs(structs=structs, random_seed=1, version=version)


class TestVersions:
    """Tests of parsing and comparing versions."""

    def test_no_version(self):
        """Files without a version (pre-v4) are not supported."""
        with pytest.raises(
            framework.UnsupportedVersionError, match=r"older than v4\.0"
        ):
            framework.parse_version(None)

    def test_too_old(self):
        """Versions before v4.0 are not supported."""
        with pytest.raises(framework.UnsupportedVersionError, match="not supported"):
            framework.check_support("3.4.0")

    def test_bad_version(self):
        """Unparseable versions raise a clear error."""
        with pytest.raises(framework.UnsupportedVersionError, match="understand"):
            framework.parse_version("not-a-version")

    @pytest.mark.parametrize(
        ("version", "n_releases"),
        [("4.0.0", 3), ("4.1.1", 2), ("4.2", 1), ("4.3.dev12+g12345", 1), ("4.3.0", 0)],
    )
    def test_releases_since(self, version, n_releases):
        """The right releases apply to files of each version."""
        assert len(_compat.releases_since(version)) == n_releases
        assert _compat.needs_upgrade(version) == bool(n_releases)

    def test_current_version_is_not_legacy(self):
        """Files of the current version are not converted."""
        assert not _compat.is_legacy(p21c.__version__)
        assert _compat.is_legacy("4.2")


class TestSupportPolicy:
    """Tests of the deprecation policy for files of older major versions."""

    @pytest.mark.parametrize(
        ("version", "current", "status"),
        [
            ("3.4.0", "4.3.0", "unsupported"),
            ("4.0.0b1.dev5", "4.3.0", "supported"),
            ("4.2", "4.3.0", "supported"),
            ("4.2", "5.0.0", "deprecated"),
            ("4.2", "5.3.dev12", "deprecated"),
            ("5.1.0", "5.3.0", "supported"),
            ("4.2", "6.0.0", "unsupported"),
            ("5.1.0", "6.0.0", "deprecated"),
            ("3.4.0", "5.0.0", "unsupported"),
        ],
    )
    def test_support_status(self, version, current, status):
        """Files of the previous major version are deprecated, older ones unsupported."""
        assert framework.support_status(version, current) == status

    def test_deprecated_file_warns(self, legacy_v42, tmp_path: Path, monkeypatch):
        """Reading a file of the previous major version warns, but still works."""
        path = legacy_v42.write_struct(tmp_path / "ics.h5", "halo:InitialConditions")
        monkeypatch.setattr(framework, "CURRENT_VERSION", Version("5.0.0"))
        with pytest.warns(framework.FormatDeprecationWarning, match="v6"):
            obj = h5.read_output_struct(path)
        assert isinstance(obj, p21c.InitialConditions)

    def test_unsupported_file_raises(self, legacy_v42, tmp_path: Path, monkeypatch):
        """Reading a file of two major versions ago raises an error."""
        path = legacy_v42.write_struct(tmp_path / "ics.h5", "halo:InitialConditions")
        monkeypatch.setattr(framework, "CURRENT_VERSION", Version("6.0.0"))
        with pytest.raises(
            framework.UnsupportedVersionError, match=r"removed in v6\.0"
        ):
            h5.read_output_struct(path)

    @pytest.mark.parametrize(
        ("current", "oldest"),
        [
            ("4.3.0", "4.0.0.dev0"),
            ("4.3.dev12", "4.0.0.dev0"),
            ("5.0.0", "4.0.0.dev0"),
            ("6.0.0", "5.0.0.dev0"),
            ("6.2.1", "5.0.0.dev0"),
        ],
    )
    def test_oldest_supported_version(self, current, oldest):
        """The oldest supported version is the start of the previous major version."""
        assert framework.oldest_supported_version(current) == Version(oldest)

    @pytest.mark.parametrize("version", ["4.1.1", "5.2.3", "5.0.0rc1"])
    def test_patch_and_pre_releases_follow_their_major(self, version):
        """Patch and pre-releases are supported/deprecated like their major version."""
        assert framework.support_status(version, "5.3.0") == framework.support_status(
            f"{Version(version).major}.0.0", "5.3.0"
        )

    def test_no_steps_from_unsupported_versions(self):
        """Fails when support for files of an old major version must be removed.

        When this fails, delete the modules of ``py21cmfast.io._compat`` that convert
        from the version given in the message (and their tests and legacy-format
        descriptions in ``tests/test_data/legacy_formats``), as described in the
        developer notes.
        """
        for release in _compat.FORMAT_HISTORY:
            assert framework.support_status(release.from_version) != "unsupported", (
                f"Files written by v{release.from_version.major} are no longer "
                f"supported: remove {release.module_name} from py21cmfast.io._compat."
            )


class TestFormatHistory:
    """Consistency checks of the format history itself."""

    def test_history_is_a_chain(self):
        """Each step converts from the version the previous step converted to."""
        history = _compat.FORMAT_HISTORY
        assert history[0].from_version >= framework.FIRST_FORMAT_VERSION
        for previous, release in itertools.pairwise(history):
            assert release.from_version == previous.to_version

    @pytest.mark.parametrize(
        ("version", "steps"),
        [
            ("4.2.0", ["4.3.0", "4.3.1", "4.4.0"]),
            ("4.2.5", ["4.3.0", "4.3.1", "4.4.0"]),
            ("4.3.0", ["4.3.1", "4.4.0"]),
            ("4.3.1.dev3", ["4.3.1", "4.4.0"]),
            ("4.3.1", ["4.4.0"]),
            ("4.3.2", ["4.4.0"]),
            ("4.4.0", []),
        ],
    )
    def test_patch_releases(self, monkeypatch, version, steps):
        """Patch releases can be steps in the history, and their files are converted.

        Files of patch releases that didn't change the format (e.g. 4.2.5, 4.3.2) get
        all steps converting to later versions.
        """
        applied = []

        def step(frm, to):
            return framework.FormatRelease(
                frm,
                to,
                [framework.ConvertParameters(lambda _: applied.append(to), to)],
            )

        history = (
            step("4.2.0", "4.3.0"),
            step("4.3.0", "4.3.1"),
            step("4.3.1", "4.4.0"),
        )
        assert [r.module_name for r in history] == [
            "_v4_2_v4_3",
            "_v4_3_v4_3_1",
            "_v4_3_1_v4_4",
        ]
        monkeypatch.setattr(_compat, "FORMAT_HISTORY", history)
        _compat.upgrade_inputs(framework.RawInputs(structs={}, version=version))
        assert applied == steps

    def test_every_release_module_is_in_history(self):
        """Each module of _compat is a step in the history, named after its versions."""
        modules = {p.stem for p in Path(_compat.__file__).parent.glob("_v*.py")}
        assert modules == {rel.module_name for rel in _compat.FORMAT_HISTORY}
        for mod in modules:
            assert getattr(_compat, mod).RELEASE in _compat.FORMAT_HISTORY

    def test_module_name(self):
        """Module names include the patch version only when it's not zero."""
        release = framework.FormatRelease("4.2.0", "4.3.1", [])
        assert release.module_name == "_v4_2_v4_3_1"

    def test_versions_must_increase(self):
        """A step can't convert to an earlier version."""
        with pytest.raises(ValueError, match="must be after"):
            framework.FormatRelease("4.3.0", "4.2.0", [])

    def test_all_changes_are_described(self):
        """Every change has a description for the changelog."""
        for release in _compat.FORMAT_HISTORY:
            for change in release.changes:
                assert change.description

    def test_describe(self):
        """The changelog lists the right releases."""
        desc = migrate.describe_format_changes()
        assert "v4.0.0 -> v4.1.0:" in desc
        assert "HaloBox -> EmissivityFields" in desc

        desc = migrate.describe_format_changes(since="4.2")
        assert "v4.0.0 -> v4.1.0:" not in desc
        assert "v4.2.0 -> v4.3.0:" in desc

    def test_parameter_renames_are_current(self):
        """Renamed parameters must exist in this version (unless renamed again)."""
        renames = [
            change
            for release in _compat.FORMAT_HISTORY
            for change in release.changes
            if isinstance(change, framework.RenameParameter)
        ]
        for i, change in enumerate(renames):
            if any(later.old == change.new for later in renames[i + 1 :]):
                continue
            kls = InputStruct._subclasses[
                snake_to_camel(change.new_struct or change.struct)
            ]
            assert change.new in {f.alias for f in attrs.fields(kls)}

    def test_field_renames_are_current(self):
        """Renamed fields of output structs must exist in this version."""
        for release in _compat.FORMAT_HISTORY:
            for change in release.changes:
                if isinstance(change, framework.RenameField):
                    kls = getattr(ostruct, change.struct)
                    names = set(kls._array_field_names) | {
                        f.alias for f in attrs.fields(kls)
                    }
                    assert change.new in names, change.description
                elif isinstance(change, framework.RenameOutputStruct):
                    assert change.new in ostruct._ALL_OUTPUT_STRUCTS
                elif (
                    isinstance(change, framework.UnconvertibleOutputStruct)
                    and change.successor is not None
                ):
                    assert change.successor in ostruct._ALL_OUTPUT_STRUCTS

    def test_struct_names(self):
        """Old and new names of output structs are mapped correctly."""
        assert _compat.historical_struct_names("EmissivityFields") == {
            "EmissivityFields",
            "HaloBox",
        }
        assert _compat.current_struct_name("HaloBox") == "EmissivityFields"
        assert _compat.current_struct_name("XraySourceBox") == "RadiationFields"
        assert "XraySourceBox" in _compat.historical_struct_names("RadiationFields")
        assert _compat.current_struct_name("TsBox") == "TsBox"

    def test_field_names(self):
        """Old and new names of fields are mapped correctly."""
        assert "halo_sfr" in _compat.historical_field_names(
            "EmissivityFields", "sfrd_acg"
        )
        assert _compat.current_field_name("halo_sfr") == "sfrd_acg"
        assert _compat.current_field_name("sfr", ["PerturbedHaloCatalog"]) == "sfr_acg"
        assert _compat.current_field_name("brightness_temp") == "brightness_temp"


class TestInputUpgrades:
    """Tests of the upgrades of input parameters."""

    def test_rename_is_tolerant(self):
        """Renames only apply to data in the old format."""
        change = framework.RenameParameter("astro_params", "F_STAR10", "F_STAR10_ACG")

        inputs = _raw_inputs(astro_params={"F_STAR10": -1.0})
        change.upgrade_inputs(inputs)
        assert inputs.structs["astro_params"] == {"F_STAR10_ACG": -1.0}

        # Nothing to do (already the new format)
        change.upgrade_inputs(inputs)
        assert inputs.structs["astro_params"] == {"F_STAR10_ACG": -1.0}

        # The new name takes precedence if both are there.
        inputs = _raw_inputs(astro_params={"F_STAR10": -1.0, "F_STAR10_ACG": -2.0})
        change.upgrade_inputs(inputs)
        assert inputs.structs["astro_params"] == {"F_STAR10_ACG": -2.0}

    def test_issue_793_min_xe(self):
        """MIN_XE_FOR_FCOLL_IN_TAUX is renamed (issue #793)."""
        inputs = _compat.upgrade_inputs(
            _raw_inputs(simulation_options={"MIN_XE_FOR_FCOLL_IN_TAUX": 0.01})
        )
        assert inputs.structs["simulation_options"] == {"MIN_XE_FOR_NION_IN_TAUX": 0.01}

    @pytest.mark.parametrize(("hii_dim", "expected"), [(1, 0.0), (10, None)])
    def test_v40_min_xe(self, hii_dim, expected):
        """v4.0 single-cell runs get a zero x_e threshold."""
        inputs = _compat.upgrade_inputs(
            _raw_inputs("4.0.0", simulation_options={"HII_DIM": hii_dim})
        )
        assert (
            inputs.structs["simulation_options"].get("MIN_XE_FOR_NION_IN_TAUX")
            == expected
        )

    @pytest.mark.parametrize(
        ("inhomo", "model"), [(True, "inhomogeneous"), (False, "none")]
    )
    def test_inhomo_reco(self, inhomo, model):
        """INHOMO_RECO is converted to RECOMB_MODEL."""
        inputs = _compat.upgrade_inputs(
            _raw_inputs("4.1.1", astro_options={"INHOMO_RECO": inhomo})
        )
        assert inputs.structs["astro_options"]["RECOMB_MODEL"] == model
        assert "INHOMO_RECO" not in inputs.structs["astro_options"]

    @pytest.mark.parametrize(
        ("version", "converted"), [("4.1.1", True), ("4.2", False)]
    )
    def test_sigma_sfr_index(self, version, converted):
        """SIGMA_SFR_INDEX is converted from base e to dex before v4.2."""
        inputs = _compat.upgrade_inputs(
            _raw_inputs(version, astro_params={"SIGMA_SFR_INDEX": -0.12})
        )
        expected = -0.12 / np.log(10) if converted else -0.12
        assert np.isclose(
            inputs.structs["astro_params"]["SIGMA_SFR_INDEX"], expected, rtol=1e-12
        )

    @pytest.mark.parametrize(
        ("rel_vel", "fix_vcb", "mini", "model"),
        [
            (False, False, False, "NONE"),
            (True, False, True, "FLUCTS"),
            (True, True, True, "AVG-DEBUG"),
            (False, True, False, "NONE"),
            (False, True, True, "AVG-DEBUG"),
        ],
    )
    def test_v_cb_model(self, rel_vel, fix_vcb, mini, model):
        """Relative-velocity options are converted to V_CB_MODEL."""
        inputs = _compat.upgrade_inputs(
            _raw_inputs(
                matter_options={"USE_RELATIVE_VELOCITIES": rel_vel},
                astro_options={"FIX_VCB_AVG": fix_vcb, "USE_MINI_HALOS": mini},
                astro_params={"FIXED_VAVG": 25.86},
            )
        )
        assert inputs.structs["matter_options"] == {"V_CB_MODEL": model}
        assert "FIX_VCB_AVG" not in inputs.structs["astro_options"]
        if model == "AVG-DEBUG":
            assert inputs.structs["astro_params"] == {"V_CB_AVG_DEBUG": 25.86}
        else:
            assert inputs.structs["astro_params"] == {}

    @pytest.mark.parametrize(
        ("source_model", "upper", "expected"),
        [
            ("CHMF-SAMPLER", True, True),
            ("CHMF-SAMPLER", False, False),
            ("E-INTEGRAL", True, False),
            ("CONST-ION-EFF", True, False),
        ],
    )
    def test_use_metallicity(self, source_model, upper, expected):
        """USE_METALLICITY reproduces the old behaviour."""
        inputs = _compat.upgrade_inputs(
            _raw_inputs(
                matter_options={"SOURCE_MODEL": source_model},
                astro_options={"USE_UPPER_STELLAR_TURNOVER": upper},
            )
        )
        assert inputs.structs["astro_options"]["USE_METALLICITY"] is expected

    @pytest.mark.parametrize("mini", [True, False])
    def test_photoheating_feedback(self, mini):
        """Photoheating feedback follows the old mini-halo flag."""
        inputs = _compat.upgrade_inputs(
            _raw_inputs(astro_options={"USE_MINI_HALOS": mini})
        )
        opts = inputs.structs["astro_options"]
        assert opts["USE_MCGS"] is mini
        assert opts["USE_REIONIZATION_PHOTOHEATING_FEEDBACK"] is mini

    def test_new_params_not_overridden(self):
        """Parameters already in the file are not changed."""
        inputs = _compat.upgrade_inputs(
            _raw_inputs(
                "4.3.dev1",
                astro_options={"USE_MCGS": False, "USE_METALLICITY": True},
            )
        )
        assert inputs.structs["astro_options"]["USE_METALLICITY"] is True

    @pytest.mark.parametrize(
        ("tables", "kept"),
        [
            ({"ps_norm": 0.8, "USE_SIGMA_8": True}, False),
            ({"ps_norm": 0.8, "USE_SIGMA_8": True, "V_CB_AVG": 27.0}, True),
        ],
    )
    def test_cosmo_tables(self, tables, kept):
        """Incomplete cosmo tables are discarded."""
        inputs = _raw_inputs()
        inputs.cosmo_tables = tables
        _compat.upgrade_inputs(inputs)
        assert (inputs.cosmo_tables is not None) == kept


class TestOutputStructUpgrades:
    """Tests of the upgrades of output structs, on in-memory raw structs."""

    @staticmethod
    def _inputs(source_model="E-INTEGRAL", photoncons="no-photoncons"):
        return _compat.upgrade_inputs(
            _raw_inputs(
                cosmo_params={"hlittle": 0.7, "OMb": 0.05},
                matter_options={"SOURCE_MODEL": source_model},
                astro_options={"PHOTON_CONS_TYPE": photoncons},
                astro_params={
                    "F_STAR10": -1.0,
                    "F_ESC10": -1.0,
                    "POP2_ION": 5000.0,
                    "F_STAR7_MINI": -2.0,
                    "F_ESC7_MINI": -1.0,
                    "POP3_ION": 44000.0,
                    "HII_EFF_FACTOR": 30.0,
                },
                simulation_options={"HII_DIM": 10},
            )
        )

    def test_halobox(self):
        """HaloBox is converted to EmissivityFields."""
        raw = framework.RawOutputStruct(
            kind="HaloBox",
            inputs=self._inputs("CHMF-SAMPLER"),
            arrays={"n_ion": np.ones(3), "halo_sfr": np.full(3, 2.0)},
            primitives={"log10_Mcrit_ACG_ave": 8.0, "log10_Mcrit_MCG_ave": -np.inf},
            version="4.2",
        )
        _compat.upgrade_output_struct(raw)

        assert raw.kind == "EmissivityFields"
        assert np.allclose(raw.arrays["sfrd_acg"], 2.0)
        rhocrit_omb = RHOCRIT_OMB_FACTOR * 0.7**2 * 0.05
        assert np.allclose(raw.arrays["n_ion"], 1 / rhocrit_omb, rtol=1e-3)
        assert raw.primitives == {
            "log10_mturn_acg_ave": 8.0,
            "log10_mturn_mcg_ave": 0.0,
        }

    @pytest.mark.parametrize(
        ("source_model", "acg_factor", "mcg_factor"),
        [
            ("E-INTEGRAL", 0.01 * 5000, 0.001 * 44000),
            ("CONST-ION-EFF", 30.0, 0.001 * 44000),
        ],
    )
    def test_ionized_box_eulerian(self, source_model, acg_factor, mcg_factor):
        """The n_ion fields of IonizedBox gain their prefactor."""
        raw = framework.RawOutputStruct(
            kind="IonizedBox",
            inputs=self._inputs(source_model),
            arrays={
                "unnormalised_nion": np.ones(3),
                "unnormalised_nion_mini": np.ones(3),
            },
            primitives={"mean_f_coll": 1.0, "mean_f_coll_MINI": 1.0},
            version="4.2",
        )
        _compat.upgrade_output_struct(raw)
        assert np.isclose(raw.primitives["nion_unconditional_acg"], acg_factor)
        assert np.isclose(raw.primitives["nion_unconditional_mcg"], mcg_factor)
        assert np.allclose(raw.arrays["nion_conditional_filtered_acg"], acg_factor)
        assert np.allclose(raw.arrays["nion_conditional_filtered_mcg"], mcg_factor)

    def test_ionized_box_lagrangian(self):
        """Lagrangian IonizedBox mean n_ion is normalised."""
        raw = framework.RawOutputStruct(
            kind="IonizedBox",
            inputs=self._inputs("CHMF-SAMPLER"),
            arrays={"unnormalised_nion": np.ones((1, 3))},
            primitives={"mean_f_coll": 1.0, "mean_f_coll_MINI": 0.0},
            version="4.0.0",
        )
        _compat.upgrade_output_struct(raw)
        rhocrit_omb = RHOCRIT_OMB_FACTOR * 0.7**2 * 0.05
        assert np.isclose(
            raw.primitives["nion_unconditional_acg"], 1 / rhocrit_omb, rtol=1e-3
        )
        assert raw.primitives["nion_unconditional_mcg"] == 0.0
        # Not used any more for Lagrangian source models.
        assert raw.arrays == {}

    def test_ionized_box_f_photoncons(self):
        """n_ion fields can't be converted with f-photoncons."""
        raw = framework.RawOutputStruct(
            kind="IonizedBox",
            inputs=self._inputs(photoncons="f-photoncons"),
            arrays={"unnormalised_nion": np.ones(3)},
            primitives={"mean_f_coll": 1.0},
            version="4.2",
        )
        with pytest.warns(framework.CompatibilityWarning, match="f-photoncons"):
            _compat.upgrade_output_struct(raw)
        assert raw.arrays == {}
        assert "nion_unconditional_acg" not in raw.primitives

    def test_perturbed_halo_catalog(self):
        """PerturbedHaloCatalog fields are renamed and converted."""
        raw = framework.RawOutputStruct(
            kind="PerturbedHaloCatalog",
            inputs=self._inputs("CHMF-SAMPLER"),
            arrays={"ion_emissivity": np.ones(3), "sfr": np.full(3, 2.0)},
            version="4.2",
        )
        _compat.upgrade_output_struct(raw)
        assert set(raw.arrays) == {"n_ion", "sfr_acg"}
        rhocrit_omb = RHOCRIT_OMB_FACTOR * 0.7**2 * 0.05
        np.testing.assert_allclose(raw.arrays["n_ion"], 1 / rhocrit_omb, rtol=1e-3)
        np.testing.assert_allclose(raw.arrays["sfr_acg"], 2.0, rtol=0)

    def test_xray_source_box(self):
        """XraySourceBox can't be converted."""
        raw = framework.RawOutputStruct(
            kind="XraySourceBox", inputs=self._inputs(), version="4.2"
        )
        with pytest.raises(
            framework.UnconvertibleError, match="renamed to RadiationFields"
        ):
            _compat.upgrade_output_struct(raw)

    @pytest.mark.parametrize("hii_dim", [1, 10])
    def test_tsbox_q_hi(self, hii_dim):
        """Missing Q_HI only warns for single-cell runs."""
        inputs = self._inputs()
        inputs.structs["simulation_options"]["HII_DIM"] = hii_dim
        raw = framework.RawOutputStruct(kind="TsBox", inputs=inputs, version="4.0.0")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _compat.upgrade_output_struct(raw)
        assert bool(caught) == (hii_dim == 1)
        assert "Q_HI" not in raw.primitives

    def test_box_quantities(self):
        """Lightcone/global quantities are renamed like box fields."""
        inputs = self._inputs("CHMF-SAMPLER")
        inputs.version = framework.parse_version("4.2")
        quantities = _compat.upgrade_box_quantities(
            {
                "halo_sfr": np.ones(2),
                "halo_xray": np.ones(2),
                "n_ion": np.ones(2),
                "brightness_temp": np.ones(2),
            },
            inputs,
        )
        # In particular, halo_xray -> xray_emissivity must not then be renamed as
        # if it were the xray_emissivity of a PerturbedHaloCatalog.
        assert set(quantities) == {
            "sfrd_acg",
            "xray_emissivity",
            "n_ion",
            "brightness_temp",
        }
        assert np.all(quantities["n_ion"] < 1e-8)


class TestReadLegacyFiles:
    """Tests of reading files with the layout of each older version."""

    def test_read_cache_files(self, legacy, tmp_path: Path):
        """All cache files of each old version can be read."""
        for key in legacy.structs:
            path = legacy.write_struct(tmp_path / f"{key.replace(':', '_')}.h5", key)
            kind = key.split(":")[1]

            if kind == "XraySourceBox":
                with pytest.raises(framework.UnconvertibleError):
                    h5.read_output_struct(path)
                continue

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                obj = h5.read_output_struct(path)

            # All changes are known, so nothing is "unknown".
            assert not [w for w in caught if "unknown" in str(w.message)], key
            assert type(obj).__name__ == _compat.current_struct_name(kind)
            assert all(a.state.is_computed for a in obj.arrays.values()), key

    def test_pure_renames_preserve_data(self, legacy, tmp_path: Path):
        """Renamed arrays keep their data."""
        key = "halo:HaloBox"
        path = legacy.write_struct(tmp_path / "halobox.h5", key)
        data = legacy.array_data(key)

        obj = h5.read_output_struct(path)
        assert isinstance(obj, p21c.EmissivityFields)
        np.testing.assert_allclose(obj.get("sfrd_acg"), data["halo_sfr"], rtol=0)
        np.testing.assert_allclose(
            obj.get("xray_emissivity"), data["halo_xray"], rtol=0
        )

    def test_read_with_current_struct_name(self, legacy, tmp_path: Path):
        """Old structs can be read by their new name."""
        path = legacy.write_struct(tmp_path / "halobox.h5", "halo:HaloBox")
        obj = h5.read_output_struct(path, struct="EmissivityFields")
        assert isinstance(obj, p21c.EmissivityFields)

    def test_read_inputs(self, legacy, tmp_path: Path):
        """Old inputs are read without deprecation warnings."""
        path = legacy.write_struct(tmp_path / "ics.h5", "halo:InitialConditions")
        with warnings.catch_warnings():
            # Using deprecated parameter names is not the user's fault here.
            warnings.simplefilter("error", DeprecationWarning)
            inputs = h5.read_inputs(path)
        assert inputs.matter_options.SOURCE_MODEL == "CHMF-SAMPLER"
        assert inputs.astro_options.RECOMB_MODEL == "inhomogeneous"

    def test_issue_793_safe_read(self, legacy_v42, tmp_path: Path):
        """Files from v4.2 can be read in safe mode (they have MIN_XE_FOR_FCOLL_IN_TAUX)."""
        path = legacy_v42.write_struct(tmp_path / "pf.h5", "halo:PerturbedField")
        with h5py.File(path, "r") as fl:
            grp = fl["PerturbedField/InputParameters/simulation_options"]
            assert "MIN_XE_FOR_FCOLL_IN_TAUX" in grp.attrs

        obj = h5.read_output_struct(path, safe=True)
        assert obj.simulation_options.MIN_XE_FOR_NION_IN_TAUX == 1e-3

    def test_missing_array_in_legacy_file(self, legacy_v42, tmp_path: Path):
        """Missing arrays in old files only warn."""
        path = legacy_v42.write_struct(tmp_path / "ts.h5", "halo:TsBox")
        with h5py.File(path, "a") as fl:
            del fl["TsBox/OutputFields/spin_temperature"]

        with pytest.warns(framework.CompatibilityWarning, match="spin_temperature"):
            obj = h5.read_output_struct(path)
        assert not obj.arrays["spin_temperature"].state.is_computed

    def test_unknown_array_in_legacy_file(self, legacy_v42, tmp_path: Path):
        """Unknown arrays in old files warn."""
        path = legacy_v42.write_struct(tmp_path / "ts.h5", "halo:TsBox")
        with h5py.File(path, "a") as fl:
            fl["TsBox/OutputFields/not_a_field"] = np.zeros(3)

        with pytest.warns(framework.CompatibilityWarning, match="not_a_field"):
            h5.read_output_struct(path)

    def test_coeval(self, legacy, tmp_path: Path):
        """Old coeval files can be read."""
        path = legacy.write_coeval(tmp_path / "coeval.h5", "halo")
        coeval = p21c.Coeval.from_file(path)
        assert isinstance(coeval.emissivity_fields, p21c.EmissivityFields)

        # Also via the generic loader.
        assert isinstance(h5.load_high_level_simulation(path), p21c.Coeval)

    def test_global_evolution(self, legacy, tmp_path: Path):
        """Old global-evolution files can be read, with renamed quantities."""
        path = legacy.write_global_evolution(tmp_path / "global.h5", "halo")
        with warnings.catch_warnings():
            # Old parameter names in the file are not the user's fault.
            warnings.simplefilter("error", DeprecationWarning)
            glob = p21c.GlobalEvolution.from_file(path)
        assert "brightness_temp" in glob.quantities
        assert "sfrd_acg" in glob.quantities
        assert "halo_sfr" not in glob.quantities
        assert isinstance(h5.load_high_level_simulation(path), p21c.GlobalEvolution)

    def test_lightcone(self, legacy, tmp_path: Path):
        """Old lightcone files can be read, with renamed quantities."""
        path = legacy.write_lightcone(tmp_path / "lc.h5", "halo")
        lc = p21c.LightCone.from_file(path)
        assert "brightness_temp" in lc.lightcones
        assert "sfrd_acg" in lc.global_quantities
        assert "halo_sfr" not in lc.global_quantities
        assert "unnormalised_nion" not in lc.global_quantities
