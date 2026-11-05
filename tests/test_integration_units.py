"""Check the physical comparisons and HDF5 format used by integration references."""

import gc
import shutil
import weakref
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from astropy import units as u

from . import produce_integration_test_data as prd
from .integration_units import (
    assert_quantity_close,
    assert_reference_fields,
    field_unit,
    power_spectrum,
    read_quantity,
    write_quantity,
)
from .test_integration_features import options_marked

_PERTURB_REFERENCE_FIELDS = {
    "power_dens",
    "power_vel",
    "k_dens",
    "k_vel",
    "pdf_dens",
    "pdf_vel",
    "x_dens",
    "x_vel",
}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("unit", [u.K, u.dimensionless_unscaled, u.Unit(1e-12 / u.s)])
def test_reference_roundtrip_keeps_values_dtype_and_scale(tmp_path, dtype, unit):
    values = np.array([0, 1, 2], dtype=dtype)
    with h5py.File(tmp_path / "reference.h5", "w") as reference:
        write_quantity(reference, "field", u.Quantity(values, unit, copy=False))
    with h5py.File(tmp_path / "reference.h5", "r") as reference:
        restored = read_quantity(reference["field"])
        assert restored.dtype == values.dtype
        assert restored.unit == unit
        assert restored.value.tobytes() == values.tobytes()


def test_reference_without_units_is_rejected(tmp_path):
    with h5py.File(tmp_path / "reference.h5", "w") as reference:
        dataset = reference.create_dataset("field", data=[1.0])
        with pytest.raises(ValueError, match="unit metadata"):
            read_quantity(dataset)


@pytest.mark.parametrize("unit", ["not_a_physical_unit", 1, None])
def test_reference_with_invalid_units_is_rejected(tmp_path, unit):
    with h5py.File(tmp_path / "reference.h5", "w") as reference:
        dataset = reference.create_dataset("field", data=[1.0])
        dataset.attrs["unit"] = unit if unit is not None else np.array([1.0])
        with pytest.raises((ValueError, TypeError), match="unit"):
            read_quantity(dataset)


def test_reference_with_dimensionless_units_is_explicit(tmp_path):
    with h5py.File(tmp_path / "reference.h5", "w") as reference:
        write_quantity(reference, "fraction", u.Quantity([1.0]))
        assert reference["fraction"].attrs["unit"] == ""
        assert read_quantity(reference["fraction"]).unit == u.dimensionless_unscaled


def test_equivalent_physical_values_pass_after_hdf5_read(tmp_path):
    with h5py.File(tmp_path / "reference.h5", "w") as reference:
        write_quantity(reference, "temperature", [1000, 2000] * u.mK)
        assert_quantity_close(
            [1, 2] * u.K, read_quantity(reference["temperature"]), rtol=0
        )


def test_changing_only_reference_unit_fails(tmp_path):
    with h5py.File(tmp_path / "reference.h5", "w") as reference:
        write_quantity(reference, "temperature", [1, 2] * u.K)
        reference["temperature"].attrs.modify("unit", "mK")
        with pytest.raises(AssertionError):
            assert_quantity_close(
                [1, 2] * u.K, read_quantity(reference["temperature"]), rtol=1e-4
            )


def test_incompatible_dimensions_fail():
    with pytest.raises(u.UnitsError):
        assert_quantity_close([1] * u.K, [1] * u.m, rtol=1e-4)


@pytest.mark.parametrize("side", ["actual", "reference"])
def test_unitless_values_cannot_enter_comparison(side):
    arguments = {"actual": [1] * u.K, "reference": [1] * u.K}
    arguments[side] = np.array([1.0])
    with pytest.raises(TypeError, match="Quantity"):
        assert_quantity_close(**arguments, rtol=1e-4)


def test_unequal_shapes_do_not_broadcast():
    with pytest.raises(AssertionError, match="shape"):
        assert_quantity_close([1, 1] * u.K, [1] * u.K, rtol=1e-4)


def test_nonfinite_values_cannot_pass_as_matching():
    with pytest.raises(AssertionError):
        assert_quantity_close([np.nan] * u.K, [np.nan] * u.K, rtol=1e-4)


def test_absolute_tolerance_carries_reference_units():
    assert_quantity_close([1.001] * u.K, [1000] * u.mK, rtol=0, atol=2 * u.mK)
    with pytest.raises(AssertionError):
        assert_quantity_close([1.003] * u.K, [1000] * u.mK, rtol=0, atol=2 * u.mK)


def test_field_unit_is_taken_from_the_output_manager():
    box = SimpleNamespace(arrays={"temperature": SimpleNamespace(unit=u.mK)})
    assert field_unit([box], "temperature") == u.mK


def test_missing_field_metadata_is_not_assumed_dimensionless():
    box = SimpleNamespace(arrays={"temperature": SimpleNamespace(unit=None)})
    with pytest.raises(ValueError, match="temperature"):
        field_unit([box], "temperature")
    with pytest.raises(KeyError, match="unknown"):
        field_unit([box], "unknown")


@pytest.fixture
def wave():
    """Supply a deterministic, nonzero three-dimensional physical field."""
    grid = np.indices((16, 16, 16))
    noise = np.random.default_rng(21).normal(0, 0.1, size=grid.shape[1:])
    return (np.sin(2 * np.pi * grid[0] / 16) + noise) * u.K


def test_power_units_include_field_squared_and_comoving_volume(wave):
    power, k = power_spectrum(wave, 16 * u.Mpc)
    assert power.unit == u.K**2 * u.Mpc**3
    assert k.unit == u.Mpc**-1
    assert np.any(power.value > 0)


def test_rescaling_field_values_and_units_preserves_physical_power(wave):
    kelvin_power, kelvin_k = power_spectrum(wave, 16 * u.Mpc)
    millikelvin_power, millikelvin_k = power_spectrum(wave.to(u.mK), 16000 * u.kpc)
    assert_quantity_close(kelvin_power, millikelvin_power, rtol=1e-12)
    assert_quantity_close(kelvin_k, millikelvin_k, rtol=1e-12)
    np.testing.assert_allclose(
        millikelvin_power.value, kelvin_power.value * 1e6, atol=1e-20, rtol=1e-12
    )


def test_inconsistent_field_rescaling_fails_power_comparison(wave):
    power, _ = power_spectrum(wave, 16 * u.Mpc)
    wrong_power, _ = power_spectrum(wave.value * u.mK, 16 * u.Mpc)
    with pytest.raises(AssertionError):
        assert_quantity_close(power, wrong_power, rtol=1e-4)


def test_doubling_box_size_changes_power_by_volume_factor(wave):
    small_power, small_k = power_spectrum(wave, 16 * u.Mpc)
    large_power, large_k = power_spectrum(wave, 32 * u.Mpc)
    np.testing.assert_allclose(
        large_power.value, 8 * small_power.value, atol=1e-20, rtol=1e-12
    )
    assert_quantity_close(large_k, small_k / 2, rtol=1e-12)


def test_reference_writer_rejects_untyped_values(tmp_path):
    with (
        h5py.File(tmp_path / "reference.h5", "w") as reference,
        pytest.raises(TypeError, match="Quantity"),
    ):
        write_quantity(reference, "temperature", np.array([1.0]))


def test_power_reference_writer_keeps_units(monkeypatch, tmp_path):
    from . import produce_integration_test_data as prd

    k = np.array([0.1, 0.2]) / u.Mpc
    field_units = {
        "spin_temperature": u.K,
        "ionisation_rate_G12": u.Unit(1e-12 / u.s),
        "J_21_LW": u.Unit(1e-21 * u.erg / (u.s * u.cm**2 * u.Hz * u.sr)),
    }
    powers = {
        name: np.array([1, 2], dtype=np.float32) * unit**2 * u.Mpc**3
        for name, unit in field_units.items()
    }
    simulation = SimpleNamespace(HII_DIM=4, DIM=8, BOX_LEN=50)
    coeval = SimpleNamespace(simulation_options=simulation)
    lightcone = SimpleNamespace(
        inputs=SimpleNamespace(matter_options=SimpleNamespace(POWER_SPECTRUM="EH")),
        global_quantities={
            "neutral_fraction": np.array([0.5], dtype=np.float32),
            "brightness_temp": np.array([10], dtype=np.float32),
        },
    )
    monkeypatch.setattr(prd, "DATA_PATH", tmp_path)
    monkeypatch.setattr(
        prd, "produce_coeval_power_spectra", lambda *a, **kw: (k, powers, coeval)
    )
    monkeypatch.setattr(
        prd, "produce_lc_power_spectra", lambda *a, **kw: (k, powers, lightcone)
    )
    monkeypatch.setattr(
        prd,
        "lightcone_field_units",
        lambda lc: {
            "neutral_fraction": u.dimensionless_unscaled,
            "brightness_temp": u.mK,
        },
    )
    path, _, _ = prd.produce_power_spectra_for_tests(
        "units", 10, force=True, direc=tmp_path
    )
    with h5py.File(path, "r") as reference:
        for group in ("coeval", "lightcone"):
            assert_quantity_close(read_quantity(reference[group]["k"]), k, rtol=0)
            for name, power in powers.items():
                restored = read_quantity(reference[group][f"power_{name}"])
                assert restored.dtype == np.float32
                assert_quantity_close(restored, power, rtol=1e-14)
        assert read_quantity(reference["lightcone/global_brightness_temp"]).unit == u.mK
        assert (
            read_quantity(reference["lightcone/global_neutral_fraction"]).unit
            == u.dimensionless_unscaled
        )


def test_perturb_reference_keeps_historical_velocity_scale(monkeypatch, tmp_path):
    from . import produce_integration_test_data as prd

    rng = np.random.default_rng(41)
    density = (
        rng.uniform(-0.5, 1, (16, 16, 16)).astype(np.float32) * u.dimensionless_unscaled
    )
    velocity = (
        rng.uniform(-1e-16, 1e-16, density.shape).astype(np.float32) * u.Mpc / u.s
    )
    simulation = SimpleNamespace(BOX_LEN=50, HII_DIM=16, DIM=32)
    inputs = SimpleNamespace(simulation_options=simulation)
    initial = SimpleNamespace(simulation_options=simulation)
    pt = SimpleNamespace(
        arrays={
            "density": SimpleNamespace(get_quantity=lambda: density),
            "velocity_z": SimpleNamespace(get_quantity=lambda: velocity),
        }
    )
    monkeypatch.setattr(prd, "DATA_PATH", tmp_path)
    monkeypatch.setattr(
        prd,
        "get_all_options_struct",
        lambda *a, **kw: {"inputs": inputs, "out_redshifts": 10},
    )
    monkeypatch.setattr(prd, "compute_initial_conditions", lambda **kw: initial)
    monkeypatch.setattr(prd, "perturb_field", lambda **kw: pt)
    baseline = prd.produce_perturb_field_data(10)
    stored_velocity_unit = u.Unit(1e-16 * u.Mpc / u.s)
    expected_power, _ = power_spectrum(
        u.Quantity(velocity.value * 1e16, stored_velocity_unit), 50 * u.Mpc
    )
    assert_quantity_close(baseline[3], expected_power, rtol=1e-6)
    assert baseline[6].unit == stored_velocity_unit
    assert baseline[7].unit == stored_velocity_unit**-1

    # A consistent representation change in the input must preserve the references.
    pt.arrays["velocity_z"] = SimpleNamespace(
        get_quantity=lambda: velocity.to(u.km / u.s)
    )
    converted = prd.produce_perturb_field_data(10)
    for actual, reference in zip(converted[:-1], baseline[:-1], strict=True):
        assert_quantity_close(actual, reference, rtol=1e-5)

    path = prd.produce_data_for_perturb_field_tests("units", 10, force=True)
    with h5py.File(path, "r") as reference:
        assert (
            read_quantity(reference["power_vel"]).unit
            == stored_velocity_unit**2 * u.Mpc**3
        )
        assert read_quantity(reference["x_vel"]).unit == stored_velocity_unit
        assert read_quantity(reference["pdf_vel"]).unit == stored_velocity_unit**-1


@pytest.fixture(params=["coeval", "lightcone"])
def integration_reference(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> SimpleNamespace:
    """Supply complete synthetic references through the real comparison entry points."""
    from . import test_integration_features as integration

    kind = request.param
    path = tmp_path / "reference.h5"
    k = np.array([0.1, 0.2]) / u.Mpc
    # Explicit fixture inventory: spin-temperature evolution, no minihalos or recombinations.
    fields = {
        "density",
        "velocity_z",
        "spin_temperature",
        "xray_ionised_fraction",
        "kinetic_temp_neutral",
        "ionisation_rate_G12",
        "neutral_fraction",
        "z_reion",
        "brightness_temp",
    }
    if kind == "coeval":
        fields |= {"lowres_density", "lowres_vx", "lowres_vx_2LPT"}
    field_units = dict.fromkeys(fields, u.dimensionless_unscaled)
    field_units.update(
        {
            "spin_temperature": u.K,
            "kinetic_temp_neutral": u.K,
            "brightness_temp": u.mK,
            "velocity_z": u.Mpc / u.s,
            "lowres_vx": u.Mpc,
            "lowres_vx_2LPT": u.Mpc,
            "ionisation_rate_G12": u.Unit(1e-12 / u.s),
        }
    )
    powers = {
        name: np.array([1.0, 2.0]) * field_units[name] ** 2 * u.Mpc**3
        for name in fields
    }
    global_units = {
        "neutral_fraction": u.dimensionless_unscaled,
        "brightness_temp": u.mK,
    }
    globals_ = {
        "neutral_fraction": np.array([0.5]),
        "brightness_temp": np.array([10.0]),
    }
    with h5py.File(path, "w") as reference:
        group = reference.create_group(kind)
        write_quantity(group, "k", k)
        for name, power in powers.items():
            write_quantity(group, f"power_{name}", power)
        if kind == "lightcone":
            for name, value in globals_.items():
                write_quantity(group, f"global_{name}", value * global_units[name])
    actual_powers = dict(powers)
    result_type = type("Result", (), {})
    result = result_type()
    result.brightness_temp = np.array([1.0])
    result.lightcones = {"brightness_temp": np.array([1.0])}
    result.global_quantities = dict(globals_)
    result.inputs = SimpleNamespace(matter_options=SimpleNamespace(POWER_SPECTRUM="EH"))
    calls = []

    def produce(
        *args: object, **kwargs: object
    ) -> tuple[u.Quantity, dict[str, u.Quantity], object]:
        calls.append(kind)
        return k, actual_powers, result

    monkeypatch.setattr(
        prd, "OPTIONS_TESTRUNS", {"units": [10, {"USE_TS_FLUCT": True}]}
    )
    monkeypatch.setattr(prd, "get_filename", lambda *a: path)
    monkeypatch.setattr(
        prd, f"produce_{'lc' if kind == 'lightcone' else kind}_power_spectra", produce
    )
    monkeypatch.setattr(prd, "lightcone_field_units", lambda lc: global_units)
    monkeypatch.setattr(integration, "OutputCache", lambda path: None)
    monkeypatch.setattr(
        integration, "Coeval" if kind == "coeval" else "LightCone", result_type
    )
    comparison_request = SimpleNamespace(node=SimpleNamespace(user_properties=[]))
    arguments = {
        "request": comparison_request,
        "name": "units",
        "module_direc": tmp_path,
        "plt": None,
    }
    if kind == "lightcone":
        arguments["benchmark"] = SimpleNamespace(
            pedantic=lambda fn, **kw: fn(**kw["kwargs"])
        )

    def compare() -> None:
        getattr(integration, f"test_power_spectra_{kind}")(**arguments)

    return SimpleNamespace(
        kind=kind,
        path=path,
        powers=actual_powers,
        result=result,
        request=comparison_request,
        calls=calls,
        compare=compare,
    )


@pytest.mark.parametrize("consistent", [False, True])
def test_integration_entry_point_checks_physical_power(
    integration_reference: SimpleNamespace, consistent: bool
) -> None:
    case = integration_reference
    reference_power = case.powers["spin_temperature"]
    actual_power = reference_power.to(u.mK**2 * u.Mpc**3)
    if not consistent:
        actual_power = u.Quantity(reference_power.value, actual_power.unit)
    case.powers["spin_temperature"] = actual_power
    if consistent:
        case.compare()
        assert not case.request.node.user_properties
    else:
        with pytest.raises(AssertionError, match="physical values differ"):
            case.compare()
        assert any(
            key == "reference_mismatch" for key, _ in case.request.node.user_properties
        )


def test_integration_dimensions_fail_independently_of_numerical_comparison(
    integration_reference: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnostic-only numeric comparison cannot hide incompatible dimensions."""
    from . import test_integration_features as integration

    case = integration_reference
    power = case.powers["spin_temperature"]
    wrong_unit = u.m**2 * u.Mpc**3
    # These composite dimensions can both have the label "unknown"; comparing
    # the physical_type strings would not establish that they are equivalent.
    assert not power.unit.is_equivalent(wrong_unit)
    case.powers["spin_temperature"] = u.Quantity(power.value, wrong_unit)
    monkeypatch.setattr(integration, "assert_quantity_close", lambda *a, **kw: None)
    with pytest.raises(u.UnitConversionError, match="spin_temperature"):
        case.compare()


def test_physical_dimension_check_ignores_global_equivalencies() -> None:
    mass = [1.0] * u.kg
    energy = mass.to(u.J, equivalencies=u.mass_energy())
    with (
        u.set_enabled_equivalencies(u.mass_energy()),
        pytest.raises(u.UnitConversionError),
    ):
        assert_quantity_close(mass, energy, rtol=1e-12)


@pytest.mark.parametrize("omission", ["one_power", "all_powers", "k", "group"])
def test_integration_rejects_missing_reference_fields(
    integration_reference: SimpleNamespace, omission: str
) -> None:
    case = integration_reference
    with h5py.File(case.path, "r+") as reference:
        group = reference[case.kind]
        if omission == "one_power":
            del group["power_spin_temperature"]
        elif omission == "all_powers":
            for key in list(group):
                if key.startswith("power_"):
                    del group[key]
        elif omission == "k":
            del group["k"]
        else:
            del reference[case.kind]
    with pytest.raises(AssertionError, match="Missing reference"):
        case.compare()
    assert not case.calls, "Incomplete references must fail before a simulation starts."


def test_integration_rejects_unexpected_reference_power(
    integration_reference: SimpleNamespace,
) -> None:
    case = integration_reference
    with h5py.File(case.path, "r+") as reference:
        write_quantity(
            reference[case.kind], "power_unexpected", np.array([1.0, 2.0]) * u.Mpc**3
        )
    with pytest.raises(AssertionError, match="Unexpected reference"):
        case.compare()
    assert not case.calls


def test_integration_rejects_missing_simulated_power(
    integration_reference: SimpleNamespace,
) -> None:
    case = integration_reference
    del case.powers["spin_temperature"]
    with pytest.raises(AssertionError, match="Simulated power fields"):
        case.compare()


@pytest.mark.parametrize("integration_reference", ["lightcone"], indirect=True)
@pytest.mark.parametrize("omission", ["one", "all"])
def test_lightcone_rejects_missing_global_reference(
    integration_reference: SimpleNamespace, omission: str
) -> None:
    case = integration_reference
    with h5py.File(case.path, "r+") as reference:
        del reference["lightcone/global_neutral_fraction"]
        if omission == "all":
            del reference["lightcone/global_brightness_temp"]
    with pytest.raises(
        AssertionError, match=r"Missing reference.*global_neutral_fraction"
    ):
        case.compare()
    assert not case.calls


@pytest.mark.parametrize("name", options_marked)
def test_saved_power_references_have_expected_fields_and_units(name: str) -> None:
    redshift, options = prd.OPTIONS_TESTRUNS[name]
    inputs = prd.get_all_options_struct(redshift, lc=True, **options)["inputs"]
    with h5py.File(prd.get_filename("power_spectra", name), "r") as reference:
        for kind in ("coeval", "lightcone"):
            assert kind in reference, (
                f"Missing reference group {kind} in {reference.filename}"
            )
            fields = {"k"} | {
                f"power_{key}" for key in prd.get_power_fields(inputs, kind=kind)
            }
            if kind == "lightcone":
                fields |= {f"global_{key}" for key in prd.GLOBAL_FIELDS}
            assert_reference_fields(reference[kind], fields)
            for key in fields:
                read_quantity(reference[kind][key])


@pytest.mark.parametrize("name", list(prd.OPTIONS_PT))
def test_saved_perturb_references_have_expected_fields_and_units(name: str) -> None:
    with h5py.File(prd.get_filename("perturb_field_data", name), "r") as reference:
        assert_reference_fields(reference, _PERTURB_REFERENCE_FIELDS)
        for key in _PERTURB_REFERENCE_FIELDS:
            read_quantity(reference[key])


@pytest.mark.parametrize(
    "missing",
    [
        "file",
        "coeval",
        "coeval/power_velocity_z",
        "lightcone/global_brightness_temp",
    ],
)
def test_saved_power_reference_rejects_omissions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str
) -> None:
    original = prd.get_filename("power_spectra", "simple")
    path = tmp_path / original.name
    if missing != "file":
        shutil.copy2(original, path)
        with h5py.File(path, "r+") as reference:
            del reference[missing]
    monkeypatch.setattr(prd, "DATA_PATH", tmp_path)
    if missing == "file":
        with pytest.raises(FileNotFoundError):
            test_saved_power_references_have_expected_fields_and_units("simple")
    else:
        with pytest.raises(AssertionError, match="Missing reference"):
            test_saved_power_references_have_expected_fields_and_units("simple")


def test_saved_power_reference_rejects_k_only_groups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(prd, "DATA_PATH", tmp_path)
    with h5py.File(prd.get_filename("power_spectra", "simple"), "w") as reference:
        for kind in ("coeval", "lightcone"):
            write_quantity(
                reference.create_group(kind), "k", np.array([0.1, 0.2]) / u.Mpc
            )
    with pytest.raises(AssertionError, match=r"Missing reference.*power_"):
        test_saved_power_references_have_expected_fields_and_units("simple")


@pytest.mark.parametrize("missing", ["file", "pdf_vel"])
def test_saved_perturb_reference_rejects_omissions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str
) -> None:
    original = prd.get_filename("perturb_field_data", "simple")
    path = tmp_path / original.name
    if missing != "file":
        shutil.copy2(original, path)
        with h5py.File(path, "r+") as reference:
            del reference[missing]
    monkeypatch.setattr(prd, "DATA_PATH", tmp_path)
    if missing == "file":
        with pytest.raises(FileNotFoundError):
            test_saved_perturb_references_have_expected_fields_and_units("simple")
    else:
        with pytest.raises(AssertionError, match=r"Missing reference.*pdf_vel"):
            test_saved_perturb_references_have_expected_fields_and_units("simple")


def test_comparison_plot_converts_to_reference_units(tmp_path):
    from matplotlib.figure import Figure

    from .test_integration_features import make_comparison_plot

    figure = Figure()
    figure_ref = weakref.ref(figure)
    axes = None
    try:
        axes = figure.subplots(2)
        k = np.array([0.1, 0.2, 0.3]) / u.Mpc
        power = np.array([1.0, 2.0, 3.0]) * u.K**2 * u.Mpc**3
        make_comparison_plot(
            k,
            k.to(u.kpc**-1),
            power,
            power.to(u.mK**2 * u.Mpc**3),
            axes,
            xlab="k",
            ylab="Temperature power",
        )
        np.testing.assert_allclose(
            axes[0].lines[0].get_ydata(),
            axes[0].lines[1].get_ydata(),
            rtol=1e-14,
            atol=0,
        )
        np.testing.assert_allclose(axes[1].lines[0].get_ydata(), 0, rtol=0, atol=1e-14)
        figure.savefig(tmp_path / "comparison.png")
    finally:
        figure.clear()
        del axes, figure
        # Release this test's Matplotlib cycles before driver cycle checks run.
        gc.collect()
    assert figure_ref() is None, "Comparison plot kept its Figure alive after cleanup."
