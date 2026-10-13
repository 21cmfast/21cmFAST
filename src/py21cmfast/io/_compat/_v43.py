"""Changes to the file format made in v4.3.0."""

from __future__ import annotations

import numpy as np

from .._compat_framework import (
    AddParameter,
    ConvertOutputStruct,
    ConvertParameters,
    FormatRelease,
    RawInputs,
    RawOutputStruct,
    RenameField,
    RenameOutputStruct,
    RenameParameter,
    UnconvertibleOutputStruct,
    load_array,
    warn,
)

_LAGRANGIAN_SOURCE_MODELS = ("L-INTEGRAL", "DEXM-ESF", "CHMF-SAMPLER")
_DISCRETE_HALO_SOURCE_MODELS = ("DEXM-ESF", "CHMF-SAMPLER")


def _source_model(inputs: RawInputs) -> str:
    return inputs.get("matter_options", "SOURCE_MODEL", "")


def _rhocrit_omb(inputs: RawInputs) -> float:
    """Compute the critical density times Omega_b (Msun/Mpc^3), as in the C code."""
    hubble = inputs.get("cosmo_params", "hlittle") * 3.2407e-18  # 1/s
    grav = 6.6743e-8  # cgs
    cm_per_mpc = 3.08567758e24
    msun = 1.989e33  # g
    rhocrit = 3 * hubble**2 / (8 * np.pi * grav) * cm_per_mpc**3 / msun
    return rhocrit * inputs.get("cosmo_params", "OMb")


def _nion_prefactor(inputs: RawInputs, mcg: bool = False) -> float | None:
    """Compute the prefactor included in the n_ion fields of IonizedBox since v4.3.

    Returns None if the prefactor can't be determined from the inputs alone (which
    is the case for f-photoncons, where the escape fraction is fit as a function of
    redshift).
    """
    if inputs.get("astro_options", "PHOTON_CONS_TYPE") == "f-photoncons":
        return None

    p = inputs.structs["astro_params"]
    if mcg:
        return 10 ** p["F_STAR7_MCG"] * 10 ** p["F_ESC7_MCG"] * p["POP3_ION"]
    if _source_model(inputs) == "CONST-ION-EFF":
        return p["HII_EFF_FACTOR"]
    return 10 ** p["F_STAR10_ACG"] * 10 ** p["F_ESC10_ACG"] * p["POP2_ION"]


def _v_cb_model(inputs: RawInputs) -> None:
    matter = inputs.structs.setdefault("matter_options", {})
    astro_opts = inputs.structs.setdefault("astro_options", {})
    astro_params = inputs.structs.setdefault("astro_params", {})

    use_rel_vel = matter.pop("USE_RELATIVE_VELOCITIES", None)
    fix_vcb = astro_opts.pop("FIX_VCB_AVG", None)
    fixed_vavg = astro_params.pop("FIXED_VAVG", None)

    if "V_CB_MODEL" in matter or use_rel_vel is None:
        return

    if use_rel_vel:
        matter["V_CB_MODEL"] = "AVG-DEBUG" if fix_vcb else "FLUCTS"
    elif fix_vcb and astro_opts.get("USE_MCGS", False):
        # The fixed average velocity was only used by mini-halos.
        matter["V_CB_MODEL"] = "AVG-DEBUG"
    else:
        matter["V_CB_MODEL"] = "NONE"

    if matter["V_CB_MODEL"] == "AVG-DEBUG" and fixed_vavg is not None:
        astro_params.setdefault("V_CB_AVG_DEBUG", fixed_vavg)


def _use_metallicity(inputs: RawInputs) -> bool:
    # Before v4.3, the metallicity-dependence of the X-ray luminosity was switched on
    # by USE_UPPER_STELLAR_TURNOVER (whose default was True), and only for discrete
    # halos.
    upper = bool(inputs.get("astro_options", "USE_UPPER_STELLAR_TURNOVER", True))
    return upper and _source_model(inputs) in _DISCRETE_HALO_SOURCE_MODELS


def _photoheating_feedback(inputs: RawInputs) -> bool:
    return bool(inputs.get("astro_options", "USE_MCGS", False))


def _drop_incomplete_cosmo_tables(inputs: RawInputs) -> None:
    required = {"ps_norm", "USE_SIGMA_8", "V_CB_AVG"}
    if inputs.cosmo_tables is not None and not required.issubset(inputs.cosmo_tables):
        inputs.cosmo_tables = None


def _halobox_n_ion(struct: RawOutputStruct) -> None:
    # Applied to the struct while it is still called HaloBox.
    if "n_ion" in struct.arrays:
        struct.arrays["n_ion"] = load_array(struct.arrays["n_ion"]) / _rhocrit_omb(
            struct.inputs
        )
    # This was -inf (log10(0)) when there were no mini-halos, and is now 0.
    if np.isneginf(struct.primitives.get("log10_Mcrit_MCG_ave", 0.0)):
        struct.primitives["log10_Mcrit_MCG_ave"] = 0.0


def _divide_by_rhocrit_omb(val, struct: RawOutputStruct):
    return val / _rhocrit_omb(struct.inputs)


def _nion_unconditional(mcg: bool):
    def transform(val, struct: RawOutputStruct):
        if _source_model(struct.inputs) in _LAGRANGIAN_SOURCE_MODELS:
            # Overwritten by the mean of the (halo) source field, which is n_ion of
            # the HaloBox (for ACGs, MCGs are included in the same field).
            return val if mcg else val / _rhocrit_omb(struct.inputs)

        prefactor = _nion_prefactor(struct.inputs, mcg=mcg)
        if prefactor is None:
            warn(
                "The mean n_ion in this IonizedBox (written by 21cmFAST "
                f"v{struct.version}) can't be converted with f-photoncons, and is "
                "unavailable."
            )
            return None
        return val * prefactor

    return transform


def _nion_conditional(mcg: bool):
    def transform(val, struct: RawOutputStruct):
        if _source_model(struct.inputs) in _LAGRANGIAN_SOURCE_MODELS:
            # This is not used (or saved) for Lagrangian source models any more.
            return None

        prefactor = _nion_prefactor(struct.inputs, mcg=mcg)
        if prefactor is None:
            warn(
                "The filtered n_ion in this IonizedBox (written by 21cmFAST "
                f"v{struct.version}) can't be converted with f-photoncons, and is "
                "unavailable."
            )
            return None
        return val * prefactor

    return transform


_NION_PREFACTOR_EXPLANATION = (
    "now includes the prefactor f_star * f_esc * N_ion (or HII_EFF_FACTOR for "
    "CONST-ION-EFF), which is applied to older files (not possible for f-photoncons)"
)


RELEASE = FormatRelease(
    "4.3.0",
    [
        # ---- Input parameters ----
        RenameParameter(
            "simulation_options",
            "MIN_XE_FOR_FCOLL_IN_TAUX",
            "MIN_XE_FOR_NION_IN_TAUX",
        ),
        RenameParameter("astro_options", "USE_MINI_HALOS", "USE_MCGS"),
        RenameParameter(
            "astro_options", "INTEGRATION_METHOD_ATOMIC", "INTEGRATION_METHOD_ACGS"
        ),
        RenameParameter(
            "astro_options", "INTEGRATION_METHOD_MINI", "INTEGRATION_METHOD_MCGS"
        ),
        RenameParameter("astro_params", "F_STAR10", "F_STAR10_ACG"),
        RenameParameter("astro_params", "F_STAR7_MINI", "F_STAR7_MCG"),
        RenameParameter("astro_params", "ALPHA_STAR", "ALPHA_STAR_ACG"),
        RenameParameter("astro_params", "ALPHA_STAR_MINI", "ALPHA_STAR_MCG"),
        RenameParameter("astro_params", "F_ESC10", "F_ESC10_ACG"),
        RenameParameter("astro_params", "F_ESC7_MINI", "F_ESC7_MCG"),
        RenameParameter("astro_params", "L_X", "LX_OVER_SFR_ACG"),
        RenameParameter("astro_params", "L_X_MINI", "LX_OVER_SFR_MCG"),
        RenameParameter("astro_params", "M_TURN", "M_TURN_STELLAR_FEEDBACK"),
        ConvertParameters(
            _v_cb_model,
            "matter_options.USE_RELATIVE_VELOCITIES, astro_options.FIX_VCB_AVG "
            "and astro_params.FIXED_VAVG replaced by matter_options.V_CB_MODEL "
            "and astro_params.V_CB_AVG_DEBUG",
        ),
        AddParameter(
            "astro_options",
            "USE_METALLICITY",
            _use_metallicity,
            "previously, the metallicity-dependence of the X-ray luminosity of "
            "discrete halos was switched on by USE_UPPER_STELLAR_TURNOVER, so it "
            "is set to that value for older files with discrete halos (and False "
            "otherwise).",
        ),
        AddParameter(
            "astro_options",
            "USE_REIONIZATION_PHOTOHEATING_FEEDBACK",
            _photoheating_feedback,
            "previously this was switched on by USE_MINI_HALOS, so it is set to "
            "that value for older files.",
        ),
        ConvertParameters(
            _drop_incomplete_cosmo_tables,
            "cosmo_tables gained V_CB_AVG (and ps_norm and USE_SIGMA_8 in v4.1). "
            "Incomplete tables in older files are discarded (they are recomputed "
            "when needed).",
        ),
        # ---- Output structs ----
        ConvertOutputStruct(
            "HaloBox",
            _halobox_n_ion,
            "n_ion is now divided by rho_crit * Omega_b, and log10_Mcrit_MCG_ave "
            "is 0 rather than -inf without mini-halos",
        ),
        RenameOutputStruct("HaloBox", "EmissivityFields"),
        RenameField("EmissivityFields", "count", "halo_number"),
        RenameField("EmissivityFields", "halo_mass", "halo_mass_density"),
        RenameField("EmissivityFields", "halo_stars", "stellar_mass_density_acg"),
        RenameField("EmissivityFields", "halo_stars_mini", "stellar_mass_density_mcg"),
        RenameField("EmissivityFields", "halo_sfr", "sfrd_acg"),
        RenameField("EmissivityFields", "halo_sfr_mini", "sfrd_mcg"),
        RenameField("EmissivityFields", "halo_xray", "xray_emissivity"),
        RenameField("EmissivityFields", "whalo_sfr", "fesc_weighted_sfrd"),
        RenameField("EmissivityFields", "log10_Mcrit_ACG_ave", "log10_mturn_acg_ave"),
        RenameField("EmissivityFields", "log10_Mcrit_MCG_ave", "log10_mturn_mcg_ave"),
        RenameField("PerturbedHaloCatalog", "sfr", "sfr_acg"),
        RenameField("PerturbedHaloCatalog", "stellar_masses", "stellar_masses_acg"),
        RenameField(
            "PerturbedHaloCatalog",
            "ion_emissivity",
            "n_ion",
            _divide_by_rhocrit_omb,
            "now divided by rho_crit * Omega_b",
        ),
        RenameField("PerturbedHaloCatalog", "xray_emissivity", "xray_luminosity"),
        RenameField("PerturbedHaloCatalog", "fesc_sfr", "fesc_weighted_sfr"),
        RenameField("PerturbedHaloCatalog", "stellar_mini", "stellar_masses_mcg"),
        RenameField("PerturbedHaloCatalog", "sfr_mini", "sfr_mcg"),
        RenameField(
            "IonizedBox",
            "mean_f_coll",
            "nion_unconditional_acg",
            _nion_unconditional(mcg=False),
            _NION_PREFACTOR_EXPLANATION,
        ),
        RenameField(
            "IonizedBox",
            "mean_f_coll_MINI",
            "nion_unconditional_mcg",
            _nion_unconditional(mcg=True),
            _NION_PREFACTOR_EXPLANATION,
        ),
        RenameField("IonizedBox", "log10_Mturnover_ave", "log10_mturn_ave_acg"),
        RenameField("IonizedBox", "log10_Mturnover_MINI_ave", "log10_mturn_ave_mcg"),
        RenameField(
            "IonizedBox",
            "unnormalised_nion",
            "nion_conditional_filtered_acg",
            _nion_conditional(mcg=False),
            _NION_PREFACTOR_EXPLANATION,
        ),
        RenameField(
            "IonizedBox",
            "unnormalised_nion_mini",
            "nion_conditional_filtered_mcg",
            _nion_conditional(mcg=True),
            _NION_PREFACTOR_EXPLANATION,
        ),
        UnconvertibleOutputStruct(
            "XraySourceBox",
            "it was renamed to RadiationFields, but the calculation of the "
            "radiation fields was then refactored. RadiationFields now holds "
            "the computed X-ray, Lyman-alpha and Lyman-Werner rates and fluxes, "
            "which older files don't contain, while the filtered source fields "
            "that the XraySourceBox held (as 4D arrays over all shells) are now "
            "per-shell scratch arrays of RadiationFieldsSetup. Since these are "
            "intermediate products, they are simply recomputed when needed.",
            successor="RadiationFields",
        ),
    ],
)
