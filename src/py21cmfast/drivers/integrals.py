"""provides access to backend integrals and computations which do not require the OutputStruct framework.

These are mostly used for testing, but may be useful as a way to get certain quantities without having to run the full simulation.
"""

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d

from .._cfg import config
from ..c_21cmfast import ffi, lib
from ..wrapper import cfuncs as cf
from ..wrapper.inputs import (
    AstroParams,
    CosmoParams,
    FlagOptions,
    InputParameters,
    UserParams,
)


def get_condition_parameters(
    inputs: InputParameters,
    redshift: float,
    cond_param: float,
    cond_array: Sequence[float],
    from_catalog: bool,
):
    """Get the mass, delta and sigma values for the conditional mass function.

    Depending on the value of `from_catalog`, we assume either a fixed cell size with
    varying delta, or halos of varying mass with a fixed timestep (delta).
    """
    growth_out = cf.get_growth_factor(inputs, redshift)
    if from_catalog:
        # cond_param is descendant redshift, cond_array are halo masses
        growth_in = cf.get_growth_factor(inputs, cond_param)
        sigma_cond = cf.evaluate_sigma(inputs, cond_array)
        delta = (
            cf.get_delta_crit_nu(inputs.user_params.cdict["HMF"], sigma_cond, growth_in)
            * growth_out
            / growth_in
        )
        cond_mass = np.exp(cond_array)
    else:
        # cond_param is cell mass, cond_array are deltas
        sigma_cond = cf.evaluate_sigma(inputs, np.log(cond_param))
        delta = cond_array
        cond_mass = cond_param

    out_dict = {
        "growth_out": growth_out,
        "growth_in": growth_in if from_catalog else 0.0,
        "cond_mass": cond_mass,
        "delta": delta,
        "sigma_cond": sigma_cond,
    }
    return out_dict


def integrate_massfunction(
    *,
    inputs: InputParameters,
    M_min: float,
    M_max: float,
    redshift: float,
    cond_param: float,
    cond_array: Sequence[float],
    from_catalog: bool,
):
    """
    Evaluate the conditional mass function integral.

    includes halo number and mass, using the 21cmfast backend
    """
    cond_params = get_condition_parameters(
        inputs, redshift, cond_param, cond_array, from_catalog
    )
    # clumsy, the backend function expects different arguments depending on from_catalog
    table_cond_arg = (
        cond_params["growth_in"] if from_catalog else np.log(cond_params["cond_mass"])
    )
    if inputs.user_params.USE_INTERPOLATION_TABLES == "hmf-interpolation":
        cf.initialise_dNdM_tables(
            inputs,
            cond_array.min(),
            cond_array.max(),
            M_min,
            M_max,
            cond_params["growth_out"],
            table_cond_arg,
            from_catalog,
        )

    nhalo, mcoll = cf.get_condition_integrals(
        cond_array,
        cond_params["growth_out"],
        M_min,
        M_max,
        cond_params["cond_mass"],
        cond_params["sigma_cond"],
        cond_params["delta"],
    )

    return nhalo, mcoll


def get_chmf_mass_at_probability(
    *,
    inputs,
    M_min: float,
    redshift: float,
    cond_param: float,
    cond_array: Sequence[float],
    probabilities: Sequence[float],
    from_catalog: bool,
):
    """
    Evaluate the inverse cumulative halo mass function.

    Used to verify sampling tables.
    """
    if inputs.user_params.USE_INTERPOLATION_TABLES != "hmf-interpolation":
        raise NotImplementedError(
            "Inverse mass function not supported without interpolation tables"
        )

    cond_params = get_condition_parameters(
        inputs, redshift, cond_param, cond_array, from_catalog
    )

    table_cond_arg = (
        cond_params["growth_in"] if from_catalog else np.log(cond_params["cond_mass"])
    )

    cf.initialise_dNdM_inverse_table(
        cond_array.min(),
        cond_array.max() * 1.01,
        np.log(M_min),
        cond_params["growth_out"],
        table_cond_arg,
        from_catalog,
    )
    masses = cf.evaluate_inverse_table(
        cond_array, probabilities, cond_params["cond_mass"]
    )
    return masses


def get_powerlaw_masslimits(inputs):
    """Get the mass limits for the power-law scaling relations."""
    ap_c = inputs.astro_params.cdict
    mlim_fstar_acg = (
        1e10 * ap_c["F_STAR10"] ** (-1.0 / ap_c["ALPHA_STAR"])
        if ap_c["ALPHA_STAR"]
        else 0.0
    )
    mlim_fstar_mcg = (
        1e7 * ap_c["F_STAR7_MINI"] ** (-1.0 / ap_c["ALPHA_STAR_MINI"])
        if ap_c["ALPHA_STAR_MINI"]
        else 0.0
    )
    mlim_fesc_acg = (
        1e10 * ap_c["F_ESC10"] ** (-1.0 / ap_c["ALPHA_ESC"])
        if ap_c["ALPHA_ESC"]
        else 0.0
    )
    mlim_fesc_mcg = (
        1e7 * ap_c["F_ESC7_MINI"] ** (-1.0 / ap_c["ALPHA_ESC"])
        if ap_c["ALPHA_ESC"]
        else 0.0
    )

    pl_mass_lims = {
        "fstar_acg": mlim_fstar_acg,
        "fstar_mcg": mlim_fstar_mcg,
        "fesc_acg": mlim_fesc_acg,
        "fesc_mcg": mlim_fesc_mcg,
    }
    return pl_mass_lims


def integrate_sfrd_z(
    inputs: InputParameters,
    redshifts: Sequence[float],
    M_min: float,
    M_max: float,
    log10mturnovers: Sequence[float],
):
    """Integrate the star formation rate density over mass at a range of redshifts."""
    pl_mass_lims = get_powerlaw_masslimits(inputs)

    return cf.evaluate_SFRD_z(
        inputs,
        M_min,
        M_max,
        redshifts,
        log10mturnovers,
        pl_mass_lims,
    )


def integrate_nion_z(
    inputs: InputParameters,
    redshifts: Sequence[float],
    M_min: float,
    M_max: float,
    log10mturnovers: Sequence[float],
):
    """Integrate the cumulative ionizing photon production over mass at a range of redshifts."""
    pl_mass_lims = get_powerlaw_masslimits(inputs)

    return cf.evaluate_Nion_z(
        inputs,
        M_min,
        M_max,
        redshifts,
        log10mturnovers,
        pl_mass_lims,
    )
