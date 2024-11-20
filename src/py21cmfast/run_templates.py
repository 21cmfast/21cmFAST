"""
Parameter Templates.

This file contains parameter default templates for use with sets of parameters which
are set to reproduce a certain published result or which we know work together.

These should make it easier to define a run, choosing the closest template to your
desired parameters and altering as few as possible.
"""

import json
import logging
from astropy.cosmology import Planck15
from os import path

logger = logging.getLogger(__name__)

# Cosmology is from https://arxiv.org/pdf/1807.06209.pdf
# Table 2, last column. [TT,TE,EE+lowE+lensing+BAO]
Planck18 = Planck15.clone(
    Om0=(0.02242 + 0.11933) / 0.6766**2,
    Ob0=0.02242 / 0.6766**2,
    H0=67.66,
    name="Planck18",
)

# TODO: could/should these be made into alterable .json files?
# TODO: Should these be made to match the actual paper fiducials
#   OR the kinds of runs they did?
_param_template_park19 = {
    "_aliases": ("park19", "simple"),
    # CosmoParams entries
    "SIGMA_8": 0.8102,
    "hlittle": Planck18.h,
    "OMm": Planck18.Om0,
    "OMb": Planck18.Ob0,
    "POWER_INDEX": 0.9665,
    # UserParams entries
    "BOX_LEN": 300.0,
    "DIM": None,  # dynamic default of 3*HII_DIM
    "HII_DIM": 200,
    "NON_CUBIC_FACTOR": 1.0,
    "USE_FFTW_WISDOM": False,
    "HMF": "ST",
    "USE_RELATIVE_VELOCITIES": False,
    "POWER_SPECTRUM": None,  # dynamic default of "CLASS" is USE_RELATIVE_VELOCITIES else "EH"
    "N_THREADS": 1,
    "PERTURB_ON_HIGH_RES": False,
    "NO_RNG": False,
    "USE_INTERPOLATION_TABLES": True,
    "INTEGRATION_METHOD_ATOMIC": "GAUSS-LEGENDRE",
    "INTEGRATION_METHOD_MINI": "GAUSS-LEGENDRE",
    "USE_2LPT": True,
    "MINIMIZE_MEMORY": False,
    "KEEP_3D_VELOCITIES": False,
    "SAMPLER_MIN_MASS": 1e8,
    "SAMPLER_BUFFER_FACTOR": 2.0,
    "MAXHALO_FACTOR": 2.0,
    "N_COND_INTERP": 200,
    "N_PROB_INTERP": 400,
    "MIN_LOGPROB": -12.0,
    "SAMPLE_METHOD": "MASS-LIMITED",
    "AVG_BELOW_SAMPLER": True,
    "HALOMASS_CORRECTION": 0.9,
    "PARKINSON_G0": 1.0,
    "PARKINSON_y1": 0.0,
    "PARKINSON_y2": 0.0,
    # AstroParams entries
    "HII_EFF_FACTOR": 30.0,
    "F_STAR10": -1.3,
    "F_STAR7_MINI": None,  # dynamic default to continue ACG scaling
    "ALPHA_STAR": 0.5,
    "ALPHA_STAR_MINI": None,  # dynamic default to match ALPHA_STAR
    "F_ESC10": -1.0,
    "F_ESC7_MINI": None,  # dynamic default to continue ACG scaling
    "ALPHA_ESC": -0.5,
    "M_TURN": None,  # dynamic default to 5 if USE_MINI_HALOS else 8.7 REQUIRES FLAG OPTIONS
    "R_BUBBLE_MAX": None,  # dynamic default of 50 if INHOMO_RECO else 15 REQUIRES FLAG OPTIONS
    "ION_Tvir_MIN": 4.69897,
    "L_X": 40.5,
    "L_X_MINI": None,  # dynamic default to match L_X
    "NU_X_THRESH": 500.0,
    "X_RAY_SPEC_INDEX": 1.0,
    "X_RAY_Tvir_MIN": None,  # dynamic default to match ION_Tvir_MIN
    "F_H2_SHIELD": 0.0,
    "t_STAR": 0.5,
    "N_RSD_STEPS": 20,
    "A_LW": 2.00,
    "BETA_LW": 0.6,
    "A_VCB": 1.0,
    "BETA_VCB": 1.8,
    "UPPER_STELLAR_TURNOVER_MASS": 11.447,
    "UPPER_STELLAR_TURNOVER_INDEX": -0.6,
    "SIGMA_STAR": 0.25,
    "SIGMA_LX": 0.5,
    "SIGMA_SFR_LIM": 0.19,
    "SIGMA_SFR_INDEX": -0.12,
    "CORR_STAR": 0.5,
    "CORR_SFR": 0.2,
    "CORR_LX": 0.2,
    # FlagOptions entries
    "USE_HALO_FIELD": False,
    "USE_MINI_HALOS": False,
    "USE_CMB_HEATING": True,
    "USE_LYA_HEATING": True,
    "USE_MASS_DEPENDENT_ZETA": True,
    "SUBCELL_RSD": False,
    "APPLY_RSDS": True,
    "INHOMO_RECO": False,
    "USE_TS_FLUCT": False,
    "M_MIN_in_Mass": True,
    "PHOTON_CONS_TYPE": "no-photoncons",
    "FIX_VCB_AVG": False,
    "HALO_STOCHASTICITY": False,
    "HALO_SCALING_RELATIONS_MEDIAN": False,
    "USE_UPPER_STELLAR_TURNOVER": False,
    "FIXED_HALO_GRIDS": False,
    "CELL_RECOMB": False,
    "USE_EXP_FILTER": False,
}

_param_template_qin20 = {
    "_aliases": ("qin20", "minihalos"),
    "USE_MINI_HALOS": True,
    "USE_TS_FLUCT": True,
    "INHOMO_RECO": True,
    "USE_RELATIVE_VELOCITIES": True,
}
_param_template_qin20 = {**_param_template_park19, **_param_template_qin20}

_param_template_davies24 = {
    "_aliases": ("davies24", "halosampler"),
    "USE_HALO_FIELD": True,
    "HALO_STOCHASTICITY": True,
    "USE_TS_FLUCT": True,
    "INHOMO_RECO": True,
    "USE_EXP_FILTER": True,
    "CELL_RECOMB": True,
    "USE_UPPER_STELLAR_TURNOVER": True,
    "L_X": 40.0,
}
_param_template_davies24 = {**_param_template_park19, **_param_template_davies24}

available_param_templates = (
    _param_template_park19,
    _param_template_qin20,
    _param_template_davies24,
)

# park19 as default template
# TODO: consider davies24 as default template?
active_param_template = _param_template_park19


def verify_template(template: dict):
    """
    Ensure a parameter template has all required keys for setting parameters and no more.

    Uses the park19 template as a comparison
    """
    keys_test = [k for k in template.keys() if not k.startswith("_")]
    keys_true = [k for k in _param_template_park19.keys() if not k.startswith("_")]

    keys_excess = [k for k in keys_test if k not in keys_true]
    keys_missing = [k for k in keys_true if k not in keys_test]

    if keys_test != keys_true:
        message = (
            f"There are extra or missing paramters in the given parameter template.\n"
            f"EXTRAS: {keys_excess}\n"
            f"MISSING: {keys_missing}\n"
        )
        raise ValueError(message)


def set_active_template(
    template_name: str,
):
    """Sets the active parameter template used in parameter construction."""
    # First check if the provided name is a path to an existsing JSON file
    if path.isfile(template_name):
        try:
            template = json.load(template_name)
        except OSError:
            logger.info(f"loaded template from {template_name}...")
        else:
            verify_template(template)  # throws an error if not valid
            active_param_template = template

    # Next, check if the string matches one of our template aliases
    for template in available_param_templates:
        if template_name.lower in template["aliases"]:
            verify_template(template)
            active_param_template = template
            return

    # We have not found a template in our templates or on file, raise an error
    logger.error(
        f"Template {template_name} not found on-disk or in native template aliases"
    )
    logger.error("Available native templates are:")
    [
        logger.error(f"Template {i} names: {t['aliases']}")
        for i, t in enumerate(available_param_templates)
    ]
    raise ValueError(
        f"template {template_name} not found on-disk or in native template aliases"
    )
