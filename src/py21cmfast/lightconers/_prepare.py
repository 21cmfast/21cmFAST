"""What the lightcone driver does to a lightconer before a run."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from astropy.cosmology import CosmologyError, z_at_value

from ..wrapper.classy_interface import run_classy
from ..wrapper.inputs import InputParameters
from .protocol import LightconerProtocol

if TYPE_CHECKING:
    from classy import Class


def prepare_lightconer(
    lightconer: LightconerProtocol,
    inputs: InputParameters,
    include_dvdr_in_tau21: bool,
    apply_rsds: bool,
    classy_output: Class | None = None,
) -> LightconerProtocol:
    """Check a lightconer against the inputs of a run, and return the one to use.

    The lightconer that is returned may differ from the one given: it makes the
    extra quantities that the velocity corrections need (``"los_velocity"`` and
    ``"tau_21"``), and, if redshift-space distortions are applied, its lightcone
    extends further along the line of sight (the extra slices are removed again at
    the end of the run).

    Parameters
    ----------
    lightconer
        The lightconer given by the user.
    inputs
        The input parameters of the run.
    include_dvdr_in_tau21
        Whether velocity gradient corrections are applied to the 21cm optical depth.
    apply_rsds
        Whether the lightcones are transformed to redshift space.
    classy_output
        The output of CLASS, used to estimate how far redshift-space distortions
        move the cells. It is computed if not given (and needed).

    Returns
    -------
    LightconerProtocol
        The lightconer to make the lightcone with.
    """
    from ._legacy import Lightconer

    kind = type(lightconer)
    if (
        Lightconer in kind.__mro__
        and kind.validate_options is not Lightconer.validate_options
    ):
        # A lightconer written for v4.3 that does its own validation.
        return lightconer.validate_options(
            inputs=inputs,
            include_dvdr_in_tau21=include_dvdr_in_tau21,
            apply_rsds=apply_rsds,
            classy_output=classy_output,
        )
    return _prepare_lightconer(
        lightconer,
        inputs=inputs,
        include_dvdr_in_tau21=include_dvdr_in_tau21,
        apply_rsds=apply_rsds,
        classy_output=classy_output,
    )


def _prepare_lightconer(
    lightconer: LightconerProtocol,
    inputs: InputParameters,
    include_dvdr_in_tau21: bool,
    apply_rsds: bool,
    classy_output: Class | None = None,
) -> LightconerProtocol:
    if len(inputs.node_redshifts) == 0:
        raise ValueError(
            "You are attempting to run a lightcone with no node_redshifts."
            "Please set node_redshifts on the `inputs` parameter."
        )

    cosmo = inputs.cosmo_params.cosmo
    if getattr(lightconer, "cosmo", cosmo) != cosmo:
        raise ValueError(
            "The cosmology used to construct the lightcone is not the same as the cosmology in the input parameters. "
            "Please make sure that your lightconer.cosmo is the same as inputs.cosmo_params.cosmo."
        )

    lcd = lightconer.lc_distances
    d_node_min = cosmo.comoving_distance(min(inputs.node_redshifts))
    d_node_max = cosmo.comoving_distance(max(inputs.node_redshifts))
    if not ((d_node_min <= lcd.min()) and (lcd.max() <= d_node_max)):
        try:
            with warnings.catch_warnings():
                # astropy warns before it raises, if there is no such redshift.
                warnings.simplefilter("ignore")
                zmin = z_at_value(cosmo.comoving_distance, lcd.min()).value
                zmax = z_at_value(cosmo.comoving_distance, lcd.max()).value
            lc_range = f"{zmin} to {zmax}"
        except CosmologyError:
            # Distances that no (reasonable) redshift corresponds to.
            lc_range = f"that of the distances {lcd.min()} to {lcd.max()}"
        raise ValueError(
            "The lightcone redshifts are not compatible with the given redshift."
            f"The range of computed redshifts is {min(inputs.node_redshifts)} to {max(inputs.node_redshifts)}, "
            f"while the lightcone redshift range is {lc_range}. "
            "Extend the limits of node redshifts to avoid this error."
        )

    if (
        include_dvdr_in_tau21
        and inputs.astro_options.USE_TS_FLUCT
        and "tau_21" not in lightconer.quantities
    ):
        lightconer = lightconer.evolve(quantities=(*lightconer.quantities, "tau_21"))

    if apply_rsds or include_dvdr_in_tau21:
        lightconer = lightconer.evolve(
            quantities=(*lightconer.quantities, "los_velocity")
        )
        needed = set(lightconer.required_fields())
        if (
            needed & {"velocity_x", "velocity_y"}
            and not inputs.matter_options.KEEP_3D_VELOCITIES
        ):
            raise ValueError(
                "To account for RSDs or velocity corrections in an angular lightcone, you need to set "
                "matter_options.KEEP_3D_VELOCITIES=True"
            )

    if not apply_rsds:
        return lightconer

    from ..rsds import find_required_lightcone_limits

    if classy_output is None:
        classy_output = run_classy(
            h=inputs.cosmo_params.hlittle,
            Omega_cdm=inputs.cosmo_params.OMm - inputs.cosmo_params.OMb,
            Omega_b=inputs.cosmo_params.OMb,
            n_s=inputs.cosmo_params.POWER_INDEX,
            sigma8=inputs.cosmo_params.SIGMA_8,
            output="vTk",
        )
    lcd_limits_rsd = find_required_lightcone_limits(
        lightconer, classy_output=classy_output, inputs=inputs
    )
    lcd_rsd = (
        np.arange(
            lcd_limits_rsd[0].value,
            lcd_limits_rsd[1].value + inputs.simulation_options.cell_size.value,
            inputs.simulation_options.cell_size.value,
        )
        * lcd.unit
    )
    # Make a new lightconer which is identical, but extends to further boundaries!
    return lightconer.evolve(lc_distances=lcd_rsd)
