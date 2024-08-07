"""Compute simulations that evolve over redshift."""

import logging
import numpy as np
import os
from pathlib import Path
from typing import Any

from ..c_21cmfast import lib
from ..photoncons import _get_photon_nonconservation_data, setup_photon_cons
from ..wrapper.globals import global_params
from ..wrapper.inputs import AstroParams, CosmoParams, FlagOptions, UserParams
from ..wrapper.outputs import Coeval, InitialConditions, PerturbedField
from . import single_field as sf
from .param_config import InputParameters, _get_config_options, _setup_inputs
from .single_field import set_globals

logger = logging.getLogger(__name__)


def get_logspaced_redshifts(min_redshift: float, z_step_factor: float, zmax: float):
    """Compute a sequence of redshifts to evolve over that are log-spaced."""
    redshifts = [min_redshift]
    while redshifts[-1] < zmax:
        redshifts.append((redshifts[-1] + 1.0) * z_step_factor - 1.0)

    return redshifts


@set_globals
def run_coeval(
    *,
    redshift: float | np.ndarray | None = None,
    user_params: UserParams | dict | None = None,
    cosmo_params: CosmoParams | dict | None = None,
    flag_options: FlagOptions | dict | None = None,
    astro_params: AstroParams | dict | None = None,
    regenerate: bool | None = None,
    write: bool | None = None,
    direc: str | Path | None = None,
    initial_conditions: InitialConditions | None = None,
    perturbed_field: PerturbedField | None = None,
    random_seed: int | None = None,
    cleanup: bool = True,
    hooks: dict[callable, dict[str, Any]] | None = None,
    always_purge: bool = False,
    **global_kwargs,
):
    r"""
    Evaluate a coeval ionized box at a given redshift, or multiple redshifts.

    This is generally the easiest and most efficient way to generate a set of coeval cubes at a
    given set of redshift. It self-consistently deals with situations in which the field needs to be
    evolved, and does this with the highest memory-efficiency, only returning the desired redshift.
    All other calculations are by default stored in the on-disk cache so they can be re-used at a
    later time.

    .. note:: User-supplied redshift are *not* used as previous redshift in any scrolling,
              so that pristine log-sampling can be maintained.

    Parameters
    ----------
    redshift: array_like
        A single redshift, or multiple redshift, at which to return results. The minimum of these
        will define the log-scrolling behaviour (if necessary).
    user_params : :class:`~inputs.UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~inputs.CosmoParams` , optional
        Defines the cosmological parameters used to compute initial conditions.
    astro_params : :class:`~inputs.AstroParams` , optional
        The astrophysical parameters defining the course of reionization.
    flag_options : :class:`~inputs.FlagOptions` , optional
        Some options passed to the reionization routine.
    init_box : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not
        be re-calculated.
    perturb : list of :class:`~PerturbedField`, optional
        If given, must be compatible with init_box. It will merely negate the necessity
        of re-calculating the perturb fields.
    use_interp_perturb_field : bool, optional
        Whether to use a single perturb field, at the lowest redshift of the lightcone,
        to determine all spin temperature fields. If so, this field is interpolated in
        the underlying C-code to the correct redshift. This is less accurate (and no more
        efficient), but provides compatibility with older versions of 21cmFAST.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning.
        Typically, if `spin_temperature` is called directly, you will want this to be
        true, as if the next box to be calculate has different shape, errors will occur
        if memory is not cleaned. Note that internally, this is set to False until the
        last iteration.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    coevals : :class:`~py21cmfast.outputs.Coeval`
        The full data for the Coeval class, with init boxes, perturbed fields, ionized boxes,
        brightness temperature, and potential data from the conservation of photons. If a
        single redshift was specified, it will return such a class. If multiple redshifts
        were passed, it will return a list of such classes.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed :
        See docs of :func:`initial_conditions` for more information.
    """
    if redshift is None and perturbed_field is None:
        raise ValueError("Either redshift or perturb must be given")

    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    singleton = False
    # Ensure perturb is a list of boxes, not just one.
    if perturbed_field is None:
        perturbed_field = []
    elif not hasattr(perturbed_field, "__len__"):
        perturbed_field = [perturbed_field]
        singleton = True

    random_seed = (
        initial_conditions.random_seed
        if initial_conditions is not None
        else random_seed
    )

    inputs = InputParameters(
        user_params=user_params,
        cosmo_params=cosmo_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )

    iokw = {"regenerate": regenerate, "hooks": hooks, "direc": direc}

    if initial_conditions is None:
        initial_conditions = sf.initial_conditions(
            user_params=inputs.user_params,
            cosmo_params=inputs.cosmo_params,
            random_seed=random_seed,
            **iokw,
        )

    # We can go ahead and purge some of the stuff in the init_box, but only if
    # it is cached -- otherwise we could be losing information.
    try:
        initial_conditions.prepare_for_perturb(
            flag_options=flag_options, force=always_purge
        )
    except OSError:
        pass

    if perturbed_field:
        if redshift is not None and any(
            p.redshift != z for p, z in zip(perturbed_field, redshift)
        ):
            raise ValueError("Input redshifts do not match perturb field redshifts")
        else:
            redshift = [p.redshift for p in perturbed_field]

    kw = {
        **{
            "astro_params": inputs.astro_params,
            "flag_options": inputs.flag_options,
            "init_boxes": initial_conditions,
        },
        **iokw,
    }
    photon_nonconservation_data = None
    if flag_options.PHOTON_CONS_TYPE != 0:
        photon_nonconservation_data = setup_photon_cons(**kw)

    if not hasattr(redshift, "__len__"):
        singleton = True
        redshift = [redshift]

    if isinstance(redshift, np.ndarray):
        redshift = redshift.tolist()

    # Get the list of redshift we need to scroll through.
    redshifts = _get_required_redshifts_coeval(flag_options, redshift)

    # Get all the perturb boxes early. We need to get the perturb at every
    # redshift.
    pz = [p.redshift for p in perturbed_field]
    perturb_ = []
    for z in redshifts:
        p = (
            sf.perturb_field(redshift=z, init_boxes=initial_conditions, **iokw)
            if z not in pz
            else perturbed_field[pz.index(z)]
        )

        if user_params.MINIMIZE_MEMORY:
            try:
                p.purge(force=always_purge)
            except OSError:
                pass

        perturb_.append(p)

    perturbed_field = perturb_

    # Now we can purge init_box further.
    try:
        initial_conditions.prepare_for_halos(
            flag_options=flag_options, force=always_purge
        )
    except OSError:
        pass

    # get the halos (reverse redshift order)
    pt_halos = []
    if inputs.flag_options.USE_HALO_FIELD and not inputs.flag_options.FIXED_HALO_GRIDS:
        halos_desc = None
        for i, z in enumerate(redshifts[::-1]):
            halos = sf.determine_halo_list(redshift=z, halos_desc=halos_desc, **kw)
            pt_halos += [sf.perturb_halo_list(redshift=z, halo_field=halos, **kw)]

            # we never want to store every halofield
            try:
                pt_halos[i].purge(force=always_purge)
            except OSError:
                pass
            halos_desc = halos

        # reverse to get the right redshift order
        pt_halos = pt_halos[::-1]

    # Now we can purge init_box further.
    try:
        initial_conditions.prepare_for_spin_temp(
            flag_options=flag_options, force=always_purge
        )
    except OSError:
        pass

    if (
        flag_options.PHOTON_CONS_TYPE == 1
        and np.amin(redshifts) < global_params.PhotonConsEndCalibz
    ):
        raise ValueError(
            f"You have passed a redshift (z = {np.amin(redshifts)}) that is lower than"
            "the endpoint of the photon non-conservation correction"
            f"(global_params.PhotonConsEndCalibz = {global_params.PhotonConsEndCalibz})."
            "If this behaviour is desired then set global_params.PhotonConsEndCalibz"
            f"to a value lower than z = {np.amin(redshifts)}."
        )

    ib_tracker = [0] * len(redshift)
    bt = [0] * len(redshift)
    # At first we don't have any "previous" st or ib.
    st, ib, pf, hb = None, None, None, None
    # optional fields which remain None if their flags are off
    hb2, ph2 = None, None

    hb_tracker = [None] * len(redshift)
    st_tracker = [None] * len(redshift)

    spin_temp_files = []
    hbox_files = []
    perturb_files = []
    ionize_files = []
    brightness_files = []
    pth_files = []

    # Iterate through redshift from top to bottom
    z_halos = []
    hbox_arr = []
    for iz, z in enumerate(redshifts):
        pf2 = perturbed_field[iz]
        pf2.load_all()

        if flag_options.USE_HALO_FIELD:
            if not flag_options.FIXED_HALO_GRIDS:
                ph2 = pt_halos[iz]

            hb2 = sf.compute_halo_grid(
                redshift=z,
                pt_halos=ph2,
                perturbed_field=pf2,
                previous_ionize_box=ib,
                previous_spin_temp=st,
                **kw,
            )

        if flag_options.USE_TS_FLUCT:
            # append the halo redshift array so we have all halo boxes [z,zmax]
            z_halos += [z]
            hbox_arr += [hb2]
            if flag_options.USE_HALO_FIELD:
                xray_source_box = sf.compute_xray_source_field(
                    redshift=z,
                    z_halos=z_halos,
                    hboxes=hbox_arr,
                    **kw,
                )

            st2 = sf.spin_temperature(
                redshift=z,
                previous_spin_temp=st,
                perturbed_field=pf2,
                xray_source_box=(
                    xray_source_box if inputs.flag_options.USE_HALO_FIELD else None
                ),
                **kw,
                cleanup=(cleanup and z == redshifts[-1]),
            )

            if z not in redshift:
                st = st2

        ib2 = sf.compute_ionization_field(
            redshift=z,
            previous_ionize_box=ib,
            perturbed_field=pf2,
            # perturb field *not* interpolated here.
            previous_perturbed_field=pf,
            halobox=hb2,
            spin_temp=st2 if inputs.flag_options.USE_TS_FLUCT else None,
            z_heat_max=global_params.Z_HEAT_MAX,
            # cleanup if its the last time through
            cleanup=cleanup and z == redshifts[-1],
            **kw,
        )

        if pf is not None:
            try:
                pf.purge(force=always_purge)
            except OSError:
                pass

        if ph2 is not None:
            try:
                ph2.purge(force=always_purge)
            except OSError:
                pass

        # we only need the SFR fields at previous redshifts for XraySourceBox
        if hb is not None:
            try:
                hb.prepare(
                    keep=[
                        "halo_sfr",
                        "halo_sfr_mini",
                        "halo_xray",
                        "log10_Mcrit_MCG_ave",
                    ],
                    force=always_purge,
                )
            except OSError:
                pass

        if z in redshift:
            logger.debug(f"PID={os.getpid()} doing brightness temp for z={z}")
            ib_tracker[redshift.index(z)] = ib2
            st_tracker[redshift.index(z)] = (
                st2 if inputs.flag_options.USE_TS_FLUCT else None
            )

            hb_tracker[redshift.index(z)] = (
                hb2 if inputs.flag_options.USE_HALO_FIELD else None
            )

            _bt = sf.brightness_temperature(
                ionized_box=ib2,
                perturbed_field=pf2,
                spin_temp=st2 if inputs.flag_options.USE_TS_FLUCT else None,
                **iokw,
            )

            bt[redshift.index(z)] = _bt

        else:
            ib = ib2
            pf = pf2
            _bt = None
            hb = hb2

        perturb_files.append((z, os.path.join(direc, pf2.filename)))
        if inputs.flag_options.USE_HALO_FIELD:
            hbox_files.append((z, os.path.join(direc, hb2.filename)))
            pth_files.append((z, os.path.join(direc, ph2.filename)))
        if inputs.flag_options.USE_TS_FLUCT:
            spin_temp_files.append((z, os.path.join(direc, st2.filename)))
        ionize_files.append((z, os.path.join(direc, ib2.filename)))

        if _bt is not None:
            brightness_files.append((z, os.path.join(direc, _bt.filename)))

    if inputs.flag_options.PHOTON_CONS_TYPE == 1:
        photon_nonconservation_data = _get_photon_nonconservation_data()

    if lib.photon_cons_allocated:
        lib.FreePhotonConsMemory()

    coevals = [
        Coeval(
            redshift=z,
            initial_conditions=initial_conditions,
            perturbed_field=perturbed_field[redshifts.index(z)],
            ionized_box=ib,
            brightness_temp=_bt,
            ts_box=st,
            halobox=hb if flag_options.USE_HALO_FIELD else None,
            photon_nonconservation_data=photon_nonconservation_data,
            cache_files={
                "init": [(0, os.path.join(direc, initial_conditions.filename))],
                "perturb_field": perturb_files,
                "halobox": hbox_files,
                "ionized_box": ionize_files,
                "brightness_temp": brightness_files,
                "spin_temp": spin_temp_files,
                "pt_halos": pth_files,
            },
        )
        for z, ib, _bt, st, hb in zip(redshift, ib_tracker, bt, st_tracker, hb_tracker)
    ]

    # If a single redshift was passed, then pass back singletons.
    if singleton:
        coevals = coevals[0]

    logger.debug("Returning from Coeval")

    return coevals


def _get_required_redshifts_coeval(flag_options, redshift) -> list[float]:
    if min(redshift) < global_params.Z_HEAT_MAX and (
        flag_options.INHOMO_RECO or flag_options.USE_TS_FLUCT
    ):
        redshifts = get_logspaced_redshifts(
            min(redshift),
            global_params.ZPRIME_STEP_FACTOR,
            global_params.Z_HEAT_MAX,
        )
        # Set the highest redshift to exactly Z_HEAT_MAX. This makes the coeval run
        # at exactly the same redshift as the spin temperature box. There's literally
        # no point going higher for a coeval, since the user is only interested in
        # the final "redshift" (if they've specified a z in redshift that is higher
        # that Z_HEAT_MAX, we add it back in below, and so they'll still get it).
        redshifts = np.clip(redshifts, None, global_params.Z_HEAT_MAX)

    else:
        redshifts = [min(redshift)]
    # Add in the redshift defined by the user, and sort in order
    # Turn into a set so that exact matching user-set redshift
    # don't double-up with scrolling ones.
    redshifts = np.concatenate((redshifts, redshift))
    redshifts = np.sort(np.unique(redshifts))[::-1]
    return redshifts.tolist()


def _get_coeval_callbacks(
    scrollz: list[float], coeval_callback, coeval_callback_redshifts
) -> list[bool]:
    compute_coeval_callback = [False] * len(scrollz)

    if coeval_callback is not None:
        if isinstance(coeval_callback_redshifts, (list, np.ndarray)):
            for coeval_z in coeval_callback_redshifts:
                assert isinstance(coeval_z, (int, float, np.number))
                compute_coeval_callback[
                    np.argmin(np.abs(np.array(scrollz) - coeval_z))
                ] = True
            if sum(compute_coeval_callback) != len(coeval_callback_redshifts):
                logger.warning(
                    "some of the coeval_callback_redshifts refer to the same node_redshift"
                )
        elif (
            isinstance(coeval_callback_redshifts, int) and coeval_callback_redshifts > 0
        ):
            compute_coeval_callback = [
                not i % coeval_callback_redshifts for i in range(len(scrollz))
            ]
        else:
            raise ValueError("coeval_callback_redshifts has to be list or integer > 0.")

    return compute_coeval_callback
