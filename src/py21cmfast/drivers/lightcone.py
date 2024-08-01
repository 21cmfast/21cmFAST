"""Module containing a driver function for creating lightcones."""

import logging
import numpy as np
import os
import warnings
from pathlib import Path

from ..c_21cmfast import lib
from ..cache_tools import get_boxes_at_redshift
from ..lightcones import Lightconer, RectilinearLightconer
from ..photoncons import _get_photon_nonconservation_data, setup_photon_cons
from ..wrapper.globals import global_params
from ..wrapper.inputs import CosmoParams
from ..wrapper.outputs import AngularLightcone, Coeval, LightCone
from ..wrapper.param_config import _get_config_options, _setup_inputs
from . import single_field as sf
from .coeval import _get_coeval_callbacks, get_logspaced_redshifts

logger = logging.getLogger(__name__)


def run_lightcone(
    *,
    redshift: float = None,
    max_redshift: float = None,
    lightcone_quantities=("brightness_temp",),
    lightconer: Lightconer | None = None,
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
    regenerate=None,
    write=None,
    global_quantities=("brightness_temp", "xH_box"),
    direc=None,
    init_box=None,
    perturb=None,
    random_seed=None,
    coeval_callback=None,
    coeval_callback_redshifts=1,
    use_interp_perturb_field=False,
    cleanup=True,
    hooks=None,
    always_purge: bool = False,
    lightcone_filename: str | Path = None,
    return_at_z: float = 0.0,
    **global_kwargs,
):
    r"""
    Evaluate a full lightcone ending at a given redshift.

    This is generally the easiest and most efficient way to generate a lightcone, though it can
    be done manually by using the lower-level functions which are called by this function.

    Parameters
    ----------
    redshift : float
        The minimum redshift of the lightcone.
    max_redshift : float, optional
        The maximum redshift at which to keep lightcone information. By default, this is equal to
        `z_heat_max`. Note that this is not *exact*, but will be typically slightly exceeded.
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.
    astro_params : :class:`~AstroParams`, optional
        Defines the astrophysical parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    flag_options : :class:`~FlagOptions`, optional
        Options concerning how the reionization process is run, eg. if spin temperature
        fluctuations are required.
    lightcone_quantities : tuple of str, optional
        The quantities to form into a lightcone. By default, just the brightness
        temperature. Note that these quantities must exist in one of the output
        structures:

        * :class:`~InitialConditions`
        * :class:`~PerturbField`
        * :class:`~TsBox`
        * :class:`~IonizedBox`
        * :class:`BrightnessTemp`

        To get a full list of possible quantities, run :func:`get_all_fieldnames`.
    global_quantities : tuple of str, optional
        The quantities to save as globally-averaged redshift-dependent functions.
        These may be any of the quantities that can be used in ``lightcone_quantities``.
        The mean is taken over the full 3D cube at each redshift, rather than a 2D
        slice.
    init_box : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not be
        re-calculated.
    perturb : list of :class:`~PerturbedField`, optional
        If given, must be compatible with init_box. It will merely negate the necessity of
        re-calculating the
        perturb fields. It will also be used to set the redshift if given.
    coeval_callback : callable, optional
        User-defined arbitrary function computed on :class:`~Coeval`, at redshifts defined in
        `coeval_callback_redshifts`.
        If given, the function returns :class:`~LightCone` and the list of `coeval_callback` outputs.
    coeval_callback_redshifts : list or int, optional
        Redshifts for `coeval_callback` computation.
        If list, computes the function on `node_redshifts` closest to the specified ones.
        If positive integer, computes the function on every n-th redshift in `node_redshifts`.
        Ignored in the case `coeval_callback is None`.
    use_interp_perturb_field : bool, optional
        Whether to use a single perturb field, at the lowest redshift of the lightcone,
        to determine all spin temperature fields. If so, this field is interpolated in the
        underlying C-code to the correct redshift. This is less accurate (and no more efficient),
        but provides compatibility with older versions of 21cmFAST.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning.
        Typically, if `spin_temperature` is called directly, you will want this to be
        true, as if the next box to be calculate has different shape, errors will occur
        if memory is not cleaned. Note that internally, this is set to False until the
        last iteration.
    minimize_memory_usage
        If switched on, the routine will do all it can to minimize peak memory usage.
        This will be at the cost of disk I/O and CPU time. Recommended to only set this
        if you are running particularly large boxes, or have low RAM.
    lightcone_filename
        The filename to which to save the lightcone. The lightcone is returned in
        memory, and can be saved manually later, but including this filename will
        save the lightcone on each iteration, which can be helpful for checkpointing.
    return_at_z
        If given, evaluation of the lightcone will be stopped at the given redshift,
        and the partial lightcone object will be returned. Lightcone evaluation can
        continue if the returned lightcone is saved to file, and this file is passed
        as `lightcone_filename`.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    lightcone : :class:`~py21cmfast.LightCone`
        The lightcone object.
    coeval_callback_output : list
        Only if coeval_callback in not None.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed
        See docs of :func:`initial_conditions` for more information.
    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    if cosmo_params is None and lightconer is not None:
        cosmo_params = CosmoParams.from_astropy(lightconer.cosmo)

    with global_params.use(**global_kwargs):
        # First, get the parameters OTHER than redshift...

        (
            random_seed,
            user_params,
            cosmo_params,
            flag_options,
            astro_params,
        ) = _setup_inputs(
            {
                "random_seed": random_seed,
                "user_params": user_params,
                "cosmo_params": cosmo_params,
                "flag_options": flag_options,
                "astro_params": astro_params,
            },
            {"init_box": init_box, "perturb": perturb},
        )

        if redshift is None and perturb is None:
            if lightconer is None:
                raise ValueError(
                    "You must provide either redshift, perturb or lightconer"
                )
            else:
                redshift = lightconer.lc_redshifts.min()

        elif redshift is None:
            redshift = perturb.redshift
        elif redshift is not None:
            warnings.warn(
                "passing redshift directly is deprecated, please use the Lightconer interface instead",
                category=DeprecationWarning,
            )

        if user_params.MINIMIZE_MEMORY and not write:
            raise ValueError(
                "If trying to minimize memory usage, you must be caching. Set write=True!"
            )

        max_redshift = (
            global_params.Z_HEAT_MAX
            if (
                flag_options.INHOMO_RECO
                or flag_options.USE_TS_FLUCT
                or (max_redshift is None and lightconer is None)
            )
            else (
                max_redshift
                if max_redshift is not None
                else lightconer.lc_redshifts.max()
            )
        )

        if lightconer is None:
            lightconer = RectilinearLightconer.with_equal_cdist_slices(
                min_redshift=redshift,
                max_redshift=max_redshift,
                resolution=user_params.cell_size,
                cosmo=cosmo_params.cosmo,
                quantities=lightcone_quantities,
                get_los_velocity=not flag_options.APPLY_RSDS,
            )
        lightconer.validate_options(user_params, flag_options)

        # Get the redshift through which we scroll and evaluate the ionization field.
        scrollz = np.array(
            get_logspaced_redshifts(
                redshift, global_params.ZPRIME_STEP_FACTOR, max_redshift
            )
        )

        lcz = lightconer.lc_redshifts
        if not np.all(min(scrollz) * 0.99 < lcz) and np.all(lcz < max(scrollz) * 1.01):
            # We have a 1% tolerance on the redshifts, because the lightcone redshifts are
            # computed via inverse fitting the comoving_distance.
            raise ValueError(
                "The lightcone redshifts are not compatible with the given redshift."
                f"The range of computed redshifts is {min(scrollz)} to {max(scrollz)}, "
                f"while the lightcone redshift range is {lcz.min()} to {lcz.max()}."
            )

        if (
            flag_options.PHOTON_CONS_TYPE == 1
            and np.amin(scrollz) < global_params.PhotonConsEndCalibz
        ):
            raise ValueError(
                f"""
                You have passed a redshift (z = {np.amin(scrollz)}) that is lower than the endpoint
                of the photon non-conservation correction
                (global_params.PhotonConsEndCalibz = {global_params.PhotonConsEndCalibz}).
                If this behaviour is desired then set global_params.PhotonConsEndCalibz to a value lower than
                z = {np.amin(scrollz)}.
                """
            )

        coeval_callback_output = []
        compute_coeval_callback = _get_coeval_callbacks(
            scrollz, coeval_callback, coeval_callback_redshifts
        )

        iokw = {"hooks": hooks, "regenerate": regenerate, "direc": direc}

        if init_box is None:  # no need to get cosmo, user params out of it.
            init_box = sf.initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                random_seed=random_seed,
                **iokw,
            )

        # We can go ahead and purge some of the stuff in the init_box, but only if
        # it is cached -- otherwise we could be losing information.
        try:
            # TODO: should really check that the file at path actually contains a fully
            # working copy of the init_box.
            init_box.prepare_for_perturb(flag_options=flag_options, force=always_purge)
        except OSError:
            pass

        if lightcone_filename and Path(lightcone_filename).exists():
            lightcone = LightCone.read(lightcone_filename)
            scrollz = scrollz[scrollz < lightcone._current_redshift]
            if len(scrollz) == 0:
                # The entire lightcone is already full!
                logger.info(
                    f"Lightcone already full at z={lightcone._current_redshift}. Returning."
                )
                return lightcone
            lc = lightcone.lightcones
        else:
            lcn_cls = (
                LightCone
                if isinstance(lightconer, RectilinearLightconer)
                else AngularLightcone
            )
            lc = {
                quantity: np.zeros(
                    lightconer.get_shape(user_params),
                    dtype=np.float32,
                )
                for quantity in lightconer.quantities
            }

            # Special case: AngularLightconer can also save los_velocity
            if getattr(lightconer, "get_los_velocity", False):
                lc["los_velocity"] = np.zeros(
                    lightconer.get_shape(user_params), dtype=np.float32
                )

            lightcone = lcn_cls(
                redshift,
                lightconer.lc_distances,
                user_params,
                cosmo_params,
                astro_params,
                flag_options,
                init_box.random_seed,
                lc,
                node_redshifts=scrollz,
                log10_mturnovers=np.zeros_like(scrollz),
                log10_mturnovers_mini=np.zeros_like(scrollz),
                global_quantities={
                    quantity: np.zeros(len(scrollz)) for quantity in global_quantities
                },
                _globals=dict(global_params.items()),
            )

        if perturb is None:
            zz = scrollz
        else:
            zz = scrollz[:-1]

        perturb_ = []
        for z in zz:
            p = sf.perturb_field(redshift=z, init_boxes=init_box, **iokw)
            if user_params.MINIMIZE_MEMORY:
                try:
                    p.purge(force=always_purge)
                except OSError:
                    pass

            perturb_.append(p)

        if perturb is not None:
            perturb_.append(perturb)
        perturb = perturb_
        perturb_min = perturb[np.argmin(scrollz)]

        # Now that we've got all the perturb fields, we can purge init more.
        try:
            init_box.prepare_for_spin_temp(
                flag_options=flag_options, force=always_purge
            )
        except OSError:
            pass

        kw = {
            **{
                "init_boxes": init_box,
                "astro_params": astro_params,
                "flag_options": flag_options,
            },
            **iokw,
        }

        photon_nonconservation_data = None
        if flag_options.PHOTON_CONS_TYPE != 0:
            setup_photon_cons(**kw)

        if return_at_z > lightcone.redshift and not write:
            raise ValueError(
                "Returning before the final redshift requires caching in order to "
                "continue the simulation later. Set write=True!"
            )

        # Iterate through redshift from top to bottom
        if lightcone.redshift != lightcone._current_redshift:
            logger.info(
                f"Finding boxes at z={lightcone._current_redshift} with seed {lightcone.random_seed} and direc={direc}"
            )
            cached_boxes = get_boxes_at_redshift(
                redshift=lightcone._current_redshift,
                seed=lightcone.random_seed,
                direc=direc,
                user_params=user_params,
                cosmo_params=cosmo_params,
                flag_options=flag_options,
                astro_params=astro_params,
            )
            try:
                st = cached_boxes["TsBox"][0] if flag_options.USE_TS_FLUCT else None
                prev_perturb = cached_boxes["PerturbedField"][0]
                ib = cached_boxes["IonizedBox"][0]
            except (KeyError, IndexError):
                raise OSError(
                    f"No component boxes found at z={lightcone._current_redshift} with "
                    f"seed {lightcone.random_seed} and direc={direc}. You need to have "
                    "run with write=True to continue from a checkpoint."
                )
            pf = prev_perturb
        else:
            st, ib, prev_perturb = None, None, None
            pf = None

        pf = None

        # Now we can purge init_box further.
        try:
            init_box.prepare_for_halos(flag_options=flag_options, force=always_purge)
        except OSError:
            pass

        # we explicitly pass the descendant halos here since we have a redshift list prior
        #   this will generate the extra fields if STOC_MINIMUM_Z is given
        pt_halos = []
        if flag_options.USE_HALO_FIELD and not flag_options.FIXED_HALO_GRIDS:
            halos_desc = None
            for iz, z in enumerate(scrollz[::-1]):
                halo_field = sf.determine_halo_list(
                    redshift=z,
                    halos_desc=halos_desc,
                    **kw,
                )
                halos_desc = halo_field
                pt_halos += [
                    sf.perturb_halo_list(redshift=z, halo_field=halo_field, **kw)
                ]

                # we never want to store every halofield
                try:
                    pt_halos[iz].purge(force=always_purge)
                except OSError:
                    pass

            # reverse the halo lists to be in line with the redshift lists
            pt_halos = pt_halos[::-1]

        # Now that we've got all the perturb fields, we can purge init more.
        try:
            init_box.prepare_for_spin_temp(
                flag_options=flag_options, force=always_purge
            )
        except OSError:
            pass

        ph = None

        perturb_files = []
        spin_temp_files = []
        ionize_files = []
        brightness_files = []
        hbox_files = []
        pth_files = []
        log10_mturnovers = np.zeros(len(scrollz))
        log10_mturnovers_mini = np.zeros(len(scrollz))
        hboxes = []
        z_halos = []
        coeval = None
        prev_coeval = None
        st2 = None
        hbox2 = None
        hbox = None

        if lightcone_filename and not Path(lightcone_filename).exists():
            lightcone.save(lightcone_filename)

        for iz, z in enumerate(scrollz):
            logger.info(f"Computing Redshift {z} ({iz + 1}/{len(scrollz)}) iterations.")

            # Best to get a perturb for this redshift, to pass to brightness_temperature
            pf2 = perturb[iz]
            # This ensures that all the arrays that are required for spin_temp are there,
            # in case we dumped them from memory into file.
            pf2.load_all()
            if flag_options.USE_HALO_FIELD:
                if not flag_options.FIXED_HALO_GRIDS:
                    ph = pt_halos[iz]
                    ph.load_all()

                hbox2 = sf.halo_box(
                    redshift=z,
                    pt_halos=ph,
                    previous_ionize_box=ib,
                    previous_spin_temp=st,
                    perturbed_field=pf2,
                    **kw,
                )

                if flag_options.USE_TS_FLUCT:
                    z_halos.append(z)
                    hboxes.append(hbox2)
                    xray_source_box = sf.xray_source(
                        redshift=z,
                        z_halos=z_halos,
                        hboxes=hboxes,
                        **kw,
                    )

            if flag_options.USE_TS_FLUCT:
                st2 = sf.spin_temperature(
                    redshift=z,
                    previous_spin_temp=st,
                    perturbed_field=perturb_min if use_interp_perturb_field else pf2,
                    xray_source_box=(
                        xray_source_box if flag_options.USE_HALO_FIELD else None
                    ),
                    cleanup=(cleanup and iz == (len(scrollz) - 1)),
                    **kw,
                )

            ib2 = sf.ionize_box(
                redshift=z,
                previous_ionize_box=ib,
                perturbed_field=pf2,
                previous_perturbed_field=prev_perturb,
                spin_temp=st2,
                halobox=hbox2,
                cleanup=(cleanup and iz == (len(scrollz) - 1)),
                **kw,
            )
            log10_mturnovers[iz] = ib2.log10_Mturnover_ave
            log10_mturnovers_mini[iz] = ib2.log10_Mturnover_MINI_ave

            bt2 = sf.brightness_temperature(
                ionized_box=ib2,
                perturbed_field=pf2,
                spin_temp=st2,
                **iokw,
            )

            coeval = Coeval(
                redshift=z,
                initial_conditions=init_box,
                perturbed_field=pf2,
                ionized_box=ib2,
                brightness_temp=bt2,
                ts_box=st2,
                halobox=hbox2,
                photon_nonconservation_data=photon_nonconservation_data,
                _globals=None,
            )

            if coeval_callback is not None and compute_coeval_callback[iz]:
                try:
                    coeval_callback_output.append(coeval_callback(coeval))
                except Exception as e:
                    if sum(compute_coeval_callback[: iz + 1]) == 1:
                        raise RuntimeError(
                            f"coeval_callback computation failed on first trial, z={z}."
                        )
                    else:
                        logger.warning(
                            f"coeval_callback computation failed on z={z}, skipping. {type(e).__name__}: {e}"
                        )

            perturb_files.append((z, os.path.join(direc, pf2.filename)))
            if flag_options.USE_HALO_FIELD and not flag_options.FIXED_HALO_GRIDS:
                hbox_files.append((z, os.path.join(direc, hbox2.filename)))
                pth_files.append((z, os.path.join(direc, ph.filename)))
            if flag_options.USE_TS_FLUCT:
                spin_temp_files.append((z, os.path.join(direc, st2.filename)))
            ionize_files.append((z, os.path.join(direc, ib2.filename)))
            brightness_files.append((z, os.path.join(direc, bt2.filename)))

            # Save mean/global quantities
            for quantity in global_quantities:
                lightcone.global_quantities[quantity][iz] = np.mean(
                    getattr(coeval, quantity)
                )

            # Get lightcone slices
            if prev_coeval is not None:
                for quantity, idx, this_lc in lightconer.make_lightcone_slices(
                    coeval, prev_coeval
                ):
                    if this_lc is not None:
                        lightcone.lightcones[quantity][..., idx] = this_lc
                        lc_index = idx

                if lightcone_filename:
                    lightcone.make_checkpoint(
                        lightcone_filename, redshift=z, index=lc_index
                    )

            # Save current ones as old ones.
            if flag_options.USE_TS_FLUCT:
                st = st2
            ib = ib2
            if flag_options.USE_MINI_HALOS:
                prev_perturb = pf2
            prev_coeval = coeval

            if pf is not None:
                try:
                    pf.purge(force=always_purge)
                except OSError:
                    pass

            if ph is not None:
                try:
                    ph.purge(force=always_purge)
                except OSError:
                    pass

            # we only need the SFR fields at previous redshifts for XraySourceBox
            if hbox is not None:
                try:
                    hbox.prepare(
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

            pf = pf2
            hbox = hbox2

            if z <= return_at_z:
                # Optionally return when the lightcone is only partially filled
                break

        if flag_options.PHOTON_CONS_TYPE == 1:
            photon_nonconservation_data = _get_photon_nonconservation_data()

        if lib.photon_cons_allocated:
            lib.FreePhotonConsMemory()

        if isinstance(lightcone, AngularLightcone) and lightconer.get_los_velocity:
            lightcone.compute_rsds(
                fname=lightcone_filename, n_subcells=astro_params.N_RSD_STEPS
            )

        # Append some info to the lightcone before we return
        lightcone.photon_nonconservation_data = photon_nonconservation_data
        lightcone.cache_files = {
            "init": [(0, os.path.join(direc, init_box.filename))],
            "perturb_field": perturb_files,
            "ionized_box": ionize_files,
            "brightness_temp": brightness_files,
            "spin_temp": spin_temp_files,
            "halobox": hbox_files,
            "pt_halos": pth_files,
        }

        lightcone.log10_mturnovers = log10_mturnovers
        lightcone.log10_mturnovers_mini = log10_mturnovers_mini

        if coeval_callback is None:
            return lightcone
        else:
            return lightcone, coeval_callback_output
