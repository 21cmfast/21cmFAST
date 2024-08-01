"""
Compute single physical fields.

These functions are high-level wrappers around C-functions that compute 3D fields, for
example initial conditions, perturbed fields and ionization fields.
"""

import contextlib
import logging
import numpy as np
import warnings
from astropy import units as un
from astropy.cosmology import z_at_value
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from ..wrapper import (
    _check_compatible_inputs,
    _get_config_options,
    _setup_inputs,
    _verify_types,
    construct_fftw_wisdoms,
)
from ..wrapper.cfuncs import get_halo_list_buffer_size
from ..wrapper.inputs import (
    AstroParams,
    CosmoParams,
    FlagOptions,
    UserParams,
    global_params,
)
from ..wrapper.outputs import (
    BrightnessTemp,
    HaloBox,
    HaloField,
    InitialConditions,
    IonizedBox,
    PerturbedField,
    PerturbHaloField,
    TsBox,
    XraySourceBox,
)

logger = logging.getLogger(__name__)


def set_globals(func: callable):
    """Decorator that sets global parameters."""

    @wraps(func)
    def inner(*args, **kwargs):
        # Get all kwargs that are actually global params
        global_kwargs = {k: v for k, v in kwargs.items() if k in global_params.keys()}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in global_kwargs}
        with global_params.use(**global_kwargs):
            return func(*args, **other_kwargs)

    return inner


@set_globals
def initial_conditions(
    *,
    user_params: UserParams | dict | None = None,
    cosmo_params: CosmoParams | dict | None = None,
    random_seed: int | None = None,
    regenerate: bool | None = None,
    write: bool | None = None,
    direc: Path | None = None,
    hooks: dict[Callable, dict[str, Any]] | None = None,
    **global_kwargs,
) -> InitialConditions:
    r"""
    Compute initial conditions.

    Parameters
    ----------
    user_params : :class:`~UserParams` instance, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams` instance, optional
        Defines the cosmological parameters used to compute initial conditions.
    random_seed : int, optional
        The random seed used to generate the phases of the initial conditions.
    regenerate : bool, optional
        Whether to force regeneration of data, even if matching cached data is found.
        This is applied recursively to any potential sub-calculations. It is ignored in
        the case of dependent data only if that data is explicitly passed to the function.
    write : bool, optional
        Whether to write results to file (i.e. cache). This is recursively applied to
        any potential sub-calculations.
    hooks
        Any extra functions to apply to the output object. This should be a dictionary
        where the keys are the functions, and the values are themselves dictionaries of
        parameters to pass to the function. The function signature should be
        ``(output, **params)``, where the ``output`` is the output object.
    direc : str, optional
        The directory in which to search for the boxes and write them. By default, this
        is the directory given by ``boxdir`` in the configuration file,
        ``~/.21cmfast/config.yml``. This is recursively applied to any potential
        sub-calculations.

    Other Parameters
    ----------------
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~InitialConditions`
    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    user_params, cosmo_params = _setup_inputs(
        {"user_params": user_params, "cosmo_params": cosmo_params}
    )

    # Initialize memory for the boxes that will be returned.
    ics = InitialConditions(
        user_params=user_params, cosmo_params=cosmo_params, random_seed=random_seed
    )

    # Construct FFTW wisdoms. Only if required
    construct_fftw_wisdoms(user_params=user_params, cosmo_params=cosmo_params)

    # First check whether the boxes already exist.
    if not regenerate:
        with contextlib.suppress(OSError):
            ics.read(direc)
            logger.info(
                f"Existing init_boxes found and read in (seed={ics.random_seed})."
            )
            return ics
    return ics.compute(hooks=hooks)


@set_globals
def perturb_field(
    *,
    redshift: float,
    initial_conditions: InitialConditions,
    regenerate: bool | None = None,
    write: bool | None = None,
    direc: Path | None = None,
    hooks: dict[Callable, dict[str, Any]] | None = None,
    **global_kwargs,
) -> PerturbedField:
    r"""
    Compute a perturbed field at a given redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to compute the perturbed field.
    initial_conditions : :class:`~InitialConditions`
        The initial conditions.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~PerturbedField`

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed:
        See docs of :func:`initial_conditions` for more information.

    Examples
    --------
    >>> init_boxes = initial_conditions()
    >>> field7 = perturb_field(7.0, init_boxes)
    >>> field8 = perturb_field(8.0, init_boxes)

    The user and cosmo parameter structures are by default inferred from the
    ``initial_conditions``.
    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    random_seed = initial_conditions.random_seed
    user_params = initial_conditions.user_params
    cosmo_params = initial_conditions.cosmo_params

    # Initialize perturbed boxes.
    fields = PerturbedField(
        redshift=redshift,
        user_params=user_params,
        cosmo_params=cosmo_params,
        random_seed=random_seed,
    )

    # Check whether the boxes already exist
    if not regenerate:
        with contextlib.suppress(OSError):
            fields.read(direc)
            logger.info(
                f"Existing z={redshift} perturb_field boxes found and read in "
                f"(seed={fields.random_seed})."
            )
            return fields

    # Construct FFTW wisdoms. Only if required
    construct_fftw_wisdoms(user_params=user_params, cosmo_params=cosmo_params)

    # Run the C Code
    return fields.compute(ics=initial_conditions, hooks=hooks)


@set_globals
def determine_halo_list(
    *,
    redshift: float,
    initial_conditions: InitialConditions,
    descendant_halos: HaloField | None = None,
    desc_z=None,
    astro_params=None,
    flag_options=None,
    regenerate=None,
    write=None,
    direc=None,
    hooks=None,
    **global_kwargs,
):
    r"""
    Find a halo list, given a redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to determine the halo list.
    initial_conditions : :class:`~InitialConditions`
        The initial conditions fields (density, velocity).
    descendant_halos : :class:`~HaloField` instance, optional
        The halos that form the descendants (i.e. lower redshift) of those computed by
        this function. If this is not provided, we generate the initial stochastic halos
        directly in this function (and progenitors can then be determined by these).
    astro_params: :class:`~AstroParams` instance, optional
        The astrophysical parameters defining the course of reionization.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~HaloField`

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed:
        See docs of :func:`initial_conditions` for more information.

    Examples
    --------
    Fill this in once finalised

    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    # Configure and check input/output parameters/structs
    user_params = initial_conditions.user_params
    cosmo_params = initial_conditions.cosmo_params
    random_seed = initial_conditions.random_seed

    (
        astro_params,
        flag_options,
        redshift,
    ) = _setup_inputs(
        {
            "astro_params": astro_params,
            "flag_options": flag_options,
        },
        {"init_boxes": initial_conditions, "prev_halo_field": descendant_halos},
        redshift=redshift,
    )

    if user_params.HMF != 1:
        warnings.warn(
            "DexM Halofinder sses a fit to the Sheth-Tormen mass function."
            "With HMF!=1 the Halos from DexM will not be from the same mass function",
        )

    hbuffer_size = get_halo_list_buffer_size(redshift, user_params, cosmo_params)

    # Initialize halo list boxes.
    fields = HaloField(
        desc_redshift=desc_z,
        buffer_size=hbuffer_size,
        redshift=redshift,
        user_params=user_params,
        cosmo_params=cosmo_params,
        astro_params=astro_params,
        flag_options=flag_options,
        random_seed=random_seed,
    )
    # Construct FFTW wisdoms. Only if required
    construct_fftw_wisdoms(user_params=user_params, cosmo_params=cosmo_params)

    if not regenerate:
        with contextlib.suppress(OSError):
            fields.read(direc)
            logger.info(
                f"Existing z={redshift} determine_halo_list boxes found and read in "
                f"(seed={fields.random_seed})."
            )
            return fields

    if descendant_halos is None:
        descendant_halos = HaloField(
            redshift=0,
            user_params=user_params,
            cosmo_params=cosmo_params,
            astro_params=astro_params,
            flag_options=flag_options,
            dummy=True,
        )

    # Run the C Code
    return fields.compute(
        ics=initial_conditions,
        hooks=hooks,
        descendant_halos=descendant_halos,
        random_seed=random_seed,
    )


@set_globals
def perturb_halo_list(
    *,
    redshift: float,
    initial_conditions: InitialConditions,
    halo_field: HaloField,
    regenerate=None,
    write=None,
    direc=None,
    hooks=None,
    **global_kwargs,
):
    r"""
    Given a halo list, perturb the halos for a given redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to determine the halo list.
    init_boxes : :class:`~InitialConditions`, optional
        If given, these initial conditions boxes will be used, otherwise initial conditions will
        be generated. If given,
        the user and cosmo params will be set from this object.
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    astro_params: :class:`~AstroParams` instance, optional
        The astrophysical parameters defining the course of reionization.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~XraySourceBox`

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed:
        See docs of :func:`initial_conditions` for more information.

    Examples
    --------
    Fill this in once finalised

    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    hbuffer_size = halo_field.n_halos

    # Initialize halo list boxes.
    fields = PerturbHaloField(
        redshift=redshift,
        buffer_size=hbuffer_size,
        user_params=halo_field.user_params,
        cosmo_params=halo_field.cosmo_params,
        astro_params=halo_field.astro_params,
        flag_options=halo_field.flag_options,
        random_seed=halo_field.random_seed,
    )

    # Check whether the boxes already exist
    if not regenerate:
        with contextlib.suppress(OSError):
            fields.read(direc)
            logger.info(
                f"Existing z={redshift} perturb_halo_list boxes found and read in "
                f"(seed={fields.random_seed})."
            )
            return fields

    # Run the C Code
    return fields.compute(ics=initial_conditions, halo_field=halo_field, hooks=hooks)


@set_globals
def compute_halo_grid(
    *,
    redshift: float,
    initial_conditions: InitialConditions,
    perturbed_halo_list: PerturbHaloField | None = None,
    perturbed_field: PerturbedField | None = None,
    previous_spin_temp: TsBox | None = None,
    previous_ionize_box: IonizedBox | None = None,
    write=None,
    direc=None,
    regenerate: bool | None = None,
    hooks=None,
    **global_kwargs,
) -> HaloBox:
    r"""
    Compute grids of halo properties from a catalogue.

    At the moment this simply produces halo masses, stellar masses and SFR on a grid of
    HII_DIM. In the future this will compute properties such as emissivities which will
    be passed directly into ionize_box etc. instead of the catalogue.

    Parameters
    ----------
    perturbed_halo_list: :class:`~PerturbHaloField` or None, optional
        This contains all the dark matter haloes obtained if using the USE_HALO_FIELD.
        This is a list of halo masses and coords for the dark matter haloes.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~HaloBox` :
        An object containing the halo box data.

    Other Parameters
    ----------------
    regenerate, write, direc :
        See docs of :func:`initial_conditions` for more information.

    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    random_seed = previous_spin_temp.random_seed
    user_params = initial_conditions.user_params
    cosmo_params = initial_conditions.cosmo_params
    astro_params = previous_spin_temp.astro_params
    flag_options = initial_conditions.flag_options

    # Initialize halo list boxes.
    box = HaloBox(
        redshift=redshift,
        random_seed=random_seed,
        user_params=user_params,
        cosmo_params=cosmo_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )

    # Check whether the boxes already exist
    if not regenerate:
        with contextlib.suppress(OSError):
            box.read(direc)
            logger.info(
                f"Existing z={redshift} halo_box boxes found and read in "
                f"(seed={box.random_seed})."
            )
            return box

    if perturbed_field is None and perturbed_halo_list is None:
        raise ValueError(
            "Either perturbed_field or perturbed_halo_list are required (or both)."
        )
    elif perturbed_field is None:
        if flag_options.FIXED_HALO_GRIDS or user_params.AVG_BELOW_SAMPLER:
            raise ValueError(
                "You must provide the perturbed field if FIXED_HALO_GRIDS is True or AVG_BELOW_SAMPLER is True"
            )
        else:
            perturbed_field = PerturbedField(
                redshift=0,
                user_params=user_params,
                cosmo_params=cosmo_params,
                dummy=True,
            )
    elif perturbed_halo_list is None:
        if not flag_options.FIXED_HALO_GRIDS:
            raise ValueError(
                "You must provide the perturbed halo list if FIXED_HALO_GRIDS is False"
            )
        else:
            perturbed_halo_list = PerturbHaloField(
                redshift=0,
                user_params=user_params,
                cosmo_params=cosmo_params,
                astro_params=astro_params,
                flag_options=flag_options,
                dummy=True,
            )

    # NOTE: due to the order, we use the previous spin temp here, like spin_temperature,
    #       but UNLIKE ionize_box, which uses the current box
    # TODO: think about the inconsistency here
    # NOTE: if USE_MINI_HALOS is TRUE, so is USE_TS_FLUCT and INHOMO_RECO
    if previous_spin_temp is None:
        if redshift >= global_params.Z_HEAT_MAX or not flag_options.USE_MINI_HALOS:
            # Dummy spin temp is OK since we're above Z_HEAT_MAX
            previous_spin_temp = TsBox(
                redshift=0,
                user_params=user_params,
                cosmo_params=cosmo_params,
                astro_params=astro_params,
                flag_options=flag_options,
                dummy=True,
            )
        else:
            raise ValueError("Below Z_HEAT_MAX you must specify the previous_spin_temp")

    if previous_ionize_box is None:
        if redshift >= global_params.Z_HEAT_MAX or not flag_options.USE_MINI_HALOS:
            # Dummy ionize box is OK since we're above Z_HEAT_MAX
            previous_ionize_box = IonizedBox(
                redshift=0,
                user_params=user_params,
                cosmo_params=cosmo_params,
                astro_params=astro_params,
                flag_options=flag_options,
                dummy=True,
            )
        else:
            raise ValueError(
                "Below Z_HEAT_MAX you must specify the previous_ionize_box"
            )

    return box.compute(
        init_boxes=initial_conditions,
        pt_halos=perturbed_halo_list,
        perturbed_field=perturbed_field,
        previous_ionize_box=previous_ionize_box,
        previous_spin_temp=previous_spin_temp,
        hooks=hooks,
    )


# TODO: make this more general and probably combine with the lightcone interp function
def interp_halo_boxes(
    halo_boxes: list[HaloBox], fields: list[str], redshift: float
) -> HaloBox:
    """
    Interpolate HaloBox history to the desired redshift.

    Photon conservation & Xray sources require halo boxes at redshifts
    that are not equal to the current redshift, and may be between redshift steps.
    So we need a function to interpolate between two halo boxes.
    We assume here that z_arr is strictly INCERASING

    Parameters
    ----------
    halo_boxes : list of HaloBox instances
        The halobox history to be interpolated
    fields: List of Strings
        The properties of the haloboxes to be interpolated
    redshift : float
        The desired redshift of interpolation

    Returns
    -------
    :class:`~HaloBox` :
        An object containing the halo box data
    """
    z_halos = [box.redshift for box in halo_boxes]
    if not np.all(np.diff(z_halos) > 0):
        raise ValueError("halo_boxes must be in ascending order of redshift")

    if redshift > z_halos[-1] or redshift < z_halos[0]:
        raise ValueError(f"Invalid z_target {redshift} for redshift array {z_halos}")

    idx_prog = np.searchsorted(z_halos, redshift, side="left")

    if idx_prog == 0 or idx_prog == len(z_halos):
        logger.debug(f"redshift {redshift} beyond limits, {z_halos[0], z_halos[-1]}")
        raise ValueError

    z_prog = z_halos[idx_prog]
    idx_desc = idx_prog - 1

    z_desc = z_halos[idx_desc]
    interp_param = (redshift - z_desc) / (z_prog - z_desc)

    # I set the box redshift to be the stored one so it is read properly into the ionize box
    # for the xray source it doesn't matter, also since it is not _compute()'d, it won't be cached
    hbox_out = HaloBox(
        redshift=redshift,
        random_seed=halo_boxes[0].random_seed,
        user_params=halo_boxes[0].user_params,
        cosmo_params=halo_boxes[0].cosmo_params,
        astro_params=halo_boxes[0].astro_params,
        flag_options=halo_boxes[0].flag_options,
    )
    # initialise hbox memory
    hbox_out()

    # interpolate halo boxes in gridded SFR
    hbox_prog = halo_boxes[idx_prog]
    hbox_desc = halo_boxes[idx_desc]

    for field in fields:
        interp_field = (1 - interp_param) * getattr(
            hbox_desc, field
        ) + interp_param * getattr(hbox_prog, field)
        setattr(hbox_out, field, interp_field)

    logger.debug(
        f"interpolated to z={redshift} between [{z_desc},{z_prog}] ({interp_param})"
    )
    logger.debug(
        f"SFR averages desc {idx_desc}: {hbox_desc.halo_sfr.mean()} interp {hbox_out.halo_sfr.mean()} prog ({idx_prog}) {hbox_prog.halo_sfr.mean()}"
    )
    logger.debug(
        f"top-level struct min-max-mean {hbox_out.halo_sfr.min()}, {hbox_out.halo_sfr.max()}, {hbox_out.halo_sfr.mean()}"
    )

    # pass the arrays to the c struct by calling again
    hbox_out()
    # HACK: Since we don't compute, we have to mark the struct as computed
    for k, state in hbox_out._array_state.items():
        if state.initialized and k in fields:
            state.computed_in_mem = True

    return hbox_out


# NOTE: the current implementation of this box is very hacky, since I have trouble figuring out a way to _compute()
#   over multiple redshifts in a nice way using this wrapper.
# TODO: if we move some code to jax or similar I think this would be one of the first candidates (just filling out some filtered grids)
@set_globals
def compute_xray_source_field(
    *,
    redshift: float,
    initial_conditions: InitialConditions,
    z_halos=None,
    hboxes: list[HaloBox],
    write=None,
    direc=None,
    regenerate=None,
    random_seed=None,
    hooks=None,
    **global_kwargs,
) -> XraySourceBox:
    r"""
    Compute filtered grid of SFR for use in spin temperature calculation.

    This will filter over the halo history in annuli, computing the contribution to the
    SFR density

    If no halo field is passed one is calculated at the desired redshift as if it is the
    first box.

    Parameters
    ----------
    hbox_file: list of str or None, optional
        If passed, this contains the list of files which contain saved
        :class:`~HaloBox` boxes which are used to create this source field
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~XraySourceBox` :
        An object containing x ray heating, ionisation, and lyman alpha rates.

    Other Parameters
    ----------------
    regenerate, write, direc :
        See docs of :func:`initial_conditions` for more information.

    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    random_seed = initial_conditions.random_seed
    user_params = initial_conditions.user_params
    cosmo_params = initial_conditions.cosmo_params
    astro_params = hboxes[0].astro_params
    flag_options = hboxes[0].flag_options

    # Initialize halo list boxes.
    box = XraySourceBox(
        redshift=redshift,
        random_seed=random_seed,
        user_params=user_params,
        cosmo_params=cosmo_params,
        astro_params=astro_params,
        flag_options=flag_options,
    )

    # Construct FFTW wisdoms. Only if required
    construct_fftw_wisdoms(user_params=user_params, cosmo_params=cosmo_params)

    # Check whether the boxes already exist
    if not regenerate:
        with contextlib.suppress(OSError):
            box.read(direc)
            logger.info(
                f"Existing z={redshift} xray_source boxes found and read in "
                f"(seed={box.random_seed})."
            )
            return box

    # set minimum R at cell size
    l_factor = (4 * np.pi / 3.0) ** (-1 / 3)
    R_min = user_params.BOX_LEN / user_params.HII_DIM * l_factor
    z_max = min(max(z_halos), global_params.Z_HEAT_MAX)

    # now we need to find the closest halo box to the redshift of the shell
    cosmo_ap = cosmo_params.cosmo
    cmd_zp = cosmo_ap.comoving_distance(redshift)
    R_steps = np.arange(0, global_params.NUM_FILTER_STEPS_FOR_Ts)
    R_factor = (global_params.R_XLy_MAX / R_min) ** (
        R_steps / global_params.NUM_FILTER_STEPS_FOR_Ts
    )
    R_range = un.Mpc * R_min * R_factor
    cmd_edges = cmd_zp + R_range  # comoving distance edges
    # NOTE(jdavies) added the 'bounded' method since it seems there are some compatibility issues with astropy and scipy
    # where astropy gives default bounds to a function with default unbounded minimization
    zpp_edges = [
        z_at_value(cosmo_ap.comoving_distance, d, method="bounded") for d in cmd_edges
    ]
    # the `average` redshift of the shell is the average of the
    # inner and outer redshifts (following the C code)
    zpp_avg = zpp_edges - np.diff(np.insert(zpp_edges, 0, redshift)) / 2

    # call the box the initialize the memory, since I give some values before computing
    box()
    final_box_computed = False
    for i in range(global_params.NUM_FILTER_STEPS_FOR_Ts):
        R_inner = R_range[i - 1].to("Mpc").value if i > 0 else 0
        R_outer = R_range[i].to("Mpc").value

        if zpp_avg[i] >= z_max:
            logger.debug(f"ignoring Radius {i} which is above Z_HEAT_MAX")
            box.filtered_sfr[i, ...] = 0
            continue

        hbox_interp = interp_halo_boxes(
            hboxes[::-1],
            ["halo_sfr", "halo_xray", "halo_sfr_mini", "log10_Mcrit_MCG_ave"],
            zpp_avg[i],
        )

        # if we have no halos we ignore the whole shell
        if np.all(hbox_interp.halo_sfr + hbox_interp.halo_sfr_mini == 0):
            box.filtered_sfr[i] = 0
            box.filtered_sfr_mini[i] = 0
            box.filtered_xray[i] = 0
            logger.debug(f"ignoring Radius {i} due to no stars")
            continue

        # HACK: so that I can compute in the loop multiple times
        # since the array state is initialized already it shouldn't re-initialise
        for k, state in box._array_state.items():
            if state.initialized:
                state.computed_in_mem = False

        # we only want to call hooks at the end so we set a dummy hook here
        hooks_in = hooks if i == global_params.NUM_FILTER_STEPS_FOR_Ts - 1 else {}

        box = box.compute(
            halobox=hbox_interp,
            R_inner=R_inner,
            R_outer=R_outer,
            R_ct=i,
            hooks=hooks_in,
        )
        if i == global_params.NUM_FILTER_STEPS_FOR_Ts - 1:
            final_box_computed = True

    # HACK: sometimes we don't compute on the last step
    # (if the first zpp > z_max or there are no halos at max R)
    # in which case the array is not marked as computed
    if not final_box_computed:
        # we need to pass the memory to C, mark it as computed and call the hooks
        box()

        for k, state in box._array_state.items():
            if state.initialized:
                state.computed_in_mem = True

        box._call_hooks(hooks)

    return box


@set_globals
def compute_ionization_field(
    *,
    perturbed_field: PerturbedField,
    initial_conditions: InitialConditions,
    previous_perturbed_field: PerturbedField | None = None,
    previous_ionized_box: IonizedBox | None = None,
    spin_temp: TsBox | None = None,
    halobox: HaloBox | None = None,
    astro_params: AstroParams | dict | None = None,
    flag_options: FlagOptions | dict | None = None,
    regenerate=None,
    write=None,
    direc=None,
    cleanup=True,
    hooks=None,
    **global_kwargs,
) -> IonizedBox:
    r"""
    Compute an ionized box at a given redshift.

    This function has various options for how the evolution of the ionization is
    computed (if at all). See the Notes below for details.

    Parameters
    ----------
    perturbed_field : :class:`~PerturbField`, optional
        The perturbed density field.
    previous_perturbed_field : :class:`~PerturbField`, optional
        An perturbed field at higher redshift. This is only used if mini_halo is included.
    previous_ionize_box: :class:`IonizedBox` or None
        An ionized box at higher redshift. This is only used if `INHOMO_RECO` and/or `do_spin_temp`
        are true. If either of these are true, and this is not given, then it will be assumed that
        this is the "first box", i.e. that it can be populated accurately without knowing source
        statistics.
    spin_temp: :class:`TsBox` or None, optional
        A spin-temperature box, only required if `do_spin_temp` is True. If None, will try to read
        in a spin temp box at the current redshift, and failing that will try to automatically
        create one, using the previous ionized box redshift as the previous spin temperature
        redshift.
    halobox: :class:`~HaloBox` or None, optional
        If passed, this contains all the dark matter haloes obtained if using the USE_HALO_FIELD.
        These are grids of halo masses and F_ESC-weighted Stellar Mass and Star Formation Rates
        If not passed, it will try and automatically create them using the available initial conditions.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning. Typically,
        if `spin_temperature` is called directly, you will want this to be true, as if the next box
        to be calculate has different shape, errors will occur if memory is not cleaned. However,
        it can be useful to set it to False if scrolling through parameters for the same box shape.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~IonizedBox` :
        An object containing the ionized box data.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed :
        See docs of :func:`initial_conditions` for more information.

    Notes
    -----
    Typically, the ionization field at any redshift is dependent on the evolution of xHI up until
    that redshift, which necessitates providing a previous ionization field to define the current
    one. This function provides several options for doing so. First, if neither the spin
    temperature field, nor inhomogeneous recombinations (specified in flag options) are used, no
    evolution needs to be done. Otherwise, either (in order of precedence)

    1. a specific previous :class`~IonizedBox` object is provided, which will be used directly,
    2. a previous redshift is provided, for which a cached field on disk will be sought,
    3. a step factor is provided which recursively steps through redshift, calculating previous
       fields up until Z_HEAT_MAX, and returning just the final field at the current redshift, or
    4. the function is instructed to treat the current field as being an initial "high-redshift"
       field such that specific sources need not be found and evolved.

    .. note:: If a previous specific redshift is given, but no cached field is found at that
              redshift, the previous ionization field will be evaluated based on `z_step_factor`.

    Examples
    --------
    By default, no spin temperature is used, and neither are inhomogeneous recombinations,
    so that no evolution is required, thus the following will compute a coeval ionization box:

    >>> xHI = ionize_box(redshift=7.0)

    However, if either of those options are true, then a full evolution will be required:

    >>> xHI = ionize_box(redshift=7.0, flag_options=FlagOptions(INHOMO_RECO=True,USE_TS_FLUCT=True))

    This will by default evolve the field from a redshift of *at least* `Z_HEAT_MAX` (a global
    parameter), in logarithmic steps of `ZPRIME_STEP_FACTOR`. To change these:

    >>> xHI = ionize_box(redshift=7.0, zprime_step_factor=1.2, z_heat_max=15.0,
    >>>                  flag_options={"USE_TS_FLUCT":True})

    Alternatively, one can pass an exact previous redshift, which will be sought in the disk
    cache, or evaluated:

    >>> ts_box = ionize_box(redshift=7.0, previous_ionize_box=8.0, flag_options={
    >>>                     "USE_TS_FLUCT":True})

    Beware that doing this, if the previous box is not found on disk, will continue to evaluate
    prior boxes based on `ZPRIME_STEP_FACTOR`. Alternatively, one can pass a previous
    :class:`~IonizedBox`:

    >>> xHI_0 = ionize_box(redshift=8.0, flag_options={"USE_TS_FLUCT":True})
    >>> xHI = ionize_box(redshift=7.0, previous_ionize_box=xHI_0)

    Again, the first line here will implicitly use ``ZPRIME_STEP_FACTOR`` to evolve the field from
    ``Z_HEAT_MAX``. Note that in the second line, all of the input parameters are taken directly from
    `xHI_0` so that they are consistent, and we need not specify the ``flag_options``.

    As the function recursively evaluates previous redshift, the previous spin temperature fields
    will also be consistently recursively evaluated. Only the final ionized box will actually be
    returned and kept in memory, however intervening results will by default be cached on disk.
    One can also pass an explicit spin temperature object:

    >>> ts = spin_temperature(redshift=7.0)
    >>> xHI = ionize_box(redshift=7.0, spin_temp=ts)

    If automatic recursion is used, then it is done in such a way that no large boxes are kept
    around in memory for longer than they need to be (only two at a time are required).
    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    # Configure and check input/output parameters/structs
    redshift = perturbed_field.redshift
    (
        random_seed,
        user_params,
        cosmo_params,
        astro_params,
        flag_options,
    ) = _setup_inputs(
        {
            "astro_params": astro_params,
            "flag_options": flag_options,
        },
        {
            "initial_conditions": initial_conditions,
            "perturbed_field": perturbed_field,
            "previous_ionized_box": previous_ionized_box,
            "spin_temp": spin_temp,
            "halobox": halobox,
        },
    )

    if spin_temp is not None and not flag_options.USE_TS_FLUCT:
        logger.warning(
            "Changing flag_options.USE_TS_FLUCT to True since spin_temp was passed."
        )
        flag_options.USE_TS_FLUCT = True

    # Get the previous redshift
    if previous_ionized_box is not None:
        prev_z = previous_ionized_box.redshift

        # Ensure the previous ionized box has a higher redshift than this one.
        if prev_z <= redshift:
            raise ValueError(
                "Previous ionized box must have a higher redshift than that being evaluated."
            )
    elif (
        flag_options.INHOMO_RECO
        or flag_options.USE_TS_FLUCT
        and redshift >= global_params.Z_HEAT_MAX
    ):
        prev_z = 0  # signal value for first box
    else:
        raise ValueError(
            "You need to provide a previous ionized box when redshift < Z_HEAT_MAX."
        )

    box = IonizedBox(
        user_params=user_params,
        cosmo_params=cosmo_params,
        redshift=redshift,
        astro_params=astro_params,
        flag_options=flag_options,
        random_seed=random_seed,
        prev_ionize_redshift=prev_z,
    )

    # Construct FFTW wisdoms. Only if required
    construct_fftw_wisdoms(user_params=user_params, cosmo_params=cosmo_params)

    # Check whether the boxes already exist
    if not regenerate:
        with contextlib.suppress(OSError):
            box.read(direc)
            logger.info(
                f"Existing z={redshift} ionized boxes found and read in (seed={box.random_seed})."
            )
            return box

    # EVERYTHING PAST THIS POINT ONLY HAPPENS IF THE BOX DOESN'T ALREADY EXIST
    # ------------------------------------------------------------------------

    # Get appropriate previous ionization box
    if previous_ionized_box is None:
        previous_ionized_box = IonizedBox(
            redshift=0, flag_options=flag_options, initial=True
        )

    if previous_perturbed_field is None:
        # If we are beyond Z_HEAT_MAX, just make an empty box
        if prev_z == 0:
            previous_perturbed_field = PerturbedField(
                redshift=0, user_params=user_params, initial=True
            )
        else:
            raise ValueError("No previous perturbed field given, but one is required.")

    if not flag_options.USE_HALO_FIELD:
        # Construct an empty halo field to pass in to the function.
        halobox = HaloBox(
            redshift=0,
            user_params=user_params,
            cosmo_params=cosmo_params,
            astro_params=astro_params,
            flag_options=flag_options,
            dummy=True,
        )
    elif halobox is None:
        raise ValueError("No halo box given but USE_HALO_FIELD=True")

    # Set empty spin temp box if necessary.
    if not flag_options.USE_TS_FLUCT:
        spin_temp = TsBox(
            redshift=0,
            user_params=user_params,
            cosmo_params=cosmo_params,
            astro_params=astro_params,
            flag_options=flag_options,
            dummy=True,
        )
    elif spin_temp is None:
        raise ValueError("No spin temperature box given but USE_TS_FLUCT=True")

    # Run the C Code
    return box.compute(
        perturbed_field=perturbed_field,
        prev_perturbed_field=previous_perturbed_field,
        prev_ionize_box=previous_ionized_box,
        spin_temp=spin_temp,
        halobox=halobox,
        ics=initial_conditions,
        hooks=hooks,
    )


@set_globals
def spin_temperature(
    *,
    astro_params: AstroParams | dict | None = None,
    flag_options: FlagOptions | dict | None = None,
    initial_conditions: InitialConditions,
    perturbed_field: PerturbedField,
    xray_source_box: XraySourceBox | None = None,
    previous_spin_temp: TsBox | None = None,
    regenerate=None,
    write=None,
    direc=None,
    cleanup=True,
    hooks=None,
    **global_kwargs,
) -> TsBox:
    r"""
    Compute spin temperature boxes at a given redshift.

    See the notes below for how the spin temperature field is evolved through redshift.

    Parameters
    ----------
    astro_params : :class:`~AstroParams`, optional
        The astrophysical parameters defining the course of reionization.
    flag_options : :class:`~FlagOptions`, optional
        Some options passed to the reionization routine.
    initial_conditions : :class:`~InitialConditions`
        The initial conditions
    perturbed_field : :class:`~PerturbField`, optional
        If given, this field will be used, otherwise it will be generated. To be generated,
        either `init_boxes` and `redshift` must be given, or `user_params`, `cosmo_params` and
        `redshift`. By default, this will be generated at the same redshift as the spin temperature
        box. The redshift of perturb field is allowed to be different than `redshift`. If so, it
        will be interpolated to the correct redshift, which can provide a speedup compared to
        actually computing it at the desired redshift.
    previous_spin_temp : :class:`TsBox` or None
        The previous spin temperature box.
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning.
        Typically, if `spin_temperature` is called directly, you will want this to be
        true, as if the next box to be calculate has different shape, errors will occur
        if memory is not cleaned. However, it can be useful to set it to False if
        scrolling through parameters for the same box shape.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~TsBox`
        An object containing the spin temperature box data.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed :
        See docs of :func:`initial_conditions` for more information.

    Notes
    -----
    Typically, the spin temperature field at any redshift is dependent on the evolution of spin
    temperature up until that redshift, which necessitates providing a previous spin temperature
    field to define the current one. This function provides several options for doing so. Either
    (in order of precedence):

    1. a specific previous spin temperature object is provided, which will be used directly,
    2. a previous redshift is provided, for which a cached field on disk will be sought,
    3. a step factor is provided which recursively steps through redshift, calculating previous
       fields up until Z_HEAT_MAX, and returning just the final field at the current redshift, or
    4. the function is instructed to treat the current field as being an initial "high-redshift"
       field such that specific sources need not be found and evolved.

    .. note:: If a previous specific redshift is given, but no cached field is found at that
              redshift, the previous spin temperature field will be evaluated based on
              ``z_step_factor``.

    Examples
    --------
    To calculate and return a fully evolved spin temperature field at a given redshift (with
    default input parameters), simply use:

    >>> ts_box = spin_temperature(redshift=7.0)

    This will by default evolve the field from a redshift of *at least* `Z_HEAT_MAX` (a global
    parameter), in logarithmic steps of `z_step_factor`. Thus to change these:

    >>> ts_box = spin_temperature(redshift=7.0, zprime_step_factor=1.2, z_heat_max=15.0)

    Alternatively, one can pass an exact previous redshift, which will be sought in the disk
    cache, or evaluated:

    >>> ts_box = spin_temperature(redshift=7.0, previous_spin_temp=8.0)

    Beware that doing this, if the previous box is not found on disk, will continue to evaluate
    prior boxes based on the ``z_step_factor``. Alternatively, one can pass a previous spin
    temperature box:

    >>> ts_box1 = spin_temperature(redshift=8.0)
    >>> ts_box = spin_temperature(redshift=7.0, previous_spin_temp=ts_box1)

    Again, the first line here will implicitly use ``z_step_factor`` to evolve the field from
    around ``Z_HEAT_MAX``. Note that in the second line, all of the input parameters are taken
    directly from `ts_box1` so that they are consistent. Finally, one can force the function to
    evaluate the current redshift as if it was beyond ``Z_HEAT_MAX`` so that it depends only on
    itself:

    >>> ts_box = spin_temperature(redshift=7.0, zprime_step_factor=None)

    This is usually a bad idea, and will give a warning, but it is possible.
    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    # Configure and check input/output parameters/structs
    (
        random_seed,
        user_params,
        cosmo_params,
        astro_params,
        flag_options,
    ) = _setup_inputs(
        {
            "astro_params": astro_params,
            "flag_options": flag_options,
        },
        {
            "init_boxes": initial_conditions,
            "perturbed_field": perturbed_field,
            "previous_spin_temp": previous_spin_temp,
            "xray_source_box": xray_source_box,
        },
    )
    redshift = perturbed_field.redshift

    # Explicitly set this flag to True, though it shouldn't be required!
    flag_options.update(USE_TS_FLUCT=True)

    # Get the previous redshift
    if previous_spin_temp is not None:
        prev_z = previous_spin_temp.redshift
    elif redshift < global_params.Z_HEAT_MAX:
        raise ValueError(
            "previous_spin_temp is required when the redshift is lower than Z_HEAT_MAX"
        )
    else:
        # Set prev_z to anything, since we don't need it.
        prev_z = np.inf

    # Ensure the previous spin temperature has a higher redshift than this one.
    if prev_z <= redshift:
        raise ValueError(
            "Previous spin temperature box must have a higher redshift than "
            "that being evaluated."
        )

    if flag_options.USE_HALO_FIELD and xray_source_box is None:
        raise ValueError("xray_source_box is required when USE_HALO_FIELD is True")

    # Set up the box without computing anything.
    box = TsBox(
        user_params=user_params,
        cosmo_params=cosmo_params,
        redshift=redshift,
        astro_params=astro_params,
        flag_options=flag_options,
        random_seed=random_seed,
        prev_spin_redshift=prev_z,
    )

    # Construct FFTW wisdoms. Only if required
    construct_fftw_wisdoms(user_params=user_params, cosmo_params=cosmo_params)

    # Check whether the boxes already exist on disk.
    if not regenerate:
        with contextlib.suppress(OSError):
            box.read(direc)
            logger.info(
                f"Existing z={redshift} spin_temp boxes found and read in "
                f"(seed={box.random_seed})."
            )
            return box

    # Create appropriate previous_spin_temp. We've already checked that if it is None,
    # we're above the Z_HEAT_MAX.
    if previous_spin_temp is None:
        # We end up never even using this box, just need to define it
        # unallocated to be able to send into the C code.
        previous_spin_temp = TsBox(
            redshift=prev_z,  # redshift here is ignored
            user_params=initial_conditions.user_params,
            cosmo_params=initial_conditions.cosmo_params,
            astro_params=astro_params,
            flag_options=flag_options,
            dummy=True,
        )

    # Run the C Code
    return box.compute(
        cleanup=cleanup,
        perturbed_field=perturbed_field,
        xray_source_box=xray_source_box,
        prev_spin_temp=previous_spin_temp,
        ics=initial_conditions,
        hooks=hooks,
    )


@set_globals
def brightness_temperature(
    *,
    ionized_box: IonizedBox,
    perturbed_field: PerturbedField,
    spin_temp: TsBox | None = None,
    write=None,
    regenerate: bool | None = None,
    direc=None,
    hooks=None,
    **global_kwargs,
) -> BrightnessTemp:
    r"""
    Compute a coeval brightness temperature box.

    Parameters
    ----------
    ionized_box: :class:`IonizedBox`
        A pre-computed ionized box.
    perturbed_field: :class:`PerturbedField`
        A pre-computed perturbed field at the same redshift as `ionized_box`.
    spin_temp: :class:`TsBox`, optional
        A pre-computed spin temperature, at the same redshift as the other boxes.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`BrightnessTemp` instance.
    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    _verify_types(
        perturbed_field=perturbed_field,
        spin_temp=spin_temp,
        ionized_box=ionized_box,
    )

    # don't ignore redshift here
    _check_compatible_inputs(ionized_box, perturbed_field, spin_temp, ignore=[])

    if spin_temp is None:
        if ionized_box.flag_options.USE_TS_FLUCT:
            raise ValueError(
                "You have USE_TS_FLUCT=True, but have not provided a spin_temp!"
            )

        # Make an unused dummy box.
        spin_temp = TsBox(redshift=0, dummy=True)

    box = BrightnessTemp(
        user_params=ionized_box.user_params,
        cosmo_params=ionized_box.cosmo_params,
        astro_params=ionized_box.astro_params,
        flag_options=ionized_box.flag_options,
        redshift=ionized_box.redshift,
        random_seed=ionized_box.random_seed,
    )

    # Construct FFTW wisdoms. Only if required
    construct_fftw_wisdoms(
        user_params=ionized_box.user_params, cosmo_params=ionized_box.cosmo_params
    )

    # Check whether the boxes already exist on disk.
    if not regenerate:
        with contextlib.suppress(OSError):
            box.read(direc)
            logger.info(
                f"Existing brightness_temp box found and read in (seed={box.random_seed})."
            )
            return box

    return box.compute(
        spin_temp=spin_temp,
        ionized_box=ionized_box,
        perturbed_field=perturbed_field,
        hooks=hooks,
    )
