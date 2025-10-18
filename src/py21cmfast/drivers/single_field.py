"""
Compute single physical fields.

These functions are high-level wrappers around C-functions that compute 3D fields, for
example initial conditions, perturbed fields and ionization fields.
"""

import logging
import warnings

import numpy as np
from astropy import units as un
from astropy.cosmology import z_at_value

from ..wrapper.inputs import InputParameters
from ..wrapper.outputs import (
    BrightnessTemp,
    HaloBox,
    HaloCatalog,
    InitialConditions,
    IonizedBox,
    PerturbedField,
    PerturbHaloCatalog,
    TsBox,
    XraySourceBox,
)
from ._param_config import (
    check_output_consistency,
    single_field_func,
)

logger = logging.getLogger(__name__)


@single_field_func
def compute_initial_conditions(*, inputs: InputParameters) -> InitialConditions:
    r"""
    Compute initial conditions.

    Parameters
    ----------
    inputs
        The InputParameters instance defining the run.
    regenerate : bool, optional
        Whether to force regeneration of data, even if matching cached data is found.
    cache
        An OutputCache object defining how to read cached boxes.
    write
        A boolean specifying whether we need to cache the box.

    Returns
    -------
    :class:`~InitialConditions`
    """
    # Initialize memory for the boxes that will be returned.
    ics = InitialConditions.new(inputs=inputs)
    return ics.compute()


@single_field_func
def perturb_field(
    *,
    redshift: float,
    inputs: InputParameters | None = None,
    initial_conditions: InitialConditions,
) -> PerturbedField:
    r"""
    Compute a perturbed field at a given redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to compute the perturbed field.
    initial_conditions : :class:`~InitialConditions` instance
        The initial conditions.

    Returns
    -------
    :class:`~PerturbedField`

    Other Parameters
    ----------------
    regenerate, write, cache:
        See docs of :func:`initial_conditions` for more information.

    Examples
    --------
    >>> initial_conditions = compute_initial_conditions()
    >>> field7 = perturb_field(7.0, initial_conditions)
    >>> field8 = perturb_field(8.0, initial_conditions)

    The user and cosmo parameter structures are by default inferred from the
    ``initial_conditions``.
    """
    # Initialize perturbed boxes.
    fields = PerturbedField.new(redshift=redshift, inputs=inputs)

    # Run the C Code
    return fields.compute(ics=initial_conditions)


@single_field_func
def determine_halo_list(
    *,
    redshift: float,
    inputs: InputParameters | None = None,
    initial_conditions: InitialConditions,
    descendant_halos: HaloCatalog | None = None,
) -> HaloCatalog:
    r"""
    Find a halo list, given a redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to determine the halo list.
    initial_conditions : :class:`~InitialConditions` instance
        The initial conditions fields (density, velocity).
    descendant_halos : :class:`~HaloCatalog` instance, optional
        The halos that form the descendants (i.e. lower redshift) of those computed by
        this function. If this is not provided, we generate the initial stochastic halos
        directly in this function (and progenitors can then be determined by these).

    Returns
    -------
    :class:`~HaloCatalog`

    Other Parameters
    ----------------
    regenerate, write, cache:
        See docs of :func:`initial_conditions` for more information.
    """
    if inputs.matter_options.HMF != "ST":
        warnings.warn(
            "DexM Halofinder sses a fit to the Sheth-Tormen mass function."
            "With HMF!=1 the Halos from DexM will not be from the same mass function",
            stacklevel=2,
        )

    if descendant_halos is None:
        descendant_halos = HaloCatalog.dummy()

    # Initialize halo list boxes.
    fields = HaloCatalog.new(
        redshift=redshift,
        desc_redshift=descendant_halos.redshift,
        inputs=inputs,
    )

    # Run the C Code
    return fields.compute(
        ics=initial_conditions,
        descendant_halos=descendant_halos,
    )


@single_field_func
def perturb_halo_list(
    *,
    initial_conditions: InitialConditions,
    inputs: InputParameters | None = None,
    previous_spin_temp: TsBox | None = None,
    previous_ionize_box: IonizedBox | None = None,
    halo_field: HaloCatalog,
) -> PerturbHaloCatalog:
    r"""
    Given a halo list, perturb the halos for a given redshift.

    Parameters
    ----------
    initial_conditions : :class:`~InitialConditions`
        The initial conditions of the run. The user and cosmo params
        as well as the random seed will be set from this object.
    halo_field: :class: `~HaloCatalog`
        The halo catalogue in Lagrangian space to be perturbed.

    Returns
    -------
    :class:`~PerturbHaloCatalog`

    Other Parameters
    ----------------
    regenerate, write, direc:
        See docs of :func:`initial_conditions` for more information.

    Examples
    --------
    Fill this in once finalised

    """
    hbuffer_size = halo_field.n_halos if halo_field.n_halos else halo_field.buffer_size
    redshift = halo_field.redshift

    # Initialize halo list boxes.
    fields = PerturbHaloCatalog.new(
        redshift=redshift,
        buffer_size=hbuffer_size,
        inputs=inputs,
    )
    if previous_spin_temp is None:
        if (
            redshift >= inputs.simulation_options.Z_HEAT_MAX
            or not inputs.astro_options.USE_MINI_HALOS
        ):
            # Dummy spin temp is OK since we're above Z_HEAT_MAX
            previous_spin_temp = TsBox.dummy()
        else:
            raise ValueError("Below Z_HEAT_MAX you must specify the previous_spin_temp")

    if previous_ionize_box is None:
        if (
            redshift >= inputs.simulation_options.Z_HEAT_MAX
            or not inputs.astro_options.USE_MINI_HALOS
        ):
            # Dummy ionize box is OK since we're above Z_HEAT_MAX
            previous_ionize_box = IonizedBox.dummy()
        else:
            raise ValueError(
                "Below Z_HEAT_MAX you must specify the previous_ionize_box"
            )

    # Run the C Code
    return fields.compute(
        ics=initial_conditions,
        halo_field=halo_field,
        previous_spin_temp=previous_spin_temp,
        previous_ionize_box=previous_ionize_box,
    )


@single_field_func
def compute_halo_grid(
    *,
    redshift: float,
    initial_conditions: InitialConditions,
    inputs: InputParameters | None = None,
    halo_field: HaloCatalog | None = None,
    previous_spin_temp: TsBox | None = None,
    previous_ionize_box: IonizedBox | None = None,
) -> HaloBox:
    r"""
    Compute grids of halo properties from a catalogue.

    At the moment this simply produces halo masses, stellar masses and SFR on a grid of
    HII_DIM. In the future this will compute properties such as emissivities which will
    be passed directly into ionize_box etc. instead of the catalogue.

    Parameters
    ----------
    initial_conditions : :class:`~InitialConditions`
        The initial conditions of the run.
    inputs : :class:`~InputParameters`, optional
        The input parameters specifying the run.
    perturbed_halo_list: :class:`~PerturbHaloCatalog`, optional
        This contains all the dark matter haloes obtained if using the USE_HALO_FIELD.
        This is a list of halo masses and coords for the dark matter haloes.
    perturbed_field : :class:`~PerturbedField`, optional
        The perturbed density field. Used when calculating fixed source grids from CMF integrals
    previous_spin_temp : :class:`TsBox`, optional
        The previous spin temperature box. Used for feedback when USE_MINI_HALOS==True
    previous_ionize_box: :class:`IonizedBox` or None
        An at the last timestep. Used for feedback when USE_MINI_HALOS==True

    Returns
    -------
    :class:`~HaloBox` :
        An object containing the halo box data.

    Other Parameters
    ----------------
    regenerate, write, cache:
        See docs of :func:`initial_conditions` for more information.
    """
    box = HaloBox.new(redshift=redshift, inputs=inputs)

    if halo_field is None:
        if not inputs.matter_options.FIXED_HALO_GRIDS:
            raise ValueError("You must provide halo_field if FIXED_HALO_GRIDS is False")
        else:
            halo_field = HaloCatalog.dummy()

    # NOTE: due to the order, we use the previous spin temp here, like spin_temperature,
    #       but UNLIKE ionize_box, which uses the current box
    # TODO: think about the inconsistency here
    # NOTE: if USE_MINI_HALOS is TRUE, so is USE_TS_FLUCT and INHOMO_RECO
    if previous_spin_temp is None:
        if (
            redshift >= inputs.simulation_options.Z_HEAT_MAX
            or not inputs.astro_options.USE_MINI_HALOS
        ):
            # Dummy spin temp is OK since we're above Z_HEAT_MAX
            previous_spin_temp = TsBox.dummy()
        else:
            raise ValueError("Below Z_HEAT_MAX you must specify the previous_spin_temp")

    if previous_ionize_box is None:
        if (
            redshift >= inputs.simulation_options.Z_HEAT_MAX
            or not inputs.astro_options.USE_MINI_HALOS
        ):
            # Dummy ionize box is OK since we're above Z_HEAT_MAX
            previous_ionize_box = IonizedBox.dummy()
        else:
            raise ValueError(
                "Below Z_HEAT_MAX you must specify the previous_ionize_box"
            )

    return box.compute(
        initial_conditions=initial_conditions,
        halo_field=halo_field,
        previous_ionize_box=previous_ionize_box,
        previous_spin_temp=previous_spin_temp,
    )


# TODO: make this more general and probably combine with the lightcone interp function
def interp_halo_boxes(
    halo_boxes: list[HaloBox],
    fields: list[str],
    redshift: float,
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
    inputs = halo_boxes[0].inputs
    z_halos = [box.redshift for box in halo_boxes]
    if not np.all(np.diff(z_halos) > 0):
        raise ValueError("halo_boxes must be in ascending order of redshift")

    if redshift > z_halos[-1] or redshift < z_halos[0]:
        raise ValueError(f"Invalid z_target {redshift} for redshift array {z_halos}")

    arr_fields = [f for f in fields if f in halo_boxes[0].arrays]
    computed = [box.ensure_arrays_computed(*arr_fields) for box in halo_boxes]
    if not all(computed):
        raise ValueError("Some of the HaloBox fields required are not computed")

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
    check_output_consistency(
        dict(zip([f"box-{i}" for i in range(len(halo_boxes))], halo_boxes, strict=True))
    )
    hbox_out = HaloBox.new(redshift=redshift, inputs=inputs)

    # initialise the memory
    hbox_out._init_arrays()

    # interpolate halo boxes in gridded SFR
    hbox_prog = halo_boxes[idx_prog]
    hbox_desc = halo_boxes[idx_desc]

    for field in fields:
        field_desc = hbox_desc.get(field)
        field_prog = hbox_prog.get(field)
        interp_field = np.zeros_like(field_desc)
        interp_field[...] = (1 - interp_param) * hbox_desc.get(
            field
        ) + interp_param * field_prog
        hbox_out.set(field, interp_field)

    return hbox_out


# NOTE: the current implementation of this box is very hacky, since I have trouble figuring out a way to _compute()
#   over multiple redshifts in a nice way using this wrapper.
# TODO: if we move some code to jax or similar I think this would be one of the first candidates (just filling out some filtered grids)
@single_field_func
def compute_xray_source_field(
    *,
    initial_conditions: InitialConditions,
    hboxes: list[HaloBox],
    redshift: float,
) -> XraySourceBox:
    r"""
    Compute filtered grid of SFR for use in spin temperature calculation.

    This will filter over the halo history in annuli, computing the contribution to the
    SFR density

    If no halo field is passed one is calculated at the desired redshift as if it is the
    first box.

    Parameters
    ----------
    initial_conditions : :class:`~InitialConditions`
        The initial conditions of the run. The user and cosmo params
    hboxes: Sequence of :class:`~HaloBox` instances
        This contains the list of Halobox instances which are used to create this source field

    Returns
    -------
    :class:`~XraySourceBox` :
        An object containing x ray heating, ionisation, and lyman alpha rates.

    Other Parameters
    ----------------
    regenerate, write, cache:
        See docs of :func:`initial_conditions` for more information.
    """
    z_halos = [hb.redshift for hb in hboxes]
    inputs = hboxes[0].inputs

    # Initialize halo list boxes.
    box = XraySourceBox.new(redshift=redshift, inputs=inputs)

    # set minimum R at cell size
    l_factor = (4 * np.pi / 3.0) ** (-1 / 3)
    R_min = (
        inputs.simulation_options.BOX_LEN / inputs.simulation_options.HII_DIM * l_factor
    )
    z_max = min(max(z_halos), inputs.simulation_options.Z_HEAT_MAX)

    # now we need to find the closest halo box to the redshift of the shell
    cosmo_ap = inputs.cosmo_params.cosmo
    cmd_zp = cosmo_ap.comoving_distance(redshift)
    R_steps = np.arange(0, inputs.astro_params.N_STEP_TS)
    R_factor = (inputs.astro_params.R_MAX_TS / R_min) ** (
        R_steps / inputs.astro_params.N_STEP_TS
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

    interp_fields = ["halo_sfr", "halo_xray"]
    if inputs.astro_options.USE_MINI_HALOS:
        interp_fields += ["halo_sfr_mini", "log10_Mcrit_MCG_ave"]

    # call the box the initialize the memory, since I give some values before computing
    box._init_arrays()
    for i in range(inputs.astro_params.N_STEP_TS):
        R_inner = R_range[i - 1].to("Mpc").value if i > 0 else 0
        R_outer = R_range[i].to("Mpc").value

        if zpp_avg[i] >= z_max:
            box.filtered_sfr.value[i] = 0
            box.filtered_xray.value[i] = 0
            if inputs.astro_options.USE_MINI_HALOS:
                box.filtered_sfr_mini.value[i] = 0
                box.mean_log10_Mcrit_LW.value[i] = inputs.astro_params.M_TURN  # minimum
            logger.debug(f"ignoring Radius {i} which is above Z_HEAT_MAX")
            continue

        hbox_interp = interp_halo_boxes(
            halo_boxes=hboxes[::-1],
            fields=interp_fields,
            redshift=zpp_avg[i],
        )

        # if we have no halos we ignore the whole shell
        sfr_allzero = np.all(hbox_interp.get("halo_sfr") == 0)
        if inputs.astro_options.USE_MINI_HALOS:
            sfr_allzero = sfr_allzero & np.all(hbox_interp.get("halo_sfr_mini") == 0)
        if sfr_allzero:
            box.filtered_sfr.value[i] = 0
            box.filtered_xray.value[i] = 0
            if inputs.astro_options.USE_MINI_HALOS:
                box.filtered_sfr_mini.value[i] = 0
                box.mean_log10_Mcrit_LW.value[i] = hbox_interp.log10_Mcrit_MCG_ave
            logger.debug(f"ignoring Radius {i} due to no stars")
            continue

        box = box.compute(
            halobox=hbox_interp,
            R_inner=R_inner,
            R_outer=R_outer,
            R_ct=i,
            allow_already_computed=True,
        )

    # Sometimes we don't compute at all
    # (if the first zpp > z_max or there are no halos at max R)
    # in which case the array is not marked as computed
    if not box.is_computed:
        for name, array in box.arrays.items():
            setattr(box, name, array.computed())

    return box


@single_field_func
def compute_spin_temperature(
    *,
    initial_conditions: InitialConditions,
    perturbed_field: PerturbedField,
    inputs: InputParameters | None = None,
    xray_source_box: XraySourceBox | None = None,
    previous_spin_temp: TsBox | None = None,
    cleanup: bool = False,
) -> TsBox:
    r"""
    Compute spin temperature boxes at a given redshift.

    See the notes below for how the spin temperature field is evolved through redshift.

    Parameters
    ----------
    initial_conditions : :class:`~InitialConditions`
        The initial conditions
    inputs : :class:`~InputParameters`
        The input parameters specifying the run. Since this may be the first box
        to use the astro params/flags, it is needed when USE_HALO_FIELD=False.
    perturbed_field : :class:`~PerturbedField`, optional
        If given, this field will be used, otherwise it will be generated. To be generated,
        either `initial_conditions` and `redshift` must be given, or `simulation_options`, `cosmo_params` and
        `redshift`. By default, this will be generated at the same redshift as the spin temperature
        box. The redshift of perturb field is allowed to be different than `redshift`. If so, it
        will be interpolated to the correct redshift, which can provide a speedup compared to
        actually computing it at the desired redshift.
    xray_source_box : :class:`XraySourceBox`, optional
        If USE_HALO_FIELD is True, this box specifies the filtered sfr and xray emissivity at all
        redshifts/filter radii required by the spin temperature algorithm.
    previous_spin_temp : :class:`TsBox` or None
        The previous spin temperature box. Needed when we are beyond the first snapshot

    Returns
    -------
    :class:`~TsBox`
        An object containing the spin temperature box data.

    Other Parameters
    ----------------
    regenerate, write, cache:
        See docs of :func:`initial_conditions` for more information.
    """
    redshift = perturbed_field.redshift

    if redshift >= inputs.simulation_options.Z_HEAT_MAX:
        previous_spin_temp = TsBox.new(inputs=inputs, redshift=0.0, dummy=True)

    if xray_source_box is None:
        if inputs.matter_options.USE_HALO_FIELD:
            raise ValueError("xray_source_box is required when USE_HALO_FIELD is True")
        else:
            xray_source_box = XraySourceBox.dummy()

    # Set up the box without computing anything.
    box = TsBox.new(
        redshift=redshift,
        inputs=inputs,
    )

    # Run the C Code
    return box.compute(
        cleanup=cleanup,
        perturbed_field=perturbed_field,
        xray_source_box=xray_source_box,
        prev_spin_temp=previous_spin_temp,
        ics=initial_conditions,
    )


@single_field_func
def compute_ionization_field(
    *,
    perturbed_field: PerturbedField,
    initial_conditions: InitialConditions,
    inputs: InputParameters | None = None,
    previous_perturbed_field: PerturbedField | None = None,
    previous_ionized_box: IonizedBox | None = None,
    spin_temp: TsBox | None = None,
    halobox: HaloBox | None = None,
) -> IonizedBox:
    r"""
    Compute an ionized box at a given redshift.

    This function has various options for how the evolution of the ionization is
    computed (if at all). See the Notes below for details.

    Parameters
    ----------
    initial_conditions : :class:`~InitialConditions` instance
        The initial conditions.
    inputs : :class:`~InputParameters`
        The input parameters specifying the run. Since this may be the first box
        to use the astro params/flags, it is needed when USE_HALO_FIELD=False and USE_TS_FLUCT=False.
    perturbed_field : :class:`~PerturbedField`
        The perturbed density field.
    previous_perturbed_field : :class:`~PerturbedField`, optional
        An perturbed field at higher redshift. This is only used if USE_MINI_HALOS is included.
    previous_ionize_box: :class:`IonizedBox` or None
        An ionized box at higher redshift. This is only used if `INHOMO_RECO` and/or `USE_TS_FLUCT`
        are true. If either of these are true, and this is not given, then it will be assumed that
        this is the "first box", i.e. that it can be populated accurately without knowing source
        statistics.
    spin_temp: :class:`TsBox` or None, optional
        A spin-temperature box, only required if `USE_TS_FLUCT` is True. If None, will try to read
        in a spin temp box at the current redshift, and failing that will try to automatically
        create one, using the previous ionized box redshift as the previous spin temperature
        redshift.
    halobox: :class:`~HaloBox` or None, optional
        If passed, this contains all the dark matter haloes obtained if using the USE_HALO_FIELD.
        These are grids of containing summed halo properties such as ionizing emissivity.

    Returns
    -------
    :class:`~IonizedBox` :
        An object containing the ionized box data.

    Notes
    -----
    Typically, the ionization field at any redshift is dependent on the evolution of xHI up until
    that redshift, which necessitates providing a previous ionization field to define the current
    one. If neither the spin temperature field, nor inhomogeneous recombinations (specified in
    flag options) are used, no evolution needs to be done. If the redshift is beyond
    Z_HEAT_MAX, previous fields are not required either.
    """
    redshift = perturbed_field.redshift

    if redshift >= inputs.simulation_options.Z_HEAT_MAX:
        # Previous boxes must be "initial"
        previous_ionized_box = IonizedBox.initial(inputs=inputs)
        previous_perturbed_field = PerturbedField.initial(inputs=inputs)

    if inputs.evolution_required:
        if previous_ionized_box is None:
            raise ValueError(
                "You need to provide a previous ionized box when redshift < Z_HEAT_MAX."
            )
        if previous_perturbed_field is None:
            raise ValueError(
                "You need to provide a previous perturbed field when redshift < Z_HEAT_MAX."
            )
    else:
        if previous_ionized_box is None:
            previous_ionized_box = IonizedBox.initial(inputs=inputs)
        if previous_perturbed_field is None:
            previous_perturbed_field = PerturbedField.initial(inputs=inputs)

    box = IonizedBox.new(inputs=inputs, redshift=redshift)

    if not inputs.matter_options.USE_HALO_FIELD:
        # Construct an empty halo field to pass in to the function.
        halobox = HaloBox.dummy()
    elif halobox is None:
        raise ValueError("No halo box given but USE_HALO_FIELD=True")

    # Set empty spin temp box if necessary.
    if not inputs.astro_options.USE_TS_FLUCT:
        spin_temp = TsBox.dummy()
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
    )


@single_field_func
def brightness_temperature(
    *,
    ionized_box: IonizedBox,
    perturbed_field: PerturbedField,
    spin_temp: TsBox | None = None,
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

    Returns
    -------
    :class:`BrightnessTemp` instance.
    """
    redshift = ionized_box.redshift
    inputs = ionized_box.inputs

    if spin_temp is None:
        if inputs.astro_options.USE_TS_FLUCT:
            raise ValueError(
                "You have USE_TS_FLUCT=True, but have not provided a spin_temp!"
            )
        else:
            spin_temp = TsBox.dummy()

    box = BrightnessTemp.new(redshift=redshift, inputs=inputs)

    return box.compute(
        spin_temp=spin_temp,
        ionized_box=ionized_box,
        perturbed_field=perturbed_field,
    )
