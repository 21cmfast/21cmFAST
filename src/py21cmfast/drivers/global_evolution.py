"""Module containing a driver function for computing the global evolution of the fields in the simulation."""

import warnings
from pathlib import Path
from typing import Self

import attrs
import h5py
import numpy as np

from .. import __version__
from ..c_21cmfast import lib
from ..io import h5
from ..io.caching import CacheConfig
from ..wrapper.arrays import Array
from ..wrapper.cfuncs import evaluate_Nion_z
from ..wrapper.inputs import InputParameters
from ..wrapper.outputs import (
    BrightnessTemp,
    HaloBox,
    IonizedBox,
    TsBox,
)
from .coeval import _redshift_loop_generator, _setup_ics_and_pfs_for_scrolling


def compute_global_reionization_at_z(
    redshift: float,
    inputs: InputParameters,
    previous_ionized_box: IonizedBox,
    spin_temp: TsBox | None,
) -> IonizedBox:
    r"""
    Compute an ionized box at a given redshift, according to the global evolution.

    Parameters
    ----------
    inputs : :class:`~InputParameters`
        The input parameters specifying the run. Since this may be the first box
        to use the astro params/flags, it is needed when we have not computed a TsBox or HaloBox.
    previous_ionize_box: :class:`IonizedBox`
        An ionized box at higher redshift.
    spin_temp: :class:`TsBox` or None, optional
        A spin-temperature box, only required if `USE_TS_FLUCT` is True. This is useful since
        the volume filling factor Q_HI is already contained in this object. If None, it will be
        evaluated by calling the C code.

    Returns
    -------
    :class:`~IonizedBox` :
        An object containing the ionized box data.
    """
    box = IonizedBox.new(inputs=inputs, redshift=redshift)
    shape = (1, 1, 1)

    if spin_temp is None:
        # We compute Q_HI very similarly as in SpinTemperatureBox.c.
        nion, _ = evaluate_Nion_z(
            inputs=inputs,
            redshifts=np.asarray(redshift),
            # I just put here an arbitrary number because we currently don't allow to have mini-halos when USE_TS_FLUCT=False.
            # TODO: should this limitation be relaxed in the future?
            log10mturns=np.asarray(5.0),
        )
        if inputs.matter_options.SOURCE_MODEL == "CONST-ION-EFF":
            ion_eff_factor = inputs.astro_params.HII_EFF_FACTOR
        else:
            ion_eff_factor = (
                pow(10.0, inputs.astro_params.F_STAR10)
                * pow(10.0, inputs.astro_params.F_ESC10)
                * inputs.astro_params.POP2_ION
            )
        Q_HI = 1.0 - ion_eff_factor * nion
    else:
        Q_HI = spin_temp.Q_HI

    # TODO: I think a more accurate global Q_HI can be achieved by solving an ODE that includes also the recombination rate
    Q_HI = Q_HI if Q_HI > 0.0 else 0.0

    # A crude way to estimate the global photoionization rate
    try:
        dQdz = (Q_HI - previous_ionized_box.neutral_fraction.value) / (
            redshift - previous_ionized_box.redshift
        )
    except TypeError:
        dQdz = 0.0
    dzdt = -(1.0 + redshift) * inputs.cosmo_params.cosmo.H(redshift)
    ionisation_rate_G12 = np.abs(dQdz * dzdt)
    # TODO: is there a more clever way to estimate global z_reion?
    z_reion = -1.0 if Q_HI > 0.0 else redshift

    # Now initialize the output box with the global values!
    required_arrays = {
        "neutral_fraction": Q_HI,
        "ionisation_rate_G12": ionisation_rate_G12.to("1/s"),
        "z_reion": z_reion,
    }
    for name, val in required_arrays.items():
        setattr(
            box,
            name,
            Array(shape=shape, dtype=np.float32)
            .initialize()
            .with_value(val=val * np.ones(shape)),
        )
    return box


@attrs.define()
class GlobalEvolution:
    """Structure that contains the global evolution (as a function of redshift) of all the fields in the simulation.

    Attributes
    ----------
    inputs: InputParameters
        The input parameters for the simulation.
    quantities: dict[str, np.ndarray] | None
        Arrays of length `node_redshifts` containing the global field across redshift.
    """

    inputs: InputParameters = attrs.field(
        validator=attrs.validators.instance_of(InputParameters)
    )
    quantities: dict[str, np.ndarray] | None = attrs.field(default=None)

    @classmethod
    def get_fields(cls, inputs: InputParameters) -> tuple:
        """Get a list of the names of the available fields in the simulation."""
        possible_outputs = [
            BrightnessTemp.new(inputs, redshift=0),
        ]
        if inputs.astro_options.USE_TS_FLUCT:
            possible_outputs.append(TsBox.new(inputs, redshift=0))
        if inputs.matter_options.lagrangian_source_grid:
            possible_outputs.append(HaloBox.new(inputs, redshift=0))
        field_names = ("neutral_fraction", "ionisation_rate_G12")
        for output in possible_outputs:
            field_names += tuple(output.arrays.keys())
        return field_names

    @property
    def simulation_options(self):
        """Matter params shared by all datasets."""
        return self.inputs.simulation_options

    @property
    def matter_options(self):
        """Matter flags shared by all datasets."""
        return self.inputs.matter_options

    @property
    def cosmo_params(self):
        """Cosmo params shared by all datasets."""
        return self.inputs.cosmo_params

    @property
    def astro_options(self):
        """Flag Options shared by all datasets."""
        return self.inputs.astro_options

    @property
    def astro_params(self):
        """Astro params shared by all datasets."""
        return self.inputs.astro_params

    @property
    def cosmo_tables(self):
        """Cosmo tables shared by all datasets."""
        return self.inputs.cosmo_tables

    @property
    def random_seed(self):
        """Random seed shared by all datasets."""
        return self.inputs.random_seed

    @property
    def node_redshifts(self):
        """Redshifts at which coeval boxes and global quantities are computed."""
        return self.inputs.node_redshifts

    def save(
        self,
        path: str | Path,
        clobber=False,
    ):
        """Save the global_evolution object to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_mode = "w" if clobber else "a"
        with h5py.File(path, file_mode) as fl:
            fl.attrs["global_evolution"] = (
                True  # marker identifying this as a global_evolution
            )

            fl.attrs["__version__"] = __version__

            global_q = fl.create_group("quantities")
            for k, v in self.quantities.items():
                global_q[k] = v

        h5._write_inputs_to_group(self.inputs, path)

    @classmethod
    def from_file(
        cls, path: str | Path, safe: bool = True, remove_buffer: bool = True
    ) -> Self:
        """Create a new instance from a saved global_evolution on disk."""
        kwargs = {}
        with h5py.File(path, "r") as fl:
            if not fl.attrs.get("global_evolution", False):
                raise ValueError(f"The file {path} is not a global_evolution file!")

            kwargs["inputs"] = h5.read_inputs(fl, safe=safe)

            glb = fl["quantities"]
            kwargs["quantities"] = {k: glb[k][...] for k in glb}

        return cls(**kwargs)

    def __eq__(self, other):
        """Determine if this is equal to another object."""
        return (
            isinstance(other, self.__class__)
            and self.inputs == other.inputs
            and self.quantities.keys() == other.quantities.keys()
        )


def run_global_evolution(
    inputs: InputParameters,
    source_model: str | None = None,
    progressbar: bool = False,
):
    r"""
    Compute the global evolution of all the fields in the simulation.

    Parameters
    ----------
    inputs: :class:`~InputParameters`
        This object specifies the input parameters for the run, including the random seed.
    source_model: str, optional
        The source model to use in the simulation. If not provided, it is taken from inputs, unless inputs.matter_options.has_discrete_halos
        is True, in which case an error is thrown. Options are:
        E-INTEGRAL : The traditional excursion-set formalism, where source properties are
            defined on the Eulerian grid after 2LPT in regions of filter scale R (see the X_FILTER options for filter shapes).
            This integrates over the CHMF using the smoothed density grids, then multiplies the result.
            by (1 + delta) to get the source properties in each cell.
        CONST-ION-EFF: Similar to E-INTEGRAL, but ionizing efficiency is constant and does not depend on the halo mass
            (see Mesinger+ 2010).
        L-INTEGRAL : Analagous to the 'ESF-L' model described in Trac+22, where source properties
            are defined on the Lagrangian (IC) grid by integrating the CHMF prior to the IGM physics
            and then mapping properties to the Eulerian grid using 2LPT.
    progressbar: bool, optional
        If True, a progress bar will be displayed throughout the simulation. Defaults to False.

    Returns
    -------
    global_evolution : :class:`~py21cmfast.GlobalEvolution`
        The object containing the global evolution of the fields in the simulation.

    Raises
    ------
    ValueError: If source_model is None and inputs.matter_options.has_discrete_halos, or if source_model is provided,
    but is not one of the said options above.

    Notes
    -----
    For convenience, the provided InputParameters object by the user is overwritten to allow this function to work smoothly
    in all cases. The InputParameters object that was used in this function can be accessed via global_evolution.inputs.
    Note for example that the actual InputParameters object that is used by this function has only one cell (HII_DIM=DIM=1)
    and no 2LPT is performed (since we have only one cell).
    In addition, note that the classic reionization module of 21cmFAST (where the excursion-set algorithm is applied to find ionized
    bubbles) is not called by this function, since there are no bubbles in a one cell box! Instead, we compute the global reionization
    history based on the global volume filling factor Q. For that reason, note that the correction due to photon non-conservation in the
    excursion-set algorithm is never applied when this function is called.

    """
    # When doing glboal evolution, we allow only source models free of discrete halos
    possible_sources = ["CONST-ION-EFF", "E-INTEGRAL", "L-INTEGRAL"]

    if source_model is None:
        if inputs.matter_options.has_discrete_halos:
            raise ValueError(
                "You did not specify 'source_model', but SOURCE_MODEL in `inputs` has discrete halos! "
                "Either specify 'source_model' or change SOURCE_MODEL in `inputs` to a model with no discrete halos."
            )
        else:
            source_model = inputs.matter_options.SOURCE_MODEL

    if source_model not in possible_sources:
        raise ValueError(
            f"'source_model' must be one of {possible_sources}, got {source_model} instead."
        )

    if not inputs.astro_options.USE_TS_FLUCT:
        warnings.warn(
            "Your inputs.astro_options.USE_TS_FLUCT = False. "
            "While this is the default settings (to reduce computation time when full coeval boxes are evaluated), "
            "it yields the incorrect 21-cm signal at high redshifts, before the spin temperature saturates the "
            "background photon temperature. Consider changing USE_TS_FLUCT to True in order to get the correct "
            "global 21-cm signal at all redshifts.",
            stacklevel=2,
        )

    new_input_kwargs = {
        "DIM": 1,  # we need only one cell
        "HII_DIM": 1,  # we need only one cell
        "BOX_LEN": 1e6,  # we need a huge box/cell in order to simulate the global evolution
        "SOURCE_MODEL": source_model,
        "PERTURB_ALGORITHM": "LINEAR",  # no need to do 2LPT
        "USE_INTERPOLATION_TABLES": "sigma-interpolation",  # only need sigma interpolation tables (hmf integrals are evaluated once per snapshot, without interpolation)
        "INTEGRATION_METHOD_ATOMIC": "GSL-QAG",  # due to above, we ought to use gsl, and not gauss-legendre (BUG?)
        "INTEGRATION_METHOD_MINI": "GSL-QAG",
        "USE_UPPER_STELLAR_TURNOVER": False,  # no upper stellar turnover without discrete halos
        "USE_EXP_FILTER": False,  # we don't run reionization module, so we can leave this parameter on False for all source models
        "KEEP_3D_VELOCITIES": False,  # we don't need any velocities
        "PHOTON_CONS_TYPE": "no-photoncons",  # we don't do photon conservation
    }
    inputs_one_cell = inputs.evolve_input_structs(**new_input_kwargs)

    global_evolution = GlobalEvolution(
        inputs=inputs_one_cell,
        quantities={},
    )
    for quantity in global_evolution.get_fields(inputs_one_cell):
        global_evolution.quantities[quantity] = np.zeros(
            len(inputs_one_cell.node_redshifts)
        )

    prev_coeval = None

    iokw = {"cache": None, "regenerate": True, "free_cosmo_tables": False}

    (
        initial_conditions,
        perturbed_fields,
        halofield_list,
        photon_nonconservation_data,
    ) = _setup_ics_and_pfs_for_scrolling(
        all_redshifts=inputs_one_cell.node_redshifts,
        inputs=inputs_one_cell,
        initial_conditions=None,
        write=CacheConfig.off(),
        progressbar=progressbar,
        **iokw,
    )

    for iz, coeval in _redshift_loop_generator(
        inputs=inputs_one_cell,
        initial_conditions=initial_conditions,
        all_redshifts=inputs_one_cell.node_redshifts,
        perturbed_field=perturbed_fields,
        halofield_list=halofield_list,
        write=CacheConfig.off(),
        cleanup=True,
        progressbar=progressbar,
        photon_nonconservation_data=photon_nonconservation_data,
        init_coeval=prev_coeval,
        iokw=iokw,
    ):
        for quantity in global_evolution.quantities:
            global_evolution.quantities[quantity][iz] = np.mean(
                getattr(coeval, quantity)
            )

        prev_coeval = coeval

    lib.Free_cosmo_tables_global()

    return global_evolution
