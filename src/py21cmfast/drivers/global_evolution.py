"""Module containing a driver function for computing the global evolution of the fields in the simulation."""

import logging
from pathlib import Path
from typing import Self

import attrs
import h5py
import numpy as np
from astropy import units

from .. import __version__
from ..c_21cmfast import lib
from ..io import h5
from ..io.caching import CacheConfig
from ..wrapper.inputs import InputParameters
from ..wrapper.outputs import (
    BrightnessTemp,
    HaloBox,
    TsBox,
)
from .coeval import _redshift_loop_generator, _setup_ics_and_pfs_for_scrolling

logger = logging.getLogger(__name__)


@attrs.define()
class GlobalEvolution:
    """Structure that contains the global evolution (as a function of redshift) of all the fields in the simulation.

    Attributes
    ----------
    inputs: InputParameters
        The input parameters corresponding to the lightcones.
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
        field_names = ("neutral_fraction",)
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

    # TODO: Needs to modify save and load methods
    def save(
        self,
        path: str | Path,
        clobber=False,
        lowz_buffer_pixels: int = 0,
        highz_buffer_pixels: int = 0,
    ):
        """Save the lightcone object to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_mode = "w" if clobber else "a"
        with h5py.File(path, file_mode) as fl:
            fl.attrs["lightcone"] = True  # marker identifying this as a lightcone box

            fl.attrs["last_completed_node"] = self._last_completed_node
            fl.attrs["last_completed_lcidx"] = self._last_completed_lcidx
            fl.attrs["lowz_buffer_pixels"] = lowz_buffer_pixels
            fl.attrs["highz_buffer_pixels"] = highz_buffer_pixels

            fl.attrs["__version__"] = __version__

            grp = fl.create_group("photon_nonconservation_data")
            for k, v in self.photon_nonconservation_data.items():
                grp[k] = v

            # Save the boxes to the file
            boxes = fl.create_group("lightcones")
            for k, val in self.lightcones.items():
                boxes[k] = val

            global_q = fl.create_group("quantities")
            for k, v in self.quantities.items():
                global_q[k] = v

            fl["lightcone_distances"] = self.lightcone_distances.to_value("Mpc")

        h5._write_inputs_to_group(self.inputs, path)

    @classmethod
    def from_file(
        cls, path: str | Path, safe: bool = True, remove_buffer: bool = True
    ) -> Self:
        """Create a new instance from a saved lightcone on disk."""
        kwargs = {}
        with h5py.File(path, "r") as fl:
            if not fl.attrs.get("lightcone", False):
                raise ValueError(f"The file {path} is not a lightcone file!")

            kwargs["inputs"] = h5.read_inputs(fl, safe=safe)
            kwargs["last_completed_node"] = fl.attrs["last_completed_node"]
            kwargs["last_completed_lcidx"] = fl.attrs["last_completed_lcidx"]

            if remove_buffer:
                lowz_buffer_pixels = fl.attrs.get("lowz_buffer_pixels", 0)
                highz_buffer_pixels = fl.attrs.get("highz_buffer_pixels", 0)
            else:
                lowz_buffer_pixels = 0
                highz_buffer_pixels = 0

            highz_buffer_pixels = len(fl["lightcone_distances"]) - highz_buffer_pixels

            grp = fl["photon_nonconservation_data"]
            kwargs["photon_nonconservation_data"] = {k: v[...] for k, v in grp.items()}

            boxes = fl["lightcones"]
            kwargs["lightcones"] = {
                k: boxes[k][..., lowz_buffer_pixels:highz_buffer_pixels] for k in boxes
            }

            glb = fl["quantities"]
            kwargs["quantities"] = {k: glb[k][...] for k in glb}
            kwargs["lightcone_distances"] = (
                fl["lightcone_distances"][..., lowz_buffer_pixels:highz_buffer_pixels]
                * units.Mpc
            )

        return cls(**kwargs)

    def __eq__(self, other):
        """Determine if this is equal to another object."""
        return (
            isinstance(other, self.__class__)
            and self.inputs == other.inputs
            and self.quantities.keys() == other.quantities.keys()
        )


# TODO: @high_level_func ???
def run_global_evolution(
    inputs: InputParameters,
    source_model: str = "L-INTEGRAL",
    progressbar: bool = False,
):
    r"""
    Create a generator function for a lightcone run.

    This is generally the easiest and most efficient way to generate a lightcone, though it can
    be done manually by using the lower-level functions which are called by this function.

    Parameters
    ----------
    inputs: :class:`~InputParameters`
        This object specifies the input parameters for the run, including the random seed
    source_model: str, optional
        The source model to use in the simulation. Default is taken from inputs, unless inputs.matter_options.has_discrete_halos
        is True. Options are:
        CONST-ION-EFF: Similarly to E-INTEGRAL (see below), but ionizing efficiency is constant and does not depend on the halo mass
            (see Mesinger+ 2010).
        E-INTEGRAL : The traditional excursion-set formalism, where source properties are
            defined on the Eulerian grid after 2LPT in regions of filter scale R (see the X_FILTER options for filter shapes).
            This integrates over the CHMF using the smoothed density grids, then multiplies the result.
            by (1 + delta) to get the source properties in each cell.
        L-INTEGRAL : Analagous to the 'ESF-L' model described in Trac+22, where source properties
            are defined on the Lagrangian (IC) grid by integrating the CHMF prior to the IGM physics
            and then mapping properties to the Eulerian grid using 2LPT.
    progressbar: bool, optional
        If True, a progress bar will be displayed throughout the simulation. Defaults to False.

    Returns
    -------
    global_evolution : :class:`~py21cmfast.GlobalEvolution`
        The object containing the global evolution of the fields in the simulation.

    """
    new_input_kwargs = {
        "DIM": 1,
        "HII_DIM": 1,
        "BOX_LEN": 1e6,
        "SOURCE_MODEL": source_model
        if inputs.matter_options.has_discrete_halos
        else inputs.matter_options.SOURCE_MODEL,
        "PERTURB_ALGORITHM": "LINEAR",
        "USE_TS_FLUCT": True,
        "USE_INTERPOLATION_TABLES": "sigma-interpolation",
        "INTEGRATION_METHOD_ATOMIC": "GSL-QAG",
        "INTEGRATION_METHOD_MINI": "GSL-QAG",
        "PERTURB_ON_HIGH_RES": False,
        "USE_UPPER_STELLAR_TURNOVER": False,
        "USE_EXP_FILTER": False,
        "FIX_VCB_AVG": True,
        "KEEP_3D_VELOCITIES": False,
        "PHOTON_CONS_TYPE": "no-photoncons",
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
