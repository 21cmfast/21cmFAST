"""Module to deal with the cache.

The module has a manager that essentially establishes a database of cached files,
and provides methods to handle the caching of output data (i.e. determining the
filename for a given set of parameters).
"""

import re
from pathlib import Path
from sys import hash_info
from typing import ClassVar, Self

import attrs
import numpy as np

from .._cfg import config
from ..wrapper import outputs as op
from ..wrapper.inputs import InputParameters
from ..wrapper.outputs import OutputStruct
from .h5 import read_inputs, read_output_struct, write_output_to_hdf5


@attrs.define(frozen=True)
class OutputCache:
    """An object that manages cache files from 21cmFAST simulations.

    This object has a single attribute -- the top-level directory of the cache. This
    directory can be anywhere on disk. A number of methods exist on the object to
    interact with the cache, including finding existing cache files for a particular
    OutputStruct, writing/reading an OutputStruct to/from the cache, and listing
    existing datasets.

    The cache is meant for single-field OutputStruct objects, not "collections" of
    outputs in an evolved universe (like Coeval or Lightcone objects).
    """

    direc: Path = attrs.field(
        default=Path(config["direc"]).expanduser(), converter=Path
    )

    _path_structures: ClassVar = {
        "InitialConditions": "{user_cosmo:x}/{seed:d}/InitialConditions.h5",
        "PerturbedField": "{user_cosmo:x}/{seed:d}/{zgrid:x}/{redshift:.4f}/PerturbedField.h5",
        "other": "{user_cosmo:x}/{seed:d}/{zgrid:x}/{redshift:.4f}/{astro_flag:x}/{cls}.h5",
    }

    @classmethod
    def _get_hashes(cls, inputs: InputParameters) -> dict[str, str]:
        """Return a dict of hashes for different components of the calculation."""
        # Python builtin hashes can be negative which looks weird in filenames
        max_hash_value = 2**hash_info.width
        return {
            "user_cosmo": hash((inputs.cosmo_params, inputs.matter_params))
            % max_hash_value,
            "seed": inputs.random_seed % max_hash_value,
            "zgrid": hash(inputs.node_redshifts) % max_hash_value,
            "astro_flag": hash((inputs.astro_params, inputs.matter_params))
            % max_hash_value,
        }

    def get_filename(self, obj: OutputStruct) -> str:
        """
        Generate a filename for a given OutputStruct object based on its properties.

        This method constructs a unique filename using the object's class name, redshift
        (if available), and hashes of its input parameters. The filename structure is
        determined by the _path_structures dictionary.

        Parameters
        ----------
        obj : OutputStruct
            The OutputStruct object for which to generate a filename.

        Returns
        -------
        str
            The generated filename for the given OutputStruct object.
        """
        hashes = self._get_hashes(obj.inputs)
        redshift = getattr(obj, "redshift", None)
        kls = obj.__class__.__name__

        pth = self._path_structures.get(kls, self._path_structures["other"])
        return pth.format(redshift=redshift, cls=kls, **hashes)

    def get_path(self, obj: OutputStruct) -> Path:
        """
        Get the full path for a given OutputStruct object.

        This method combines the cache directory with the filename generated
        for the given OutputStruct object to create a complete file path.

        Parameters
        ----------
        obj : OutputStruct
            The OutputStruct object for which to generate the full path.

        Returns
        -------
        Path
            The complete file path for the given OutputStruct object.
        """
        return self.direc / self.get_filename(obj)

    def find_existing(self, obj: OutputStruct) -> Path | None:
        """
        Try to find existing boxes which match the parameters of this instance.

        Parameters
        ----------
        obj : OutputStruct
            The OutputStruct instance to search for.

        Returns
        -------
        Path
            The path to an existing cached OutputStruct matching this instance, or
            None if no match is found.
        """
        # Try an explicit path
        f = self.get_path(obj)
        return f if f.exists() else None

    def write(self, obj: OutputStruct) -> None:
        """
        Write an OutputStruct object to the cache.

        This method writes the given OutputStruct object to an HDF5 file in the cache,
        using the path determined by the object's properties.

        Parameters
        ----------
        obj : OutputStruct
            The OutputStruct object to be written to the cache.
        """
        pth = self.get_path(obj)
        write_output_to_hdf5(obj, path=pth)

    def list_datasets(
        self,
        *,
        kind: str | None = None,
        inputs: InputParameters | None = None,
        all_seeds: bool = True,
        redshift: float | None = None,
    ) -> list[Path]:
        """Return all datasets in the cache which match a given set of filters.

        Parameters
        ----------
        kind: str, optional
            Filter by this kind (a class name of an OutputStruct).
        inputs : InputParameters
            Filter by these input parameters
        all_seeds
            Set to False to only include the seed within `inputs`.
        redshift
            The redshift to search for.

        Returns
        -------
        files
            list of paths pointing to files matching the filters.
        """
        if inputs is not None:
            hashes = self._get_hashes(inputs)
        else:
            hashes = {
                "user_cosmo": ".+?",
                "seed": r"\d+",
                "zgrid": ".+?",
                "astro_flag": ".+?",
            }

        if all_seeds:
            hashes["seed"] = r"\d+"

        hashes["redshift"] = str(redshift) if redshift is not None else ".+?"

        allfiles = self.direc.glob("**/*")
        template = self._path_structures.get(kind, self._path_structures["other"])
        template = template.format(**hashes)
        matches = []
        for fl in allfiles:
            match = re.search(template, fl.name)
            if match is not None:
                matches.append(match)

        return matches

    def load(self, obj: OutputStruct) -> OutputStruct:
        """Load a cache-backed object from disk corresponding to a given object."""
        existing = self.find_existing(obj)

        if existing is None:
            raise OSError(f"No cache exists for {obj} yet!")

        return read_output_struct(existing, struct=obj.__class__.__name__)


def _pathfield():
    return attrs.field(
        default=None,
        converter=attrs.converters.optional(Path),
    )


def _dict_of_paths_field():
    def _convert(x: dict | None) -> tuple[Path]:
        if x is None:
            return x

        if isinstance(x, dict):
            return {float(z): Path(d) for z, d in x.items()}

    return attrs.field(
        default=None,
        converter=_convert,
    )


@attrs.define
class RunCache:
    """An object that specifies all cache files that should/can exist for a full run.

    This object should be instantiated via the `.from_inputs()` class method.

    The instance simply holds references to all possible cache files for a particular
    total simulation (including all evolution over redshift). Not all of these files
    might exist: if a file doesn't exist it implies that the simulation has not run
    for that redshift/field yet. Attributes with values of None are not meant to exist
    as part of the simulation (e.g. they may be TsBox instances when USE_TS_FLUCT=False).
    """

    InitialConditions: Path = _pathfield()
    PerturbedField: dict[float, Path] = _dict_of_paths_field()
    TsBox: dict[float, Path] = _dict_of_paths_field()
    IonizedBox: dict[float, Path] = _dict_of_paths_field()
    BrightnessTemp: dict[float, Path] = _dict_of_paths_field()
    HaloBox: dict[float, Path] | None = _dict_of_paths_field()
    PerturbHaloField: dict[float, Path] | None = _dict_of_paths_field()
    XraySourceBox: dict[float, Path] | None = _dict_of_paths_field()
    inputs: InputParameters | None = attrs.field(default=None)

    @classmethod
    def from_inputs(cls, inputs: InputParameters, cache: OutputCache) -> Self:
        """
        Create a RunCache instance from input parameters and an OutputCache.

        This method generates file paths for various output structures based on the
        provided input parameters and cache configuration.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters for the simulation.
        cache : OutputCache
            The output cache object containing directory and path structure information.

        Returns
        -------
        RunCache
            A new RunCache instance with file paths for various output structures.
        """
        hashes = cache._get_hashes(inputs)
        ics = cache.direc / cache._path_structures["InitialConditions"].format(**hashes)
        pfs = {}

        others = {
            "IonizedBox": {},
            "BrightnessTemp": {},
        }
        if inputs.astro_flags.USE_TS_FLUCT:
            others |= {"TsBox": {}}
        if inputs.astro_flags.USE_HALO_FIELD:
            others |= {"PerturbHaloField": {}, "XraySourceBox": {}, "HaloBox": {}}

        for z in inputs.node_redshifts:
            pfs[z] = cache.direc / cache._path_structures["PerturbedField"].format(
                redshift=z, **hashes
            )

            for name, val in others.items():
                val[z] = cache.direc / cache._path_structures["other"].format(
                    redshift=z, cls=name, **hashes
                )

        return cls(
            InitialConditions=ics,
            PerturbedField=pfs,
            **others,
            inputs=inputs,
        )

    @classmethod
    def from_example_file(cls, path: Path | str) -> Self:
        """Create a RunCache object from an example file.

        This method can be used to determine all the cache files that make up a full
        simulation, given a single example file. Note that this method is somewhat
        ambiguous when the input file is "high up" in the simulation heirarchy (e.g.
        InitialConditions or PerturbedField) because the input parameters to these
        objects may differ from those of the full simulation, in their astro_params
        and astro_flags. For this reason, it is better to supply a cache object like
        IonizedBox or BrightnessTemp.

        Parameters
        ----------
        path : Path | str
            The path to a particular file in cache. The returned OutputCache object
            will include this file.
        """
        inputs = read_inputs(Path(path))
        hashes = OutputCache._get_hashes(inputs)
        hashes["redshift"] = ".+?"
        hashes["cls"] = ".+?"

        for template in OutputCache._path_structures.values():
            # We have to replace the redshift formatter because it's not a float here
            template = template.replace("{redshift:.4f}", "{redshift}")
            template = template.format(**hashes)
            match = re.search(template, str(path))
            if match is not None:
                parent = Path(str(path)[: match.start()])
                break
        else:
            raise ValueError(
                f"The file {path} does not seem to be within a cache structure."
            )

        return cls.from_inputs(inputs, OutputCache(parent))

    def is_complete_at(
        self, z: float | None = None, index: float | None = None
    ) -> bool:
        """Determine whether the simulation has been completed down to a given redshift."""
        if index is not None and z is not None:
            raise ValueError("Cannot specify both z and index")
        if index is not None:
            z = self.inputs.node_redshifts[index]

        for kind in attrs.asdict(self, recurse=False).values():
            if not isinstance(kind, dict):
                continue

            if not kind[z].exists():
                return False
        return True

    def get_output_struct_at_z(
        self,
        kind: type[OutputStruct] | str,
        z: float | None = None,
        index: int | None = None,
        match_z_within: float = 0.01,
    ):
        """Return an output struct of a given kind at or close to a given redshift.

        Parameters
        ----------
        z : float
            The redshift at which to return an output struct.
        index : int
            The node-redshift index at which to return the output struct.
        allow_closest : bool
            Whether to allow the closest redshift available in the cache to be returned.

        Returns
        -------
        OutputStruct
            The output struct corresponding to the kind and redshift.
        """
        if not isinstance(kind, str):
            kind = kind.__name__
        if kind not in attrs.fields_dict(self.__class__):
            raise ValueError(f"Unknown output kind: {kind}")
        if index is not None:
            if z is not None:
                raise ValueError("Cannot specify both z and index")
            z = self.inputs.node_redshifts[index]

        zs_of_kind = np.array(list(getattr(self, kind).keys()))
        if z not in zs_of_kind:
            closest = np.argmin(np.abs(zs_of_kind - z))
            if abs(closest - z) > match_z_within:
                raise ValueError(
                    f"No output struct found for kind '{kind}' at redshift {z} (closest available: {zs_of_kind[closest]} at z={closest})"
                )
            z = closest

        fl = getattr(self, kind)[z]
        return read_output_struct(fl)

    def get_ics(self) -> op.InitialConditions:
        """Return the initial conditions."""
        return read_output_struct(self.InitialConditions)

    def get_all_boxes_at_z(
        self,
        z: float | None = None,
        index: int | None = None,
        match_z_within: float = 0.01,
        return_ics: bool = False,
    ) -> dict[str, OutputStruct]:
        """Return all boxes at or close to a given redshift.

        Parameters
        ----------
        z : float
            The redshift at which to return the boxes.
        index : int
            The node-redshift index at which to return the boxes.
        match_z_within : float
            The maximum difference between the requested and closest available redshift.

        Returns
        -------
        dict[str, Box]
            A dictionary mapping box names to their corresponding Box instances.
        """
        kinds = [
            k
            for k, v in attrs.asdict(self, recurse=False).items()
            if isinstance(v, dict)
        ]

        out = {
            k: self.get_output_struct_at_z(k, z, index, match_z_within) for k in kinds
        }
        if return_ics:
            out["InitialConditions"] = self.get_ics()
        return out

    def get_coeval_at_z(
        self,
        z: float | None = None,
        index: int | None = None,
        match_z_within: float = 0.01,
    ):
        """Return a Coeval object at or close to a given redshift.

        Parameters
        ----------
        z : float
            The redshift at which to return the Coeval object.
        index : int
            The node-redshift index at which to return the Coeval object.
        match_z_within : float
            The maximum difference between the requested and closest available redshift.

        Returns
        -------
        Coeval
            The Coeval object at the given redshift.
        """
        from py21cmfast.drivers.coeval import Coeval

        boxes = self.get_all_boxes_at_z(z, index, match_z_within, return_ics=True)
        return Coeval(
            initial_conditions=boxes["InitialConditions"],
            perturbed_field=boxes["PerturbedField"],
            ionized_box=boxes["IonizedBox"],
            brightness_temperature=boxes["BrightnessTemp"],
            ts_box=boxes.get("TsBox"),
            halobox=boxes.get("HaloBox"),
        )

    def is_complete(self) -> bool:
        """Whether the cache for the full simulation is complete."""
        if not self.InitialConditions.exists():
            return False

        for kind in attrs.asdict(self, recurse=False).values():
            if not isinstance(kind, dict):
                continue

            for fl in kind.values():
                if not fl.exists():
                    return False
        return True


@attrs.define
class CacheConfig:
    """A configuration object that specifies whether a certain field should be cached."""

    initial_conditions: bool = attrs.field(default=True, converter=bool)
    perturbed_field: bool = attrs.field(default=True, converter=bool)
    spin_temp: bool = attrs.field(default=True, converter=bool)
    ionized_box: bool = attrs.field(default=True, converter=bool)
    brightness_temp: bool = attrs.field(default=True, converter=bool)
    halobox: bool = attrs.field(default=True, converter=bool)
    perturbed_halo_field = attrs.field(default=True, converter=bool)
    halo_field = attrs.field(default=True, converter=bool)
    xray_source_box = attrs.field(default=True, converter=bool)

    @classmethod
    def on(cls) -> Self:
        """Generate a CacheConfig where all boxes are cached."""
        return cls()

    @classmethod
    def off(cls):
        """Generate a CacheConfig where no boxes are cached."""
        return cls(
            initial_conditions=False,
            perturbed_field=False,
            spin_temp=False,
            ionized_box=False,
            brightness_temp=False,
            halobox=False,
            perturbed_halo_field=False,
            halo_field=False,
            xray_source_box=False,
        )

    @classmethod
    def noloop(cls):
        """Generate a CacheConfig where only boxes not requiring evolution are cached."""
        return cls(
            initial_conditions=True,
            perturbed_field=True,
            spin_temp=False,
            ionized_box=False,
            brightness_temp=False,
            halobox=False,
            perturbed_halo_field=False,
            halo_field=False,
            xray_source_box=False,
        )
