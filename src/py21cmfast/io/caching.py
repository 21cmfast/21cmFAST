"""Module to deal with the cache.

The module has a manager that essentially establishes a database of cached files,
and provides methods to handle the caching of output data (i.e. determining the
filename for a given set of parameters).
"""

import logging
import re
import warnings
from collections.abc import Mapping
from hashlib import md5
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self, TypedDict, Unpack

import attrs
import deprecation
import numpy as np

from .._cfg import config
from ..wrapper import outputs as op
from ..wrapper.inputs import InputParameters
from ..wrapper.outputs import (
    _ALL_OUTPUT_STRUCTS,
    OutputStruct,
    _HashType,
)
from .h5 import read_inputs, read_output_struct, write_output_to_hdf5

if TYPE_CHECKING:
    # Only imported for annotations, to avoid a circular import.
    from ..drivers.coeval import Coeval

logger = logging.getLogger(__name__)


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

    _output_to_cache_map: ClassVar[dict[str, _HashType]] = {
        kls.__name__: kls._compat_hash
        for kls in _ALL_OUTPUT_STRUCTS.values()
        if not kls._meta
    }
    _path_structures: ClassVar[dict[_HashType, str]] = {
        _HashType.user_cosmo: "{matter_cosmo}/{seed}/InitialConditions.h5",
        _HashType.zgrid: "{matter_cosmo}/{seed}/{zgrid}/{redshift}/{cls}.h5",
        _HashType.full: "{matter_cosmo}/{seed}/{zgrid}/{redshift}/{astro_flag}/{cls}.h5",
    }

    @classmethod
    def _get_hashes(cls, inputs: InputParameters) -> dict[str, str]:
        """Return a dict of hashes for different components of the calculation."""
        # Python builtin hashes can be negative which looks weird in filenames
        return {
            "matter_cosmo": md5(
                (
                    repr(inputs.cosmo_params)
                    + repr(inputs.simulation_options)
                    + repr(inputs.matter_options)
                ).encode()
            ).hexdigest(),
            "seed": str(inputs.random_seed),
            "zgrid": md5(repr(inputs.node_redshifts).encode()).hexdigest(),
            "astro_flag": md5(
                (repr(inputs.astro_params) + repr(inputs.astro_options)).encode()
            ).hexdigest(),
        }

    @classmethod
    def _fill_path_template(
        cls,
        *,
        kind: str | _HashType | None = None,
        inputs: InputParameters | None = None,
        all_seeds: bool = False,
        redshift: float | None = None,
    ) -> str:
        """Fill the path templates with given values.

        Does the conditional formatting required for each field,
        since we don't do number formatting for the wildcards.

        Returns the filled template path with optional wildcards for
        searching.
        """
        # get the hashes
        if inputs is not None:
            hashes = cls._get_hashes(inputs)
            if all_seeds:
                hashes["seed"] = r"\d+"
        else:
            hashes = {
                "matter_cosmo": ".+?",
                "seed": r"\d+",
                "zgrid": ".+?",
                "astro_flag": ".+?",
            }

        # do the conditional formatting
        hashes["redshift"] = f"{redshift:.4f}" if redshift is not None else ".+?"
        hashes["cls"] = (
            kind
            if isinstance(kind, str) and kind in cls._output_to_cache_map
            else ".+?"
        )

        # precedence: outputclass mapped (class name -> template) > template provided as _HashType (template directly) > full astro path
        if isinstance(kind, _HashType):
            hash_type = kind
        elif kind is not None and kind in cls._output_to_cache_map:
            hash_type = cls._output_to_cache_map[kind]
        else:
            hash_type = _HashType.full
        return cls._path_structures[hash_type].format(**hashes)

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
        template = self._fill_path_template(
            kind=obj.__class__.__name__,
            redshift=getattr(obj, "redshift", None),
            inputs=obj.inputs,
            all_seeds=False,
        )
        if isinstance(obj, op.EmissivityFields) and config["EXTRA_EMISSIVITY_FIELDS"]:
            template = template.replace(".h5", "_extra.h5")
        return template

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
        kinds_list = list(self._output_to_cache_map) if kind is None else [kind]
        templates = [
            self._fill_path_template(
                kind=k,
                inputs=inputs,
                all_seeds=all_seeds,
                redshift=redshift,
            )
            for k in kinds_list
        ]

        allfiles = self.direc.glob("**/*")
        matches: list[Path] = []
        for fl in allfiles:
            for template in templates:
                match = re.search(template, str(fl))
                if match is not None:
                    matches.append(Path(match.string))

        return matches

    def load(self, obj: OutputStruct) -> OutputStruct:
        """Load a cache-backed object from disk corresponding to a given object."""
        existing = self.find_existing(obj)

        if existing is None:
            raise OSError(f"No cache exists for {obj} yet!")

        return read_output_struct(existing, struct=obj.__class__.__name__)


def _to_dict_of_paths(
    x: Mapping[float, str | Path] | None,
) -> dict[float, Path] | None:
    """Convert a mapping of redshifts to paths to a dict of floats to Paths."""
    if x is None:
        return None
    if not isinstance(x, Mapping):
        raise TypeError(
            f"Expected a mapping of redshifts to paths (or None), got {type(x)}"
        )
    return {float(z): Path(d) for z, d in x.items()}


def _expect_struct[T: OutputStruct](obj: OutputStruct, kls: type[T]) -> T:
    """Check that an output struct read from the cache is of the expected kind."""
    if not isinstance(obj, kls):
        raise TypeError(
            f"Expected a {kls.__name__} in the cache, but got a {type(obj).__name__}"
        )
    return obj


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

    InitialConditions: Path | None = attrs.field(
        default=None, converter=attrs.converters.optional(Path)
    )
    PerturbedField: dict[float, Path] | None = attrs.field(
        default=None, converter=_to_dict_of_paths
    )
    TsBox: dict[float, Path] | None = attrs.field(
        default=None, converter=_to_dict_of_paths
    )
    IonizedBox: dict[float, Path] | None = attrs.field(
        default=None, converter=_to_dict_of_paths
    )
    BrightnessTemp: dict[float, Path] | None = attrs.field(
        default=None, converter=_to_dict_of_paths
    )
    EmissivityFields: dict[float, Path] | None = attrs.field(
        default=None, converter=_to_dict_of_paths
    )
    HaloCatalog: dict[float, Path] | None = attrs.field(
        default=None, converter=_to_dict_of_paths
    )
    RadiationFields: dict[float, Path] | None = attrs.field(
        default=None, converter=_to_dict_of_paths
    )
    inputs: InputParameters | None = attrs.field(default=None)
    _optional_fields: ClassVar[set[str]] = {"RadiationFields"}

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
        ics = cache.direc / cache._fill_path_template(
            kind="InitialConditions",
            inputs=inputs,
        )

        others: dict[str, dict[float, Path]] = {
            "PerturbedField": {},
            "EmissivityFields": {},
            "IonizedBox": {},
            "BrightnessTemp": {},
        }
        if inputs.astro_options.USE_TS_FLUCT:
            others |= {"RadiationFields": {}, "TsBox": {}}
        if inputs.matter_options.has_discrete_halos:
            others |= {"HaloCatalog": {}}

        for z in inputs.node_redshifts:
            for name, val in others.items():
                val[z] = cache.direc / cache._fill_path_template(
                    kind=name,
                    redshift=z,
                    inputs=inputs,
                )

        return cls(
            InitialConditions=ics,
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
        and astro_options. For this reason, it is better to supply a cache object like
        IonizedBox or BrightnessTemp.

        Parameters
        ----------
        path : Path | str
            The path to a particular file in cache. The returned OutputCache object
            will include this file.
        """
        inputs = read_inputs(Path(path))

        for kind in OutputCache._output_to_cache_map:
            template = OutputCache._fill_path_template(
                kind=kind,
                redshift=None,
                inputs=inputs,
                all_seeds=False,
            )
            match = re.search(template, str(path))
            if match is not None:
                parent = Path(str(path)[: match.start()])
                break
        else:
            raise ValueError(
                f"The file {path} does not seem to be within a cache structure."
            )

        return cls.from_inputs(inputs, OutputCache(parent))

    def get_required_fields(self) -> dict[str, dict[float, Path]]:
        """Return the dict-typed cache fields that matter for completeness checks."""
        return {
            name: kind
            for name, kind in attrs.asdict(self, recurse=False).items()
            if isinstance(kind, dict) and name not in self._optional_fields
        }

    def _node_redshift(self, index: int) -> float:
        """Return the node redshift with a given index."""
        if self.inputs is None:
            raise ValueError(
                "Cannot determine the node redshifts of a RunCache without inputs"
            )
        return self.inputs.node_redshifts[index]

    def is_complete_at(self, z: float | None = None, index: int | None = None) -> bool:
        """Determine whether the simulation has been completed down to a given redshift."""
        if index is not None and z is not None:
            raise ValueError("Cannot specify both z and index")
        if index is not None:
            z = self._node_redshift(index)
        if z is None:
            raise ValueError("Must specify either z or index")

        return all(kind[z].exists() for kind in self.get_required_fields().values())

    def get_output_struct_at_z(
        self,
        kind: type[OutputStruct] | str,
        z: float | None = None,
        index: int | None = None,
        match_z_within: float = 0.01,
    ) -> OutputStruct:
        """Return an output struct of a given kind at or close to a given redshift.

        Parameters
        ----------
        kind : type[OutputStruct] or str
            The kind of output struct (the class, or its name) to return.
        z : float
            The redshift at which to return an output struct.
        index : int
            The node-redshift index at which to return the output struct.
        match_z_within : float
            The maximum difference between the requested and closest available
            redshift.

        Returns
        -------
        OutputStruct
            The output struct corresponding to the kind and redshift.
        """
        if not isinstance(kind, str):
            kind = kind.__name__
        if kind == "HaloBox":
            warnings.warn(
                deprecation.DeprecatedWarning(
                    "HaloBox",
                    deprecated_in="4.3.0",
                    removed_in="5.0.0",
                    details="'HaloBox' has been renamed to 'EmissivityFields'. Please use 'EmissivityFields' instead.",
                ),
                stacklevel=2,
            )
            kind = "EmissivityFields"
        if kind not in attrs.fields_dict(self.__class__):
            raise ValueError(f"Unknown output kind: {kind}")
        if index is not None:
            if z is not None:
                raise ValueError("Cannot specify both z and index")
            z = self._node_redshift(index)
        if z is None:
            raise ValueError("Must specify either z or index")

        paths: dict[float, Path] | Path | InputParameters | None = getattr(self, kind)
        if paths is None or isinstance(paths, Path | InputParameters):
            raise ValueError(
                f"The cache has no output structs of kind '{kind}' at each redshift"
            )

        zs_of_kind = np.array(list(paths.keys()))
        if z not in zs_of_kind:
            closest = np.argmin(np.abs(zs_of_kind - z))
            if abs(zs_of_kind[closest] - z) > match_z_within:
                raise ValueError(
                    f"No output struct found for kind '{kind}' at redshift {z} (closest available: {zs_of_kind[closest]} at z idx = {closest})"
                )
            z = float(zs_of_kind[closest])

        return read_output_struct(paths[z])

    def get_ics(self) -> op.InitialConditions:
        """Return the initial conditions."""
        if self.InitialConditions is None:
            raise ValueError("The cache has no InitialConditions")
        return _expect_struct(
            read_output_struct(self.InitialConditions), op.InitialConditions
        )

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
        kinds = self.get_required_fields().keys()

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
    ) -> "Coeval":
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
        ts_box = boxes.get("TsBox")
        emissivity_fields = boxes.get("EmissivityFields")
        return Coeval(
            initial_conditions=_expect_struct(
                boxes["InitialConditions"], op.InitialConditions
            ),
            perturbed_field=_expect_struct(boxes["PerturbedField"], op.PerturbedField),
            ionized_box=_expect_struct(boxes["IonizedBox"], op.IonizedBox),
            brightness_temperature=_expect_struct(
                boxes["BrightnessTemp"], op.BrightnessTemp
            ),
            ts_box=None if ts_box is None else _expect_struct(ts_box, op.TsBox),
            emissivity_fields=(
                None
                if emissivity_fields is None
                else _expect_struct(emissivity_fields, op.EmissivityFields)
            ),
        )

    def is_complete(self) -> bool:
        """Whether the cache for the full simulation is complete."""
        if self.InitialConditions is None or not self.InitialConditions.exists():
            return False

        for kind in self.get_required_fields().values():
            for fl in kind.values():
                if not fl.exists():
                    return False
        return True

    @property
    def HaloBox(self) -> dict[float, Path] | None:
        """A deprecated property that returns the EmissivityFields dictionary as HaloBox."""
        warnings.warn(
            deprecation.DeprecatedWarning(
                "HaloBox",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details=(
                    "HaloBox has been renamed to EmissivityFields. "
                    "Please use EmissivityFields instead."
                ),
            ),
            stacklevel=2,
        )
        return self.EmissivityFields


class CacheConfigUpdate(TypedDict, total=False):
    """A TypedDict for updating CacheConfig objects."""

    initial_conditions: bool
    perturbed_field: bool
    spin_temp: bool
    ionized_box: bool
    brightness_temp: bool
    emissivity_fields: bool
    halo_catalog: bool
    radiation_fields: bool
    # Deprecated aliases (still accepted, with a DeprecatedWarning).
    halobox: bool
    xray_source_box: bool


@attrs.define(init=False)
class CacheConfig:
    """A configuration object that specifies whether a certain field should be cached."""

    initial_conditions: bool = attrs.field(default=True, converter=bool)
    perturbed_field: bool = attrs.field(default=True, converter=bool)
    spin_temp: bool = attrs.field(default=True, converter=bool)
    ionized_box: bool = attrs.field(default=True, converter=bool)
    brightness_temp: bool = attrs.field(default=True, converter=bool)
    emissivity_fields: bool = attrs.field(default=True, converter=bool)
    halo_catalog: bool = attrs.field(default=True, converter=bool)
    radiation_fields: bool = attrs.field(default=True, converter=bool)

    def __init__(self, *args: bool, **kwargs: Unpack[CacheConfigUpdate]) -> None:
        """Create the object, accepting deprecated aliases of the fields."""
        fields = self._resolve_deprecated_aliases(kwargs)
        # attrs generates __attrs_init__ (the init it would have created) because of
        # init=False, but ty does not know about it.
        self.__attrs_init__(*args, **fields)  # ty: ignore[unresolved-attribute]

    @classmethod
    def _resolve_deprecated_aliases(
        cls, kwargs: CacheConfigUpdate, stacklevel: int = 3
    ) -> CacheConfigUpdate:
        """Return the given fields, with deprecated aliases renamed (with a warning).

        The stacklevel is that of the caller of the public method (relative to this
        method).
        """
        fields = kwargs.copy()
        if "halobox" in fields:
            cls._warn_deprecated_alias("halobox", "emissivity_fields", stacklevel)
            fields["emissivity_fields"] = fields.pop("halobox")
        if "xray_source_box" in fields:
            cls._warn_deprecated_alias(
                "xray_source_box", "radiation_fields", stacklevel
            )
            fields["radiation_fields"] = fields.pop("xray_source_box")
        return fields

    @staticmethod
    def _warn_deprecated_alias(old: str, new: str, stacklevel: int) -> None:
        warnings.warn(
            deprecation.DeprecatedWarning(
                old,
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details=f"'{old}' has been renamed to '{new}'. Please use '{new}' instead.",
            ),
            stacklevel=stacklevel + 1,
        )

    def update(self, **kwargs: Unpack[CacheConfigUpdate]) -> Self:
        """Return a new CacheConfig with the given fields updated."""
        return attrs.evolve(self, **self._resolve_deprecated_aliases(kwargs))

    @classmethod
    def on(cls, **kwargs: Unpack[CacheConfigUpdate]) -> Self:
        """Generate a CacheConfig where all boxes are cached."""
        return cls().update(**kwargs)

    @classmethod
    def off(cls, **kwargs: Unpack[CacheConfigUpdate]) -> Self:
        """Generate a CacheConfig where no boxes are cached."""
        return cls(
            initial_conditions=False,
            perturbed_field=False,
            spin_temp=False,
            ionized_box=False,
            brightness_temp=False,
            emissivity_fields=False,
            halo_catalog=False,
            radiation_fields=False,
        ).update(**kwargs)

    @classmethod
    def noloop(cls, **kwargs: Unpack[CacheConfigUpdate]) -> Self:
        """Generate a CacheConfig where only boxes not requiring evolution are cached."""
        return cls(
            initial_conditions=True,
            perturbed_field=True,
            spin_temp=False,
            ionized_box=False,
            brightness_temp=False,
            emissivity_fields=False,
            halo_catalog=True,
            radiation_fields=False,
        ).update(**kwargs)

    @classmethod
    def last_step_only(cls, **kwargs: Unpack[CacheConfigUpdate]) -> Self:
        """Generate a CacheConfig where only boxes needed from more than one step away are cached.

        This represents the minimum caching setup which will *never* store every redshift in memory.
        PerturbedField and PerturbedHaloCatalogs are all calculated at the start of the run, and EmissivityFields
        is required at multiple redshifts for RadiationFields. So this caching setup allows free
        purging of these objects without losing data.
        """
        return cls(
            initial_conditions=False,
            perturbed_field=True,
            spin_temp=False,
            ionized_box=False,
            brightness_temp=False,
            emissivity_fields=True,
            halo_catalog=True,
            radiation_fields=False,
        ).update(**kwargs)

    @property
    def halobox(self) -> bool:
        """A deprecated property that returns the emissivity_fields boolean as halobox."""
        warnings.warn(
            deprecation.DeprecatedWarning(
                "halobox",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details=(
                    "halobox has been renamed to emissivity_fields. "
                    "Please use emissivity_fields instead."
                ),
            ),
            stacklevel=2,
        )
        return self.emissivity_fields
