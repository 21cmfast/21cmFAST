"""Module to deal with the cache.

The module has a manager that essentially establishes a database of cached files,
and provides methods to handle the caching of output data (i.e. determining the
filename for a given set of parameters).
"""

import attrs
import numpy as np
import re
from pathlib import Path
from typing import Self

from .._cfg import config
from ..wrapper.inputs import InputParameters
from ..wrapper.outputs import OutputStruct
from .h5 import read_inputs, read_output_struct, write_output_to_hdf5


@attrs.define(frozen=True)
class OutputCache:
    direc: Path = attrs.field(
        default=Path(config["direc"]).expanduser(), converter=Path
    )

    _path_structures = {
        "InitialConditions": "{user_cosmo}/{seed}/InitialConditions.h5",
        "PerturbedField": "{user_cosmo}/{seed}/{zgrid}/{redshift}/PerturbedField.h5",
        "other": "{user_cosmo}/{seed}/{zgrid}/{redshift}/{astro_flag}/{cls}.h5",
    }

    @classmethod
    def get_hashes(cls, inputs: InputParameters) -> dict[str, str]:
        """Return a dict of hashes for different components of the calculation."""
        return {
            "user_cosmo": hash((inputs.cosmo_params, inputs.user_params)),
            "seed": inputs.random_seed,
            "zgrid": hash(inputs.node_redshifts),
            "astro_flag": hash((inputs.astro_params, inputs.user_params)),
        }

    def get_filename_for_obj(self, obj: OutputStruct):
        hashes = self.get_hashes(obj.inputs)
        redshift = getattr(obj, "redshift", None)
        kls = obj.__class__.__name__

        pth = self._path_structures.get(kls, self._path_structures["other"])
        return pth.format(redshift=redshift, cls=kls, **hashes)

    def get_path_for_obj(self, obj: OutputStruct):
        return self.direc / self.get_filename_for_obj(obj)

    def find_existing(self, obj: OutputStruct) -> Path | None:
        """
        Try to find existing boxes which match the parameters of this instance.

        Parameters
        ----------
        direc : str, optional
            The directory in which to search for the boxes. By default, this is the
            centrally-managed directory, given by the ``config.yml`` in ``~/.21cmfast/``.

        Returns
        -------
        str
            The filename of an existing set of boxes, or None.
        """
        # Try an explicit path
        f = self.get_path_for_obj(obj)
        return f if f.exists() else None

    def write(self, obj: OutputStruct):
        pth = self.get_path_for_obj(obj)
        write_output_to_hdf5(obj, path=pth)

    def list_datasets(
        self,
        *,
        kind: str | None = None,
        inputs: InputParameters | None = None,
        all_seeds: bool = True,
        redshift: float | None = None,
    ) -> list[Path]:
        """Yield all datasets in the cache which match a given set of filters.

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
            hashes = self.get_hashes(inputs)
        else:
            hashes = {
                "user_cosmo": ".+?",
                "seed": r"\d+",
                "zgrid": ".+?",
                "astro_flag": ".+?",
            }

        if all_seeds:
            hashes["seed"] = r"\d+"

        hashes["reshift"] = str(redshift) if redshift is not None else ".+?"

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


def _exists(obj, att, val: Path):
    if not val.exists():
        raise ValueError(f"{att.name}: {val} does not exist")

    if not val.is_file():
        raise ValueError(f"{att.name}: {val} is not a file")


def pathfield():
    return attrs.field(
        default=None,
        converter=attrs.converters.optional(Path),
    )


def dict_of_paths_field():
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
    """An object that specifies all cache files that should/can exist for a full run."""

    InitialConditions: Path = pathfield()
    PerturbedField: tuple[Path] = dict_of_paths_field()
    TsBox: tuple[Path] = dict_of_paths_field()
    IonizedBox: tuple[Path] = dict_of_paths_field()
    BrightnessTemp: tuple[Path] = dict_of_paths_field()
    HaloBox: tuple[Path] = dict_of_paths_field()
    PerturbHaloField: tuple[Path] = dict_of_paths_field()
    XraySourceBox: tuple[Path] = dict_of_paths_field()
    HaloBox: tuple[Path] = dict_of_paths_field()
    _inputs: InputParameters | None = attrs.field(default=None)

    @classmethod
    def from_inputs(cls, inputs: InputParameters, cache: OutputCache):
        hashes = cache.get_hashes(inputs)
        ics = cache.direc / cache._path_structures["InitialConditions"].format(**hashes)
        pfs = {}

        others = {
            "IonizedBox": {},
            "BrightnessTemp": {},
        }
        if inputs.flag_options.USE_TS_FLUCT:
            others |= {"TsBox": {}}
        if inputs.flag_options.USE_HALO_FIELD:
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

    @property
    def inputs(self) -> InputParameters:
        return self._inputs

    @classmethod
    def from_example_file(cls, path: Path | str) -> Self:
        """Create a RunCache object from an example file."""
        inputs = read_inputs(Path(path))
        hashes = OutputCache.get_hashes(inputs)
        hashes["redshift"] = ".+?"
        hashes["cls"] = ".+?"
        for template in OutputCache._path_structures.values():
            template = template.format(**hashes)
            match = re.search(template, str(path))
            if match is not None:
                parent = Path(str(path)[: match.start])
                break
        else:
            raise ValueError(
                f"The file {path} does not seem to be within a cache structure."
            )

        return cls.from_inputs(inputs, OutputCache(parent))

    def is_complete_at(
        self, z: float | None = None, index: float | None = None
    ) -> bool:
        if index is not None and z is not None:
            raise ValueError("Cannot specify both z and index")
        if index is not None:
            z = self.inputs.node_redshifts[index]

        for kind in attrs.asdict(self, recurse=False).values():
            if not isinstance(kind, dict):
                continue

            if not kind[z].exists():
                return False

    def get_output_struct_at_z(
        self,
        kind: type[OutputStruct] | str,
        z: float,
        index: int,
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
        if kind not in attrs.fields_dict(kind):
            raise ValueError(f"Unknown output kind: {kind}")
        if index is not None and z is not None:
            raise ValueError("Cannot specify both z and index")
        if index is not None:
            z = self.inputs.node_redshifts[index]

        zs_of_kind = list(getattr(self, kind).keys())
        if z not in zs_of_kind:
            closest = np.argmin(np.abs(zs_of_kind - z))
            if abs(closest - z) > match_z_within:
                raise ValueError(
                    f"No output struct found for kind '{kind}' at redshift {z} (closest available: {zs_of_kind[closest]} at z={closest})"
                )
            z = closest

        fl = getattr(self, kind)[z]
        return read_output_struct(fl)

    def is_complete(self) -> bool:
        """Whether the cache is complete."""
        if not self.InitialConditions.exists():
            return False

        for kind in attrs.asdict(self, recurse=False).values():
            if not isinstance(kind, dict):
                continue

            for fl in kind.values():
                if not fl.exists():
                    return False

    def get_completion_redshift(self) -> tuple[float, int]:
        """Obtain the redshift down to which the cache is complete."""
        if not self.InitialConditions.exists():
            return None, -1

        zgrid_files = {
            k: v
            for k, v in attrs.asdict(self, recurse=False).items()
            if isinstance(v, dict)
        }

        for i, z in enumerate(self.inputs.node_redshifts):
            for file_dict in zgrid_files.values():
                if z not in file_dict:
                    return self.inputs.node_redshifts[i - 1], i - 1

        return self.inputs.node_redshifts[-1], len(self.inputs.node_redshifts) - 1

    def is_partial(self):
        """Whether the cache is complete down to some redshift, but not the last z."""
        z, idx = self.get_completion_redshift()
        return idx == len(self.inputs.node_redshifts) - 1


@attrs.define
class CacheConfig:
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
    def off(cls):
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
