"""Module to deal with the cache.

The module has a manager that essentially establishes a database of cached files,
and provides methods to handle the caching of output data (i.e. determining the
filename for a given set of parameters).
"""

import attrs
from pathlib import Path

from .._cfg import config
from ..wrapper.outputs import OutputStruct


@attrs.define(frozen=True)
class OutputCache:
    direc: Path = attrs.field(default=Path(config["direc"]).expanduser())

    _path_structures = {
        "ics": "{hash_user_cosmo}/{seed}/InitialConditions.h5",
        "pf": "{hash_user_cosmo}/{seed}/{redshift:.5f}/{cls}/{cls}.h5",
        "other": "{hash_user_cosmo}/{seed}/{redshift:.5f}/{hash_astro_flag}/{cls}/{cls}.h5",
    }

    def get_filename_for_obj(self, obj: OutputStruct, with_seed: bool = True):
        # get a hash of the object.
        hsh_user = obj.inputs.hash_user_cosmo_glb()
        hsh_astro = obj.inputs.hash_astro_flag()

        seed = "*" if (not with_seed or obj.random_seed is None) else obj.random_seed
        redshift = obj.inputs.redshift
        kls = obj.__class__.__name__

        if kls == "InitialConditions":
            return self._path_structures["ics"].format(
                hash_user_cosmo_glb=hsh_user, seed=seed
            )
        elif kls == "PerturbedField":
            return self._path_structures["pf"].format(
                hash_user_cosmo=hsh_user,
                cls=kls,
                redshift=redshift,
                seed=seed,
            )
        else:
            return self._path_structures["other"].format(
                hash_user_cosmo=hsh_user,
                hash_astro_flag=hsh_astro,
                cls=kls,
                redshift=redshift,
                seed=seed,
            )

    def get_path_for_obj(self, obj: OutputStruct):
        return self.direc / self.get_filename_for_obj(obj)

    def _find_file_without_seed(self, obj: OutputStruct):
        glob = self.get_filename_for_obj(obj, with_seed=False)
        return allfiles[0] if (allfiles := list(self.direc.glob(glob))) else None

    def find_existing(self, obj) -> Path | None:
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
        # First, if appropriate, find a file without specifying seed.
        # Need to do this first, otherwise the seed will be chosen randomly upon
        # choosing a filename!
        if not obj.random_seed:
            f = self._find_file_without_seed(obj)
            if f:
                return f

        # Try an explicit path
        f = self.get_path_for_obj(obj)
        return f if f.exists() else None
