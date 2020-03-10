"""A set of tools for reading/writing/querying the in-built cache."""
import glob
import logging
import os
from os import path

import h5py

from . import wrapper
from ._cfg import config
from .wrapper import global_params

logger = logging.getLogger("21cmFAST")


def readbox(*, direc=None, fname=None, hsh=None, kind=None, seed=None, load_data=True):
    """
    Read in a data set and return an appropriate object for it.

    Parameters
    ----------
    direc : str, optional
            The directory in which to search for the boxes. By default, this is the
            centrally-managed directory, given by the ``config.yml`` in ``~/.21cmfast/``.
    fname: str, optional
        The filename (without directory) of the data set. If given, this will be
        preferentially used, and must exist.
    hsh: str, optional
        The md5 hsh of the object desired to be read. Required if `fname` not given.
    kind: str, optional
        The kind of dataset, eg. "InitialConditions". Will be the name of a class
        defined in :mod:`~wrapper`. Required if `fname` not given.
    seed: str or int, optional
        The random seed of the data set to be read. If not given, and filename not
        given, then a box will be read if it matches the kind and hsh, with an
        arbitrary seed.
    load_data: bool, optional
        Whether to read in the data in the data set. Otherwise, only its defining
        parameters are read.

    Returns
    -------
    dataset :
        An output object, whose type depends on the kind of data set being read.

    Raises
    ------
    IOError :
        If no files exist of the given kind and hsh.
    ValueError :
        If either ``fname`` is not supplied, or both ``kind`` and ``hsh`` are not supplied.
    """
    direc = direc or path.expanduser(config["direc"])

    # We either need fname, or hsh and kind.
    if not fname and not (hsh and kind):
        raise ValueError("Either fname must be supplied, or kind and hsh")

    if fname:
        kind, hsh, seed = _parse_fname(fname)

    if not seed:
        fname = kind + "_" + hsh + "_r*.h5"
        files = glob.glob(path.join(direc, fname))
        if files:
            fname = files[0]
        else:
            raise IOError("No files exist with that kind and hsh.")
    else:
        fname = kind + "_" + hsh + "_r" + str(seed) + ".h5"

    # Now, open the file and read in the parameters
    with h5py.File(path.join(direc, fname), "r") as fl:
        # First get items out of attrs.
        top_level = {}
        for k, v in fl.attrs.items():
            top_level[k] = v

        # Now descend into each group of parameters
        params = {}
        for grp_nm, grp in fl.items():
            if grp_nm != kind:  # is a parameter
                params[grp_nm] = {}
                for k, v in grp.attrs.items():
                    params[grp_nm][k] = None if v == "none" else v

    # Need to map the parameters to input parameters.
    passed_parameters = {}
    for k, v in params.items():
        if "global_params" in k:
            for kk, vv in v.items():
                setattr(global_params, kk, vv)

        else:
            # The following line takes something like "cosmo_params", turns it into
            # "CosmoParams", and instantiates
            # that particular class with the dictionary parameters.
            passed_parameters[k] = getattr(wrapper, k.title().replace("_", ""))(**v)

    for k, v in top_level.items():
        passed_parameters[k] = v

    # Make an instance of the object.
    inst = getattr(wrapper, kind)(**passed_parameters)

    # Read in the actual data (this avoids duplication of reading data).
    if load_data:
        inst.read(direc=direc)

    return inst


def _parse_fname(fname):
    try:
        kind = fname.split("_")[0]
        hsh = fname.split("_")[1]
        seed = fname.split("_")[-1].split(".")[0][1:]
    except IndexError:
        raise ValueError("fname does not have correct format")

    if kind + "_" + hsh + "_r" + seed + ".h5" != fname:
        raise ValueError("fname does not have correct format")

    return kind, hsh, seed


def list_datasets(*, direc=None, kind=None, hsh=None, seed=None):
    """Yield all datasets which match a given set of filters.

    Can be used to determine parameters of all cached datasets, in conjunction with :func:`readbox`.

    Parameters
    ----------
    direc : str, optional
        The directory in which to search for the boxes. By default, this is the centrally-managed
        directory, given by the ``config.yml`` in ``.21cmfast``.
    kind: str, optional
        Filter by this kind (one of {"InitialConditions", "PerturbedField", "IonizedBox",
        "TsBox", "BrightnessTemp"}
    hsh: str, optional
        Filter by this hsh.
    seed: str, optional
        Filter by this seed.

    Yields
    ------
    fname: str
        The filename of the dataset (without directory).
    parts: tuple of strings
        The (kind, hsh, seed) of the data set.
    """
    direc = direc or path.expanduser(config["direc"])

    kind = kind or "*"
    hsh = hsh or "*"
    seed = seed or "*"

    fname = path.join(direc, str(kind) + "_" + str(hsh) + "_r" + str(seed) + ".h5")

    files = [path.basename(file) for file in glob.glob(fname)]

    for file in files:
        yield file, _parse_fname(file)


def query_cache(*, direc=None, kind=None, hsh=None, seed=None, show=True):
    """Get or print datasets in the cache.

    Walks through the cache, with given filters, and return all un-initialised dataset
    objects, optionally printing their representation to screen.
    Useful for querying which kinds of datasets are available within the cache, and
    choosing one to read and use.

    Parameters
    ----------
    direc : str, optional
        The directory in which to search for the boxes. By default, this is the
        centrally-managed directory, given by the ``config.yml`` in ``~/.21cmfast``.
    kind: str, optional
        Filter by this kind. Must be one of "InitialConditions", "PerturbedField",
        "IonizedBox", "TsBox" or "BrightnessTemp".
    hsh: str, optional
        Filter by this hsh.
    seed: str, optional
        Filter by this seed.
    show: bool, optional
        Whether to print out a repr of each object that exists.

    Yields
    ------
    obj:
       Output objects, un-initialized.
    """
    for file, parts in list_datasets(direc=direc, kind=kind, hsh=hsh, seed=seed):
        cls = readbox(direc=direc, fname=file, load_data=False)
        if show:
            print(file + ": " + str(cls))
        yield file, cls


def clear_cache(**kwargs):
    """Delete datasets in the cache.

    Walks through the cache, with given filters, and deletes all un-initialised dataset
    objects, optionally printing their representation to screen.

    Parameters
    ----------
    kwargs :
        All options passed through to :func:`query_cache`.
    """
    direc = kwargs.get("direc", path.expanduser(config["direc"]))
    number = 0
    for fname, cls in query_cache(show=False, **kwargs):
        if kwargs.get("show", True):
            logger.info("Removing {}".format(fname))
        os.remove(path.join(direc, fname))
        number += 1

    logger.info("Removed {} files from cache.".format(number))
