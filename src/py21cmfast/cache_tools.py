"""A set of tools for reading/writing/querying the in-built cache."""

import glob
import logging
import os
import re
import warnings
from collections import defaultdict
from os import path
from typing import Tuple, Union

from . import outputs
from ._cfg import config

logger = logging.getLogger("21cmFAST")


def readbox(
    *,
    direc=None,
    fname=None,
    hsh=None,
    kind=None,
    seed=None,
    redshift=None,
    load_data=True,
):
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
    direc = path.expanduser(direc or config["direc"])

    if not (fname or (hsh and kind)):
        raise ValueError("Either fname must be supplied, or kind and hsh")

    zstr = f"z{redshift:.4f}_" if redshift is not None else ""
    if not fname:
        if not seed:
            fname = kind + "_" + zstr + hsh + "_r*.h5"
            files = glob.glob(path.join(direc, fname))
            if files:
                fname = files[0]
            else:
                raise OSError("No files exist with that kind and hsh.")
        else:
            fname = kind + "_" + zstr + hsh + "_r" + str(seed) + ".h5"

    kind = _parse_fname(fname)["kind"]
    cls = getattr(outputs, kind)

    if hasattr(cls, "from_file"):
        inst = cls.from_file(
            fname, direc=direc, load_data=load_data
        )  # for OutputStruct
    else:
        inst = cls.read(fname, direc=direc)  # for HighlevelOutputStruct

    return inst


def _parse_fname(fname):
    patterns = (
        r"(?P<kind>\w+)_(?P<hash>\w{32})_r(?P<seed>\d+).h5$",
        r"(?P<kind>\w+)_z(?P<redshift>\d+.\d{1,4})_(?P<hash>\w{32})_r(?P<seed>\d+).h5$",
    )
    for pattern in patterns:
        match = re.match(pattern, os.path.basename(fname))
        if match:
            break

    if not match:
        raise ValueError(
            "filename {} does not have correct format for a cached output.".format(
                fname
            )
        )

    return match.groupdict()


def list_datasets(*, direc=None, kind=None, hsh=None, seed=None, redshift=None):
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
    direc = path.expanduser(direc or config["direc"])

    fname = "{}{}_{}_r{}.h5".format(
        kind or r"(?P<kind>[a-zA-Z]+)",
        f"_z{redshift:.4f}" if redshift is not None else "(.*)",
        hsh or r"(?P<hash>\w{32})",
        seed or r"(?P<seed>\d+)",
    )

    for fl in os.listdir(direc):
        if re.match(fname, fl):
            yield fl


def query_cache(
    *, direc=None, kind=None, hsh=None, seed=None, redshift=None, show=True
):
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
    for file in list_datasets(
        direc=direc, kind=kind, hsh=hsh, seed=seed, redshift=redshift
    ):
        try:
            kls = readbox(direc=direc, fname=file, load_data=False)
        except OSError as e:
            warnings.warn(f"Failed to read {file}: {e}")
            continue

        if show:
            print(f"{file}: {str(kls)}")  # noqa: T201

        yield file, kls


def get_boxes_at_redshift(
    redshift: Union[float, Tuple[float, float]], seed=None, direc=None, **params
):
    """Retrieve objects for each file in cache within given redshift bounds."""
    if not hasattr(redshift, "__len__"):
        redshift = (redshift / 1.001, redshift * 1.001)

    out = defaultdict(list)
    for file in list_datasets(direc=direc, seed=seed):
        try:
            obj = readbox(direc=direc, fname=file, load_data=False)
        except OSError:
            warnings.warn(f"Failed to read {file}")

        if not hasattr(obj, "redshift"):
            logger.debug(f"{file} has no redshift")
            continue
        if not (redshift[0] <= obj.redshift < redshift[1]):
            logger.debug(f"{file} redshift out of range: {obj.redshift}, {redshift}")
            continue
        for paramtype, paramobj in params.items():
            if hasattr(obj, paramtype) and paramobj != getattr(obj, paramtype):
                logger.debug(
                    f"{file} {paramtype} don't match: {getattr(obj, paramtype)} vs. {paramobj}"
                )
                break
        else:
            out[obj.__class__.__name__].append(obj)

    return out


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
    for fname in list_datasets(**kwargs):
        if kwargs.get("show", True):
            logger.info(f"Removing {fname}")
        os.remove(path.join(direc, fname))
        number += 1

    logger.info(f"Removed {number} files from cache.")
