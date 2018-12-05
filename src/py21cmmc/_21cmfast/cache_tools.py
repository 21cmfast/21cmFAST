from os import path
from .wrapper import config, global_params, CosmoParams, UserParams, AstroParams, FlagOptions, InitialConditions, PerturbedField, IonizedBox, TsBox, BrightnessTemp # require these for the global call below
import glob
import h5py


def readbox(*, direc=None, fname=None, hash=None, kind=None, seed=None, load_data=True):
    """
    A function to read in a data set and return an appropriate object for it.

    Parameters
    ----------
    direc : str, optional
        The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
        by the ``config.yml`` in ``.21CMMC``.
    fname: str, optional
        The filename (without directory) of the data set. If given, this will be preferentially used, and must exist.
    hash: str, optional
        The md5 hash of the object desired to be read. Required if `fname` not given.
    kind: str, optional
        The kind of dataset, eg. "InitialConditions". Will be the name of a class defined in :mod:`~wrapper`. Required
        if `fname` not given.
    seed: str or int, optional
        The random seed of the data set to be read. If not given, and filename not given, then a box will be read if
        it matches the kind and hash, with an arbitrary seed.
    load_data: bool, optional
        Whether to read in the data in the data set. Otherwise, only its defining parameters are read.

    Returns
    -------
    dataset:
        An output object, whose type depends on the kind of data set being read.
    """
    direc = direc or path.expanduser(config['boxdir'])

    # We either need fname, or hash and kind.
    if not fname and not (hash and kind):
        raise ValueError("Either fname must be supplied, or kind and hash")

    if fname:
        kind, hash, seed = _parse_fname(fname)

    if not seed:
        fname = kind + "_" + hash + "_r*.h5"
        files = glob.glob(path.join(direc, fname))
        if files:
            fname = files[0]
        else:
            raise IOError("No files exist with that kind and hash.")
    else:
        fname = kind + "_" + hash + "_r" + str(seed) + ".h5"

    # Now, open the file and read in the parameters
    with h5py.File(path.join(direc, fname), 'r') as fl:
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
                    params[grp_nm][k] = None if v == u"none" else v

    # Need to map the parameters to input parameters.
    passed_parameters = {}
    for k, v in params.items():
        if "global_params" in k:
            for kk, vv in v.items():
                setattr(global_params, kk, vv)

        else:
            # The following line takes something like "cosmo_params", turns it into "CosmoParams", and instantiates
            # that particular class with the dictionary parameters.
            passed_parameters[k] = globals()[k.title().replace("_", "")](**v)

    for k, v in top_level.items():
        passed_parameters[k] = v

    # Make an instance of the object.
    inst = globals()[kind](**passed_parameters)

    # Read in the actual data (this avoids duplication of reading data).
    if load_data:
        inst.read(direc=direc)

    return inst


def _parse_fname(fname):
    try:
        kind = fname.split("_")[0]
        hash = fname.split("_")[1]
        seed = fname.split("_")[-1].split(".")[0][1:]
    except IndexError:
        raise ValueError("fname does not have correct format")

    if kind + "_" + hash + "_r" + seed + ".h5" != fname:
        raise ValueError("fname does not have correct format")

    return kind, hash, seed


def list_datasets(*, direc=None, kind=None, hash=None, seed=None):
    """
    Yield all datasets which match a given set of filters.

    Can be used to determine parameters of all cached datasets, in conjunction with readbox.

    Parameters
    ----------
    direc : str, optional
        The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
        by the ``config.yml`` in ``.21CMMC``.
    kind: str, optional
        Filter by this kind. Must be one of "InitialConditions", "PerturbedField", "IonizedBox", "TsBox" or "BrightnessTemp".
    hash: str, optional
        Filter by this hash.
    seed: str, optional
        Filter by this seed.

    Yields
    ------
    fname: str
        The filename of the dataset (without directory).
    parts: tuple of strings
        The (kind, hash, seed) of the data set.
    """
    direc = direc or path.expanduser(config['boxdir'])

    kind = kind or "*"
    hash = hash or "*"
    seed = seed or "*"

    fname = path.join(direc, str(kind) + "_" + str(hash) + "_r" + str(seed) + ".h5")

    files = [path.basename(file) for file in glob.glob(fname)]

    for file in files:
        yield file, _parse_fname(file)


def query_cache(*, direc=None, kind=None, hash=None, seed=None, show=True):
    """
    Walk through the cache, with given filters, and return all un-initialised dataset objects, optionally printing
    their representation to screen.

    Usefor for querying which kinds of datasets are available within the cache, and choosing one to read and use.

    Parameters
    ----------
    direc : str, optional
        The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
        by the ``config.yml`` in ``.21CMMC``.
    kind: str, optional
        Filter by this kind. Must be one of "InitialConditions", "PerturbedField", "IonizedBox", "TsBox" or "BrightnessTemp".
    hash: str, optional
        Filter by this hash.
    seed: str, optional
        Filter by this seed.
    show: bool, optional
        Whether to print out a repr of each object that exists.

    Yields
    ------
    obj:
       Output objects, un-initialized.
    """
    for file, parts in list_datasets(direc=direc, kind=kind, hash=hash, seed=seed):
        cls = readbox(direc=direc, fname=file, load_data=False)
        if show:
            print(file + ": " + str(cls))
        yield file, cls
