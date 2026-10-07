Reading Outputs from Older Versions
===================================

From time to time, new versions of ``21cmFAST`` change the format of the files that
it writes: input parameters and output fields are renamed, added or removed, and the
meaning of some fields changes. ``21cmFAST`` keeps a record of every such change since
v4.0 (in :mod:`py21cmfast.io.compat`), and uses it to read files written by older
versions, converting them to objects of the current version.

Reading old files
-----------------
This happens automatically whenever you read a file, e.g. with
:func:`~py21cmfast.io.h5.read_output_struct`,
:func:`~py21cmfast.io.h5.read_inputs`, ``Coeval.from_file``, ``LightCone.from_file``
or ``GlobalEvolution.from_file``::

    >>> import py21cmfast as p21c
    >>> coeval = p21c.Coeval.from_file("coeval_written_by_v4.2.h5")

Renamed parameters and fields are renamed, and fields whose meaning changed are
converted where possible. Where an old file is missing information that the current
version needs (or contains information that can't be converted), a
:class:`~py21cmfast.io.compat.CompatibilityWarning` is raised. Some outputs have no
equivalent in the current version at all (e.g. the ``XraySourceBox`` of versions before
v4.3), and raise an :class:`~py21cmfast.io.compat.UnconvertibleError`.

Input parameters are converted so that they describe the physics of the older version
as closely as possible. For example, ``USE_METALLICITY`` (new in v4.3) is set to the
value of ``USE_UPPER_STELLAR_TURNOVER`` for older files, since that switched on the
metallicity-dependence of the X-ray luminosity in older versions.

To see every change to the file format since v4.0, use::

    $ 21cmfast migrate --explain

Files written before v4.0 are not supported. Their arrays can still be read directly
with ``h5py``.

Migrating a cache
-----------------
The cache (see :class:`~py21cmfast.io.caching.OutputCache`) finds the boxes of a
simulation using hashes of their input parameters. Since these change whenever input
parameters are added or renamed, the current version can't *find* boxes written by
an older version, even if it can read them. To make it able to, migrate the cache::

    $ 21cmfast migrate path/to/old-cache --out path/to/new-cache

This reads every box in the old cache, and writes it in the current format to the
location that the current version expects. If ``--out`` is not given, the migrated
boxes are written into the old cache directory itself. Original files are never
removed. Use ``--dry-run`` to see what would be done, and ``--kind`` to migrate only
some kinds of boxes, e.g.::

    $ 21cmfast migrate old-cache --out new-cache --kind InitialConditions PerturbedField

The same can be done from Python with :func:`~py21cmfast.io.migrate.migrate_cache`.

Saved coeval boxes, lightcones and global evolutions can be migrated too::

    $ 21cmfast migrate lightcone.h5 --out lightcone-v4.3.h5

.. warning:: Migrating a box changes its *format*, not its *contents*. Boxes that
   depend on parts of the physics that changed since they were written (e.g. spin
   temperatures, ionization boxes and brightness temperatures written before v4.3) are
   not what the current version would compute for the same parameters. A simulation
   that continues from such boxes mixes the physics of the two versions. Initial
   conditions and perturbed fields are typically unaffected, so migrating only those
   (with ``--kind``) is the safest way to re-use an old cache.
