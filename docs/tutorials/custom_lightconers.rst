====================================
How lightcones are made: lightconers
====================================

:func:`~py21cmfast.run_lightcone` does not decide how a lightcone is made. It computes
the coeval boxes at each node redshift, and hands them, one redshift at a time, to a
**lightconer**: the object you pass as ``lightconer=``. The lightconer returns the
lightcone piece by piece, and ``run_lightcone`` stores the pieces (and takes care of
everything else: global quantities, checkpoints, velocity corrections).

This page shows

* how to use the lightconer that comes with 21cmFAST,
* where to get others (angular, HEALPix, ...),
* how to write your own, and
* what to change in code written for v4.3.

.. contents::
   :local:
   :depth: 1


The loop that ``run_lightcone`` runs
====================================

Stripped of the bookkeeping, this is all that ``run_lightcone`` does with a
lightconer:

.. code-block:: python

    session = lightconer.start(context)
    lightcones = {q: np.zeros(session.shape) for q in lightconer.quantities}

    for snapshot in snapshots:  # one per node redshift, from high to low
        for quantity, index, values in session.ingest(snapshot):
            lightcones[quantity][index] = values

    for quantity, index, values in session.finish():
        lightcones[quantity][index] = values

There are two objects:

The **lightconer**
    What you make and pass in. It *describes* the lightcone: its ``quantities``, the
    comoving distances of its slices (``lc_distances``), and whatever else defines
    its geometry. It does not change during a run.

The **session**
    The state of one run, made by ``lightconer.start(context)``. It is given each
    ``snapshot`` (the fields at one node redshift) in turn, and returns ``Chunk``
    objects: pieces of the lightcone that it can now make. Everything that has to be
    kept between redshifts lives in the session.

``run_lightcone`` assumes nothing about *how* the session makes the lightcone. In
particular it does not assume that a slice is made from the two boxes around it, or
that the lightcone is filled in any order. It is the session that keeps earlier
snapshots (as many as it needs), and any interpolators or other things worth caching.

What exactly is expected of the two objects is defined in
:mod:`py21cmfast.lightconers.protocol`
(:class:`~py21cmfast.lightconers.LightconerProtocol` and
:class:`~py21cmfast.lightconers.LightconeSession`), and summarised in
`What run_lightcone calls, and when`_ below.


Using the lightconer of 21cmFAST
================================

21cmFAST comes with one lightconer, :class:`~py21cmfast.RectilinearLightconer`. Its
lightcones are boxes with two transverse axes and a line-of-sight axis: each slice is
a slice of the coeval box, interpolated linearly in comoving distance between the two
node redshifts around it.

.. code-block:: python

    import py21cmfast as p21c

    inputs = p21c.InputParameters.from_template(
        ["simple", "tiny"], random_seed=1
    ).with_logspaced_redshifts(zmin=7.0, zmax=12.0)

    lightconer = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=7.0,
        max_redshift=10.0,
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
        quantities=("brightness_temp", "density"),
    )

    lightcone = p21c.run_lightcone(lightconer=lightconer, inputs=inputs)
    lightcone.lightcones["brightness_temp"].shape  # (32, 32, 551)
    sorted(lightcone.lightcones)  # ['brightness_temp', 'density', 'los_velocity']

Nothing has changed here with respect to v4.3. (The ``los_velocity`` lightcone is
there because velocity-gradient corrections are applied by default: see
`Writing your own lightconer`_ for how a lightconer is asked for it.)


Other lightconers: ``tuesday``
==============================

Any other kind of lightcone is made by a lightconer from another package. The
`tuesday <https://tuesday.readthedocs.io>`_ package has lightconers for angular
lightcones (sets of directions on the sky, including HEALPix maps, with sub-sampled
pixels), and a base class that makes it easy to write lightconers of the same family.
They follow the protocol described on this page, so they are passed to
``run_lightcone`` in the same way:

.. code-block:: python

    from tuesday.core.lightcones import AngularLightconer

    lightconer = AngularLightconer.healpix(
        nside=16,
        min_redshift=7.0,
        max_redshift=10.0,
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )

    # The velocity along the line of sight of an angular lightcone needs all three
    # components of the velocity.
    inputs_3d = inputs.evolve_input_structs(KEEP_3D_VELOCITIES=True)

    lightcone = p21c.run_lightcone(lightconer=lightconer, inputs=inputs_3d)
    lightcone.lightcones["brightness_temp"].shape  # (3072, 551)

``tuesday`` is not installed with 21cmFAST. Install it with ``pip install
tuesday-eor``, or with ``pip install 21cmFAST[lightcones]``.

.. note::

    Until v5, ``py21cmfast.AngularLightconer`` still exists: it is the lightconer of
    ``tuesday``, with the defaults and methods it had in 21cmFAST v4.3. It raises an
    ``ImportError`` that says what to install if ``tuesday`` is missing. From v5, use
    ``tuesday.core.lightcones.AngularLightconer`` directly.


Writing your own lightconer
===========================

A lightconer does not need to inherit from anything, or even to import 21cmFAST: it
only needs the right attributes and methods. Here is a complete one. It does no
interpolation at all: each slice of the lightcone is taken from the *nearest*
snapshot.

.. code-block:: python

    import dataclasses

    import numpy as np
    from astropy import units as un

    from py21cmfast.lightconers import Chunk, slices_between


    @dataclasses.dataclass(frozen=True)
    class NearestLightconer:
        """Each slice of the lightcone is a slice of the nearest snapshot."""

        lc_distances: un.Quantity
        quantities: tuple[str, ...] = ("brightness_temp",)

        def evolve(self, **changes):
            # run_lightcone uses this to add quantities, and to extend the lightcone.
            return dataclasses.replace(self, **changes)

        def required_fields(self):
            # The fields of the simulation that the quantities are made from.
            # "los_velocity" is not a field: here it is made from velocity_z.
            return tuple(
                "velocity_z" if q == "los_velocity" else q for q in self.quantities
            )

        def start(self, context):
            return NearestSession(self, context)


    class NearestSession:
        # After a restart, one earlier snapshot is enough to carry on.
        resume_depth = 1

        def __init__(self, lightconer, context):
            self.lightconer = lightconer
            self.context = context
            # The shape of the lightcone of each quantity (line of sight last).
            self.shape = (*context.box_shape[:2], len(lightconer.lc_distances))
            self.distances = lightconer.lc_distances.to_value(un.Mpc)
            self.previous = None  # (distance, snapshot)

        def ingest(self, snapshot):
            distance = self.context.cosmo.comoving_distance(snapshot.redshift)
            distance = distance.to_value(un.Mpc)
            previous, self.previous = self.previous, (distance, snapshot)
            if previous is None:
                return  # nothing can be made from the first snapshot alone

            far_distance, far = previous
            fields = self.lightconer.required_fields()
            # The indices of the slices that lie between the two snapshots.
            for i in slices_between(self.distances, distance, far_distance):
                d = self.distances[i]
                nearest = far if far_distance - d < d - distance else snapshot
                for quantity, field in zip(self.lightconer.quantities, fields):
                    box = nearest[field]  # only now is the box got (or read)
                    yield Chunk(quantity, (..., i), box[:, :, i % box.shape[2]])

        def finish(self):
            return ()  # everything was returned by ingest

It is used like any other:

.. code-block:: python

    cosmo = inputs.cosmo_params.cosmo
    cell = inputs.simulation_options.cell_size.to_value(un.Mpc)
    distances = np.arange(
        cosmo.comoving_distance(7.0).to_value(un.Mpc),
        cosmo.comoving_distance(10.0).to_value(un.Mpc),
        cell,
    ) * un.Mpc

    lightcone = p21c.run_lightcone(
        lightconer=NearestLightconer(lc_distances=distances), inputs=inputs
    )

A few things to note:

**The snapshot.** ``snapshot.redshift`` is its redshift, and ``snapshot["density"]``
is the box of a field. A box is only computed, or read from the cache, when it is
asked for, so ask only for what you use (here, nothing is read at a node redshift
that has no slices next to it).

**The context.** ``start`` receives what the lightconer needs to know about the
simulation, as plain values: ``context.cosmo``, ``context.box_shape``,
``context.cell_size`` and ``context.node_redshifts``.

**Chunks.** ``Chunk(quantity, index, values)`` means ``lightcones[quantity][index] =
values``. For a whole slice the index is ``(..., i)``, but any index that numpy
accepts will do: part of a slice (some of the pixels, say), or several slices at once.

**Quantities and fields.** ``quantities`` are the names of the lightcones that are
made. ``required_fields()`` are the names of the fields of the simulation that they
are made from. These are usually the same, but need not be. ``run_lightcone`` checks,
before running anything, that the simulation computes the required fields.

**Velocities.** With ``include_dvdr_in_tau21=True`` (the default) or
``apply_rsds=True``, ``run_lightcone`` adds the quantity ``"los_velocity"`` (and, with
``USE_TS_FLUCT``, ``"tau_21"``) with ``lightconer.evolve(quantities=...)``. The
lightconer decides what the velocity along *its* line of sight is. With
``apply_rsds=True`` the lightcone is also extended along the line of sight, with
``lightconer.evolve(lc_distances=...)``, and trimmed again at the end.


Keeping more than two snapshots
-------------------------------

Since the session owns everything that is kept between redshifts, an algorithm that
needs more than the two nearest snapshots is not a special case. This session
interpolates *quadratically*, through three snapshots:

.. code-block:: python

    from collections import deque


    class QuadraticSession:
        # After snapshot k, the slices between snapshots k-1 and k are still to be
        # made (from k-1, k and k+1): a new session needs the last two to carry on.
        resume_depth = 2

        def __init__(self, lightconer, context):
            self.lightconer = lightconer
            self.context = context
            self.shape = (*context.box_shape[:2], len(lightconer.lc_distances))
            self.distances = lightconer.lc_distances.to_value(un.Mpc)
            self.history = deque(maxlen=3)  # (distance, {field: box})

        def ingest(self, snapshot):
            distance = self.context.cosmo.comoving_distance(snapshot.redshift)
            # Get the boxes now: they are needed for the next two snapshots as well.
            boxes = {f: snapshot[f] for f in self.lightconer.required_fields()}
            self.history.append((distance.to_value(un.Mpc), boxes))
            if len(self.history) == 3:
                # The slices between the two older snapshots.
                yield from self.make(self.history[1][0], self.history[0][0])

        def finish(self):
            # The slices between the last two snapshots.
            yield from self.make(self.history[-1][0], self.history[-2][0])

        def make(self, dmin, dmax):
            nodes = [d for d, _ in self.history]
            for i in slices_between(self.distances, dmin, dmax):
                d = self.distances[i]
                weights = [
                    np.prod([(d - m) / (n - m) for m in nodes if m != n])
                    for n in nodes
                ]
                for quantity in self.lightconer.quantities:
                    values = sum(
                        w * boxes[quantity][:, :, i % self.shape[0]]
                        for w, (_, boxes) in zip(weights, self.history)
                    )
                    yield Chunk(quantity, (..., i), values)

Three things differ from the session above:

* It keeps three snapshots instead of one. A session that keeps a snapshot should
  ask it for the boxes it will need *during the* ``ingest`` *call that passes it*:
  ``run_lightcone`` releases the memory of the simulation afterwards, and a box that
  the session holds on to stays in memory for as long as the session keeps it.
* The slices between two snapshots are returned one snapshot later, and the last ones
  by ``finish()``, which is called once after the last snapshot.
* ``resume_depth`` is 2. When a run is resumed from a checkpoint
  (``lightcone_filename=``), a new session is started, and it is first passed the
  last ``resume_depth`` snapshots of the earlier run again (read from the cache).
  Whatever it returns for those is dropped, since it is in the file already.

The same goes for caches: an interpolator that is the same for every quantity of a
slice, or for every slice, is made once and kept on the session, for as long as the
session wants. ``run_lightcone`` never sees it.


Writing a lightconer the easy way
---------------------------------

If your lightconer makes each slice from the two snapshots around it, you do not need
to write the session. ``tuesday.core.lightcones.BracketingLightconer`` is a base class
that finds the slices between each pair of snapshots, and calls a few methods that you
define:

.. code-block:: python

    import attrs
    from tuesday.core.lightcones import BracketingLightconer


    @attrs.define(kw_only=True, slots=False)
    class ColumnLightconer(BracketingLightconer):
        """A lightcone of one line of sight through the box: shape (1, n_slices)."""

        x: int = 0
        y: int = 0

        def lightcone_shape(self, context):
            return (1, len(self.lc_distances))

        def prepare_slice(self, slice_, session):
            # Called once per slice. What it returns is ``slice_.state``, shared by
            # all the quantities of the slice.
            return slice_.index % session.context.box_shape[2]

        def make_field_slice(self, near, far, slice_, session, kind="mean"):
            # ``near`` and ``far`` are the boxes of one field around the slice.
            k = slice_.state
            return slice_.interpolate(
                near[self.x, self.y, k], far[self.x, self.y, k], kind=kind
            )

``session.cache`` has three dictionaries for things that are worth keeping:
``cache.slice`` (emptied after each slice), ``cache.interval`` (emptied after each
pair of snapshots) and ``cache.run`` (kept for the whole run). See the documentation
of ``tuesday`` for more.


What ``run_lightcone`` calls, and when
======================================

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - What
     - When
     - Why
   * - ``lightconer.quantities``, ``lightconer.lc_distances``
     - before the run
     - The lightcones to make, and the distances of their slices.
   * - ``lightconer.evolve(quantities=...)``
     - before the run
     - To add ``"los_velocity"`` and ``"tau_21"`` for the velocity corrections.
   * - ``lightconer.evolve(lc_distances=...)``
     - before the run
     - To extend the lightcone for redshift-space distortions.
   * - ``lightconer.required_fields()``
     - before the run
     - To check that the simulation computes the fields.
   * - ``lightconer.start(context)``
     - once
     - To get the session.
   * - ``session.shape``
     - once
     - To make the arrays (and files) of the lightcones.
   * - ``session.resume_depth``
     - when resuming
     - How many earlier snapshots the session is passed again.
   * - ``session.ingest(snapshot)``
     - at each node redshift
     - Returns the chunks that can be made now.
   * - ``session.finish()``
     - after the last node
     - Returns the chunks that are left.

A lightconer may also have the attribute ``los_velocity_rms_factor``: the ratio of the
rms of its ``"los_velocity"`` to that of the full (3D) velocity (:math:`1/\sqrt{3}`
for one component). It is only used to choose how far the lightcone is extended for
redshift-space distortions, and is taken to be 1 (the safe choice) if absent.


Changes from v4.3
=================

``run_lightcone`` is called as before, and gives the same lightcones. What has
changed is what a lightconer *is*:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - v4.3
     - Now
   * - ``py21cmfast.RectilinearLightconer``
     - Unchanged.
   * - ``py21cmfast.AngularLightconer``
     - Still available until v5 if ``tuesday`` is installed. Use
       ``tuesday.core.lightcones.AngularLightconer``.
   * - Subclassing ``py21cmfast.lightconers.Lightconer``
     - Still works until v5 (with a deprecation warning). Follow
       :class:`~py21cmfast.lightconers.LightconerProtocol` instead, or subclass
       ``tuesday.core.lightcones.BracketingLightconer``.
   * - ``lightconer.make_lightcone_slices(c1, c2)``
     - Deprecated. ``session = lightconer.start(context)``, then
       ``session.ingest(snapshot)``.
   * - ``lightconer.validate_options(inputs, ...)``
     - Deprecated. :func:`py21cmfast.lightconers.prepare_lightconer`.
   * - ``lightconer.get_shape(simulation_options)``
     - Deprecated. ``lightconer.start(context).shape``.
   * - ``lightconer.find_required_lightcone_limits(...)``
     - Deprecated. :func:`py21cmfast.rsds.find_required_lightcone_limits`.
   * - ``lightconer.redshift_interpolation(...)``
     - Deprecated. :func:`py21cmfast.lightconers.redshift_interpolation`.
   * - A quantity that the simulation does not compute raises ``AttributeError``
       during the run
     - It raises ``ValueError`` before anything is run.

To call a lightconer yourself (outside ``run_lightcone``), make the context and the
snapshots from 21cmFAST objects with
:func:`~py21cmfast.lightconers.context_from_inputs` and
:class:`~py21cmfast.lightconers.CoevalSnapshot`:

.. code-block:: python

    from py21cmfast.lightconers import CoevalSnapshot, context_from_inputs

    session = lightconer.start(context_from_inputs(inputs))
    for coeval in coevals:  # from high to low redshift
        for quantity, index, values in session.ingest(CoevalSnapshot(coeval)):
            ...
