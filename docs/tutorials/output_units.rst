Working with Output Units
=========================

Output structures provide units and descriptions for their ``Array``-managed fields.
Their public field attributes give numerical NumPy arrays; the corresponding
``Array`` handles provide unit and description metadata. The fields and their
handles both provide explicit Astropy quantity access through ``get_quantity()``.

Inspecting Field Metadata
-------------------------

The examples below use a ``Coeval`` result named ``coeval``, such as one produced
in :doc:`coeval_cubes`. Access the neutral-fraction data and its metadata::

    >>> from astropy import units as u
    >>> values = coeval.ionized_box.neutral_fraction
    >>> neutral = coeval.ionized_box.arrays["neutral_fraction"]
    >>> neutral.unit == u.dimensionless_unscaled
    True
    >>> neutral.description
    'Neutral hydrogen fraction.'

A dimensionless unit describes fractions, ratios, and other quantities without
a physical dimension. For example, the neutral hydrogen fraction can be
expressed as a percentage.

Converting Units
----------------

Use ``get_quantity()`` to attach the stored unit and ``to()`` to convert it::

    >>> neutral_quantity = values.get_quantity()
    >>> neutral_percent = neutral_quantity.to(u.percent)

Quantity operations carry units through a calculation. Raw NumPy operations on
``values`` operate on the stored numerical values without unit conversion.
Slices and derived raw arrays do not retain ``get_quantity()``; obtain a Quantity
before operations that should preserve units.

Large scaled quantities may overflow when converted using single precision.
For example, cast a scaled X-ray luminosity to ``float64`` before converting to
``erg/s``. The cast makes a copy::

    >>> import numpy as np
    >>> luminosity = u.Quantity([100.0], u.Unit(1e38 * u.erg / u.s), dtype=np.float32)
    >>> luminosity_cgs = luminosity.astype(np.float64).to(u.erg / u.s)
    >>> bool(np.isfinite(luminosity_cgs.value).all())
    True

Sharing and Copying Data
------------------------

By default, ``get_quantity()`` returns a read-only view sharing the field's
memory. It reflects subsequent changes to the underlying numerical array.
Use ``to()`` to obtain values in another unit, or request an independent,
writable quantity::

    >>> independent = coeval.ionized_box.neutral_fraction.get_quantity(copy=True)

Field access and ``Array.get_quantity()`` load purged data when needed and follow
the ``CACHE_ARRAYS_ON_ACCESS`` setting. Calling ``get_quantity()`` on an already
obtained field uses those values without another disk read. See :doc:`caching`
for the cache policy.

Assigning Values
----------------

Array setters accept raw numerical values in the field's stored unit. They do
not automatically convert a Quantity's units. First convert to stored values::

    >>> stored_values = neutral_percent.to_value(neutral.unit)

The conversion alone does not modify the field. Assign the converted values
to update its data::

    >>> coeval.ionized_box.neutral_fraction = stored_values

Obtain fresh handles or quantities after replacing a field; earlier ones can
still refer to its previous data.
