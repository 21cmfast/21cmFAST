Quickstart
==========

``21cmFAST`` can be used both as a Python library and from the command line as a
standalone application. In this quickstart guide we will briefly cover both use cases.
We have more involved `tutorials <tutorials.html>`_, `FAQs <faqs/misc.html>`_ and
`API Documentation <reference/py21cmfast.html>`_ for when you need more than this!

In this quickstart guide, we'll only cover the two most basic usages of ``21cmFAST``:
the all-in-one methods of simulating:

    * **Coeval** fields (i.e. 3D periodic boxes containing various physical fields, like
      the 21cm brightness temperature and ionization fraction); and
    * **Lightcones**, which stitch together multiple coeval fields as they evolve over
      cosmic history into one "observable" lightcone.

The high-level entry-points for creating these two kinds of simulations (which exist
both in the library and on the CLI) are *probably* all you'll ever need, with all their
associated options. If you're doing something that requires more careful management of
the redshift evolution and memory management, then you'll need the lower-level functions
that we describe in the `tutorials <tutorials.html>`_, and
`API Documentation <reference/py21cmfast.html>`_.

Using the Python Library
------------------------

Running a Coeval Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's go through the most basic example of running a (very small) coeval simulation at
a given redshift, and plotting an image of a slice through it.

First, import the library::

    >>> import py21cmfast as p21c

We use ``p21c`` as a standard convention. Then, setup your simulation parameters::

    >>> inputs = p21c.InputParameters.from_template('simple-small', random_seed=1234)

The ``inputs`` created here contain a large number of simulation parameters, including
options like the size of the box and its resolution, feature flags to toggle various
physical models on and off, and astrophysical and cosmological parameter values.
Here, we used the simplest method of setting up parameters: starting with a built-in
template. See the `Running and Plotting Coeval Cubes <tutorials/coeval_cubes.html>`_
tutorial for more information on how to setup your parameters.

Now, run the simulation::

    >>> coevals = p21c.run_coeval(inputs=inputs, out_redshifts=[8.0])

This will simply run the simulation at a redshift of 8.0 with the given parameters.
You can run it at as many redshifts as you like.

The objects returned (as a list) are :class:`py21cmfast.drivers.coeval.Coeval` instances,
from which you can access the 3D arrays corresponding to various physical fields.
To plot a single 2D slice of one of these fields::

    >>> p21c.plotting.coeval_sliceplot(coevals[0], kind='brightness_temp')

The ``coeval`` object here has much more than just the ``brightness_temp`` field in it.
You can plot the (matter) ``density`` field, ``velocity`` field or a number of other
fields.

Simulating a Lightcone
~~~~~~~~~~~~~~~~~~~~~~

To simulate a full lightcone, we first setup a configuration for constructing the
lightcone from the series of coeval boxes over redshift. First, since our simulation
configuration is very simple, it does not intrinsically require redshift evolution, and
therefore does not contain information about the redshifts forming the "nodes" of the
simulation (other sets of parameters do have this information intrinsically, and will
not require the following step), so we add that information::

    >>> inputs = inputs.with_logspaced_redshifts(zmin=6.0, zmax=25.0, zstep_factor=1.1)

The default lightcone configuration constructs a rectilinear lightcone, where each 3D
voxel has the same volume in comoving volume units::

    >>> lc_cfg = p21c.RectilinearLightconer.between_redshifts(
    >>>     min_redshift=6.0,
    >>>     max_redshift=25.0,
    >>>     resolution=inputs.simulation_options.cell_size,
    >>>     quantities=['brightness_temp'],
    >>> )

Note that we *also* had to specify the min/max redshift here -- while it might seem like
we should just know this information from the ``inputs``, in many cases you want your
*physics* to evolve over a broader range of cosmic history than you want to eventually
store in your lightcone. THe ``quantities`` above define which fields end up being
turned into lightcones (any field can be added, but by default only the 21cm
brightness temperature is constructed). Finally, we simply run the lightcone:

    >>> lc = p21c.run_lightcone(
    >>>     lightconer=lc_cfg,
    >>>     inputs=inputs,
    >>> )

And we can make a 2D plot of a slice along the line-of-sight axis of the lightcone::

    >>> p21c.plotting.lightcone_sliceplot(lc)


Running Simulations Using the CLI
---------------------------------
The CLI can be used to generate boxes on-disk directly from a configuration file or
command-line parameters. More details on using the CLI can be found at our
`CLI tutorial <tutorials/cli_usage.html>`_ or by using the standard ``--help`` option with any
sub-command, for example::

    $ 21cmfast --help  # help on what sub-commands are available
    $ 21cmfast run coeval --help  # help specifically for running a coeval box.

Running a Coeval Simulation from the CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To run a coeval simulation, use the following as an example::

    $ 21cmfast run coeval --redshifts 8.0 -z 10.0 \  # specify multiple redshifts
        --template simple-small \                    # base your simulation on the simple-small template
        --out . --cachedir cache \                   # configure outputs and cache
        --sigma-8 0.85 --perturb-on-high-res         # simulations params that override the template

Here, the ``--redshifts`` argument can be specified as many times as you want, and it
has a "short" alias of ``-z`` for convenience. The ``--template`` argument specifies a base
template from which to build your simulation configuration, and must be a name of one
of the builtins (there are CLI commands for listing available templates as well, see
the `tutorial <tutorials/cli_usage.html>`_).

There are two arguments for where to store outputs.
The main one is ``--out`` which is a directory inside of which will be written a number
of files with names like ``coeval_z8.00.h5`` and ``coeval_z10.00.h5``. These are the only
high-level output files of the simulation, and they are self-contained (i.e. they
contain all the parameters used to run the simulation, and all the 3D fields that
were simulated). The ``--cachedir`` is the directory where intermediate files will be
stored during the simulation. Set this to be a temporary directory if you are not
planning on using these files (they can sometimes be used in later simulations
to speed them up, but you need to know what you're doing). The default for both of these
options is the current working directory.

Finally, you can over-ride the parameter template by directly passing any simulation
parameter. Because the list of parameters is very long, we don't list them when you
call ``21cmfast run coeval --help``. To list them all, use ``21cmfast run params --help``.
They are also all listed in the `API Documentation <reference/_autosummary/py21cmfast.wrapper.inputs.html>`_,
though on the CLI they are normalized so that only hyphens are used, not underscores,
and all names are lower-case.

Running a lightcone from the CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The process here is very similar to running a ``coeval`` as described above, with only
a couple of differences::

    $ 21cmfast run lightcone \
        --redshift-range 6.0 12.0 \           # redshift range instead of specific redshifts
        --template simple-small \             # base your simulation on the simple-small template
        --out lightcone.h5 --cachedir cache \ # configure outputs and cache
        --sigma-8 0.85 --perturb-on-high-res\ # simulations params that override the template
        --lq brightness_temp --lq neutral_fraction \  # fields that become lightcones
        --gq brightness_temp --gq kinetic_temperature # fields to compute as globally-averaged

The major differences here are that:

1. Instead of setting specific redshifts, we specify a range of redshifts and let the
   algorithm decide how to evolve within that range.
2. The output here is a single file, not one file per redshift, so we specify exactly
   the file, rather than a directory for ``--out``
3. Note that since ``--cachedir`` is the same here as it was when we ran ``coeval``, many
   of the boxes here would not be resimulated, but instead just read from disk.
4. We can pass multiple ``--lq`` (or more verbosely, ``--lightcone-quantities``) to specify
   the physical fields we want written out as lightcones. The default is to save only
   the 21cm brightness temperature.
5. We can pass multiple ``--gq`` (or more verbosely, ``--global-quantities``) to specify
   the fields we want to save as globally-averaged values as a function of redshift.
   The default is to save the 21cm brightness temperature and the neutral fraction
   of hydrogen.

There are many more options, so make sure to read the full
`CLI tutorial <tutorials/cli_usage.html>`_.
