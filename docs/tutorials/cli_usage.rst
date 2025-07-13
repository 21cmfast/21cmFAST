Using the CLI
=============

In this mini tutorial we outline the features of the CLI, and various common use-cases.
We recommend first reading `Quickstart <../quickstart.html>`_ guide to get a high-level
feel for how to use the CLI.

General Principles
------------------

Our CLI is built with the excellent `cyclopts <https://cyclopts.readthedocs.io>`_
package. As such, it has a number of features. The first is that all the parameters of
a particular command are described by using the `--help` command. For example::

    $ 21cmfast --help

    Usage: 21cmfast COMMAND

    ╭─ Commands ──────────────────────────────────────────────────────╮
    │ dev        Run development tasks.                               │
    │ run        Run 21cmFAST simulations.                            │
    │ template   Manage 21cmFAST configuration files/templates.       │
    │ --help -h  Display this message and exit.                       │
    │ --version  Display application version.                         │
    ╰─────────────────────────────────────────────────────────────────╯

This prints out the available commands with a short description.


Managing Templates and Parameter Configurations
-----------------------------------------------
While it is possible to pass specific simulation parameters to ``21cmfast run`` commands,
it is generally a better idea to run your simulation directly from a specific
set-in-stone parameter file, to enhance reproducibility. We provide a number of commands
and options to view and manage such files.

To be clear, the parameter files we're talking about here include only parameters that
affect the physical output of the simulation (box sizes, astrophysical and cosmological
parameters, flags for toggling physical modules etc) *not* options for how you run
this particular instance of the simulation (e.g. with or without a progress bar).

Our configuration file format is TOML, and you can just sit down and write
one yourself, if you like. However, the easiest way to create a new configuration file
is by starting with a built-in template, of which we have several. To see all the available
built-in templates, use the command::

    $ 21cmfast template avail

You can view the parameters of any of the builtin templates with the ``show`` command::

    $ 21cmfast template show latest-dhalos

By default, this will display *all* of the parameters of that model. To only display
the non-default parameters::

    $ 21cmfast template show latest-dhalos --mode minimal

Each of these built-in templates is itself a TOML config file, but it's better not to
mess with them directly. To create a new parameter TOML that is exactly the same as
an existing template, use the `create` command::

    $ 21cmfast template create --template simple-small --out my-simple-small.toml

This creates a new template TOML ``my-simple-small.toml`` which lists the values of
**all** available parameters, and is functionally identical to the `simple-small`
built-in template.

To over-ride particular settings, simply add them to the command as options::

    $ 21cmfast template create --template simple-small --out my-custom-template.toml \
        --hii-dim 50 --box-len 100 --use-ts-fluct

These options are precisely the names of input parameters which are all listed in the
`API Documentation <../reference/_autosummary/py21cmfast.wrapper.inputs.html>`_, with the
caveat that they are fully lower-case and substitute underscores for hyphens
(which is standard for CLI's). To get a list of available parameters and their
descriptions, you can run::

    $ 21cmfast run params --help

Specifying Parameters for Simulations
-------------------------------------

Running simulations from the command line is always achieved through the ``21cmfast run``
commmand. All sub-commands of ``run`` have the same methods of setting the
simulation parameters. In this section of the tutorial we will use the ``ics`` sub-command
to illustrate the options for setting parameters for simulations, since it the simplest
sub-command.

The simplest way to specify parameters (but not the best, see below!) is by using one
of the built-in templates. In the simplest case, you just do::

    $ 21cmfast run ics --template simple-small

To override specific simulation parameters on top of this base template, simply pass
them as options, for example::

    $ 21cmfast run ics --template simple-small --use-ts-fluct --sigma-8 1.0

However, while overriding parameters like this is convenient for simple one-off
tests, it is generally better to run your simulations from a fully-specific parameter
configuration TOML (see above), becasue that allows you to more easily reproduce your
results at a later time (and to share the configurtion with others). The recommended
way of achieving this is to first construct a parameter TOML, and then to pass that
to the `run` command, like so::

    $ 21cmfast template create --template simple-small --use-ts-fluct --out custom.toml
    $ 21cmfast run ics --param-file custom.toml

This two-step process is more explicit and allows you to share ``custom.toml`` for
reproducibility. Even when passing ``--param-file``, you may opt to override specific
parameters::

    $ 21cmfast run ics --param-file custom.toml --perturb-on-high-res

Again, doing so is generally not a good idea, but can be useful for quick explorations.

In summary, you have three ways to specify parameters: via ``--template``, ``--param-file``
and explicit parameters. We encourage using *only* ``--param-file``, but it's always
possible to use *either* ``--template`` or ``--param-file`` in conjunction with
explicit parameter overrides. However, you *must* always specific one *and only one* of
``--template`` or ``--param-file``.

One final thing. Whenever you use ``21cmfast run``, a fully-specific parameter TOML will
be automatically created for you, consistent with all of the parameters of your simulation
(after consideration of all of ``--template``, ``--param-file`` and explicit params).
This will be saved in your ``--cachedir`` (by default, the current working directory,
see below) and be named according to the following rules:

1. If you passed ``--param-file`` and no explicit params, no new file will be written,
   regardless of any of the following.
2. If you passed ``--cfgfile <path.toml>`` then it will be saved to ``<path.toml>``
3. If you only passed ``--template <name>`` and no explicit params, it will be called
   ``<name>.toml``. In effect, this TOML is the same specification as the built-in TOML,
   however the built-ins are generally minimally-specified (i.e. they rely on the
   default parameters of ``21cmFAST`` to fill in missing parameters) while the output
   here will be fully-specified.
4. If you pass any explicit parameters, regardless of whether these are building on a
   ``--template`` or ``--param-file``, the file will be called ``config-<uuid>.toml``,
   where the ``uuid`` is a 6-character random string ensuring that you don't overwrite
   previous configurations. The output file will be printed to screen as part of the
   run, so you will know what it is.

This way, you can also ensure reproducibility of your simulation by sharing this output
TOML. However, it's still better to control the TOML yourself by creating it explicitly
with ``21cmfast template create``.

Managing Simulation Outputs and Cache
-------------------------------------

There are two kinds of outputs that ``21cmfast run`` can create. The "primary" outputs
are the ``Coeval`` boxes and ``LightCone`` files, which are the end-products of the
simulations. These are saved according to the ``--out`` parameter, but they behave a little
differently depending on the simualation:

1. For ``21cmfast run coeval`` the ``--out <direc>`` parameter specifies a *directory*,
   and the coeval boxes are written to ``out/coeval_z<redshift>.h5``.
2. For ``21cmfast run lightcone`` the ``--out <path.h5>`` parameter specifies an output
   *file*, and there is only lightcone file created.

The other kind of output is the cache. The way that ``21cmFAST `` works is that it
simulates several kinds of physical fields that build on each other. Each step of this
process can be written to file. These files can be used for three purposes:

1. Internally, within e.g. ``run_coeval()``, we can use the cache to offload data from
   memory temporarily, so it can be read back in as necessary as the simualation evolves.
2. If a simulation is halted for any reason, upon re-running the simualtion, the existence
   of the cache means that those boxes will not need to be re-run, speeding up the
   re-simulation.
3. If running a new simulation with some different parameters, there are certain parts
   of previous simulations that may be re-usable (often, this will be the
   ``InitialConditions`` and ``PerturbedField``). If you point to the same cache, these
   will be re-used instead of re-simulated, saving time.

While in principle the cache does not need to be used at all, in the most recent models
it is highly encouraged to use the cache for the purposes of reducing peak memory usage.
You can manage where  the cache is written with the ``--cachedir`` option.
By default it is set to the *current working directory*.
If you don't want to keep the cache around long-term, you can set it to a temporary
directory, for example::

    $ 21cmfast run coeval -z 8.0 --template simple-small --cachedir /tmp/21cmfast-cache

Note that by default, the fully-specified parameter TOML that is automatically output
by any ``run`` command is saved into the ``--cachedir``.

To change which field types are cached, use the ``--cache-strategy`` parameter (note
that this only affects the ``coeval`` and ``lightcone`` commands, not the ``ics``).
By default this is set to ``dmfield``, which caches the initial conditions, perturbed
matter fields, and perturbed halo fields (if applicable). Since all later boxes depend
on these fields, and these fields are pre-computed at **all** redshifts before any of the
astrophysics, it is generally advantageous to cache these. You can ensure all fields are
cached by passing ``--cache-strategy on``, and opt to cache nothing with
``--cache-strategy off``. Finally, you can optimize the tradeoff between disk usage
and memory usage by using ``--cache-strategy last_step_only``, which only caches boxes
that are required for more than just the next step.

.. note:: All cache files are stored inside sub-directories of the ``--cachedir``
          which are named uniquely via hashing the input parameters. This is not meant
          to be human-readable. You can run **multiple simulations** with different
          parameters pointing to the same ``--cachedir`` -- they will not interfere with
          each other, and in fact, you may get the benefit of reducing unnecessary
          recalculation!

.. note:: In the special case of ``21cmfast run ics`` the only output is the
          ``InitialConditions.h5`` file, which is normally a part of the internal cache.
          Thus, there is no ``--out`` parameter to this command, and the only "output"
          will be in ``<cachedir>/<param_hash>/<seed>/InitialConditions.h5``. The
          precise location of this file is only determined at run-time, and will be
          printed to stdout so you can locate it.

Defining Redshifts and Evolution
--------------------------------

When running either `run coeval` or `run lightcone`, you will need to specify the
redshifts of interest. This can be a little more subtle than you might expect, so here
we describe the ways you can do this, and the difference between the output redshifts
and the internal redshifts used for evaluating cosmic evolution.

The fundamental outputs of ``21cmFAST`` are 3D coeval fields -- that is, 3D periodic boxes
representing the value of various physical fields at a set cosmic time/redshift.
Sometimes, one is directly interested in such an output, though we can never actually
observe such a field. What we *observe* is a 3D *lightcone*, where each 2D slice corresponds
to a set of angular coordinates at a particular redshift, and redshift/distance/time
is changing for each slice. These lightcones have two "transverse" or "plane of the sky"
axes, and one "line of sight" or "redshift" axis.

Back to the point -- even though one is often interested in the lightcones, which can
be created with ``21cmfast run lightcone``, the fundamental outputs are still coeval boxes,
which are stitched together to obtain the lightcone.

Even though coeval boxes are defined at a particular redshift, it is often the case that
the state of the simulation at one particular redshift depends non-trivially on the
state at higher redshifts. That is, depending on the specific modules enabled,
``21cmFAST`` often needs to simulate the universe at a sequence of redshifts, starting
at high redshift and descending until it arrives at the redshift of interest. The
set of redshifts used in this physical evolution is called the ``node_redshifts``.

Separate from the ``node_redshifts``, which really define the simulation output itself,
are the "output" redshifts. For a ``coeval``, there will be one redshift per output that
defines the cosmic time of that particular snapshot. This redshift does not need to be
"on the grid" of ``node_redshifts`` -- it will be computed ad hoc based on the
evolutionary ``node_redshift`` grid. Conversely, for a ``lightcone``, we have a
*range* of redshifts -- one for each 2D slice -- which are constrained by being
incremented in regular intervals of *comoving distance*. The set of redshifts of each
slice does not need to match the ``node_redshifts`` (again, the ``node_redshifts``
define how the simulation is evolved, while these slice redshifts are simply
interpolated from that grid).

Specifying the ``node_redshifts``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ``coeval`` and ``lightcone`` runs the ``node_redshifts`` can be configured by the
following options:

1. ``--min-evolved-redshift`` (aliased to ``--zmin-evolution`` and ``--zmin``)
2. ``--zprime-step-factor``
3. ``--z-heat-max``

The resulting grid will be regular in ``log(1 + z)``, starting from exactly
``--min-evolved-redshift``, increasing by a geometric factor of ``--zprime-step-factor``
and ending *above* ``--z-heat-max``.

You do not need to specify any of these options for ``ics`` (though you *can* specify
both ``--zprime-step-factor`` and ``--z-heat-max``, they will not affect the hash
under which the output is stored).

For ``coeval`` and ``lightcone`` runs, all of the options have defaults. The default
of ``--min-evolved-redshift`` is 5.5, which covers all reasonable physical scenarios
where ``21cmFAST`` is well-specified.
The defaults of ``--zprime-step-factor`` and ``--z-heat-max`` depend on the template
that is being used, but are usually 1.02 and 35.0 respectively.

.. note:: ``21cmFAST`` in general does not enforce that the ``node_redshifts`` are
          geometrically-spaced, and if you use the library, you can specify any
          node redshifts that you like, so long as the maximum is greater than
          ``Z_HEAT_MAX``. However, a geometric redshift grid is close to optimal
          for standard cases, and so we currently enforce this from the CLI.

Output Redshifts for Coeval Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For  ``run coeval``, you can specify multiple specific redshifts like so::

    $ 21cmfast run coeval --param-file custom.toml --redshift 8.0 --redshift 10.0

This will create two output files, ``coeval_z8.00.h5`` and ``coeval_z10.00.h5``.
The ``--redshift`` argument is aliased to ``-z`` for convenience, so the following would
also work::

    $ 21cmfast run coeval --param-file custom.toml -z 8 -z 10

However, in the case that the simulation requires evolution over redshift, many coeval
boxes will be simulated, but only these two will be output. To have the other boxes
also written to file, use the ``--save-all-redshifts`` option (aliased to ``--all``)::

    $ 21cmfast run coeval --param-file custom.toml --use-ts-fluct -z 8 --all

.. note:: Even when ``--save-all-redshifts`` is not specified, the cache will hold the
    data for all ``node_redshifts``. Using ``--save-all-redshifts`` only affects what is
    output to the high-level output ``coeval.h5`` files.

Output Redshifts for Lightcones
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The set of redshifts at each 2D slice of the output lightcone are fully specified by
their range, which is defined by ``--redshift-range``. This is a two-element argument,
for example::

    $ 21cmfast run lightcone --param-file custom.toml --redshift-range 6 12

.. note:: The precise redshifts of each slice within this  ``--redshift-range`` are
    determined by enforcing that the slices are equidistant in comoving distance, with
    a resolution matching that of the underlying coeval simulations (i.e.
    ``BOX_LEN/HII_DIM``) and also that the highest-redshift slice is exactly at the
    highest ``node_redshift`` (any redshifts outside the ``--redshift-range`` are
    clipped, but they can be determined based on these).

.. warning:: An error will be raised if the ``--redshift-range`` doesn't fit inside the
    ``node_redshifts``.


Common Options when Running Simulations
---------------------------------------

You have the following options available to any subcommand of `run`, beyond those
already discussed above (all are optional, with defaults):

* ``--seed``: this specifies the random seed used to initialize the dark matter field,
  as well as potentially other stochasticity used in the simulation (depending on the
  modules being used). The seed is included in the cache so that simulations with
  different seeds are not mixed.
* ``--regenerate``: tell the simulator to regenerate all the boxes, even if they exist
  in the cache. This can be useful for testing, or if you recently upgraded ``21cmFAST``
  and expect results to change a little.
* ``--verbosity``: set how much info is printed to screen by the simulator. The options
  here are the standard logging levels (INFO, DEBUG, WARNING, etc).

Cookbook
--------

Here we outline some common usage patterns to make your life easier.

Temporary/Exploratory Coeval Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One use-case is to run off a coeval (or lightcone) just for exploratory purposes
(for example, to test that everything runs as expected, or to make a quick
comparison plot). It's often easiest to do this by starting with a builtin base template,
toggling the parameters you care about, and only keeping around the final result.

For example::

    $ 21cmfast run coeval \
        --template latest \                   # Latest model, without discrete halos
        --hii-dim 64 --dim 192 --box-len 96 \ # Over-ride particular parameters
        --redshift 6.0                        # At redshift 6.0
        --cachedir /tmp/21cmfast-cache        # Save cache to a temporary directory

This will run the latest model, but at a smaller size that you control, saving the output
coeval to the current directory, and storing the cache in a temporary directory so it is
removed automatically by your OS.

Running a single lightcone
~~~~~~~~~~~~~~~~~~~~~~~~~~

When running a single large-scale lightcone, it is best to be more careful about
reproducibility. A typical workflow might be something like the following.

First, check out the available built in templates to see which you might want to build
on::

    $ 21cmfast template avail

Let's say you chose to use the "latest" model, then you would go ahead and create your
custom parameter configuration based on this template::

    $ 21cmfast template create --template latest --hii-dim 512 --dim 1536 --box-len 768 --out big-latest.toml

Now there is a file ``big-latest.toml`` in your current directory. You can use this file
to run off your simulation::

    $ 21cmfast run lightcone --param-file big-latest.toml --redshift-range 5.6 25

You will get a file ``lightcone.h5`` as an output, which holds all the relevant information
of the simulation. Also, since the default cache directory is the current working
directory, you'll get a weird folder like ``a649nr0f6...`` in your current folder,
holding all the coeval fields from all ``node_redshifts``.

Running Multiple Simulations as a Database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the case that you have to run off many simulations from some distribution of
parameters, it is best to be a little more careful again about how you store your
cache. Let's imagine you were modifying only some astrophysical parameters, and
otherwise keeping the structure of the box, and the cosmology the same. This is a very
common situation.

We first make a directory to hold all of our cache, and our outputs::

    $ mkdir cache
    $ mkdir cache/configs
    $ mkdir lightcones

Then setup a "base" configuration::

    $ 21cmfast template create --template latest --hii-dim 512 --dim 1536 --box-len 768 --out cache/configs/base.toml

Now, before running off the other simulations, run off some initial conditions::

    $ 21cmfast run ics --param-file cache/configs/base.toml --seed 77577 --cachedir cache

We'll then have a folder ``cache/<ugly_hash>/77577`` in which will be an
``InitialConditions.h5`` file. Now we can start running our lightcones. In a real
application you may want to put this part into a script and run it via SLURM to
parallelize over the different parameters, but here we just show the basics::

    $ for zeta in 30.0 29.0 31.0 35.0          # iterate over all parameters
      do
        21cmfast run lightcone --param-file cache/config/base.toml \
          --seed 77577 --cachedir cache \      # need these to specify the same ICs
          --redshift-range 5.8 25 \            # specify redshift range
          --hii-eff-factor $zeta \             # override the astrophysical parameter
          --out lightcones/lc_zeta${zeta}.h5 \ # unique name of ligthcone output
          --cfgfile cache/configs/zeta${zeta}.h5  # unique configuration file
      done

This will result in four lightcones in the ``lightcones/`` directory, tagged with their
parameter values for ``HII_EFF_FACTOR``, and also four fully-specified parameter TOMLs,
along with all of the cache files required.
