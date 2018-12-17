"""
A module providing Core Modules for cosmoHammer. This is the basis of the plugin system for py21cmmc.
"""
import logging
import warnings

import py21cmmc as p21

logger = logging.getLogger("21CMMC")


class NotSetupError(AttributeError):
    def __init__(self):
        default_message = 'setup() must have been called on the chain to use this method/attribute!'
        super().__init__(default_message)


class NotAChain(AttributeError):
    def __init__(self):
        default_message = 'this Core or Likelihood must be part of a LikelihoodComputationChain to enable this method/attribute!'
        super().__init__(default_message)


class CoreBase:
    def __init__(self, store=None):
        self.store = store or {}

    def prepare_storage(self, ctx, storage):
        """Add variables to special dict which cosmoHammer will automatically store with the chain."""
        for name, storage_function in self.store.items():
            try:
                storage[name] = storage_function(ctx)
            except Exception:
                logger.error("Exception while trying to evaluate storage function %s" % name)
                raise

    @property
    def chain(self):
        """
        A reference to the LikelihoodComputationChain of which this Core is a part.
        """
        try:
            return self._LikelihoodComputationChain
        except AttributeError:
            raise NotAChain

    @property
    def parameter_names(self):
        return getattr(self.chain.params, "keys", [])

    def simulate_data(self, ctx):
        """
        Passed a standard context object, should simulate data and place it in the context.

        This is to be used as a way to simulate mocks, rather than constructing forward models, though sometimes the
        two can be interchanged.

        Parameters
        ----------
        ctx : dict-like
            The context, from which parameters are

        Returns
        -------
        dct : dict
            A dictionary of data which was simulated.
        """
        pass

    def __call__(self, ctx):
        """
        Call the class. By default, it will just simulate data.
        """
        self.simulate_data(ctx)


class CoreCoevalModule(CoreBase):
    """
    A Core Module which evaluates coeval cubes at given redshift.

    On each iteration, this module will add to the context:

    1. ``init``: an :class:`~py21cmmc._21cmfast.wrapper.InitialConditions` instance
    2. ``perturb``: a :class:`~py21cmmc._21cmfast.wrapper.PerturbedField` instance
    3. ``xHI``: an :class:`~py21cmmc._21cmfast.wrapper.IonizedBox` instance
    4. ``brightness_temp``: a :class:`~py21cmmc._21cmfast.wrapper.BrightnessTemp` instance
    """

    def __init__(self, redshift,
                 user_params=None, flag_options=None, astro_params=None,
                 cosmo_params=None, regenerate=True, do_spin_temp=False, z_step_factor=1.02,
                 z_heat_max=None, change_seed_every_iter=False, ctx_variables=None,
                 keep_data_in_memory=True, initial_conditions_seed=None,
                 **io_options):
        """
        Initialize the class.

        .. note:: None of the parameters provided here affect the *MCMC* as such; they merely provide a background
                  model on which the MCMC will be performed. Thus for example, passing `HII_EFF_FACTOR=30` in
                  `astro_params` here will be over-written per-iteration if `HII_EFF_FACTOR` is also passed as a
                  `parameter` to an MCMC routine using this Core Module.

        Parameters
        ----------
        redshift : float or array_like
             The redshift(s) at which to evaluate the coeval cubes.
        user_params : dict or :class:`~py21cmmc._21cmfast.wrapper.UserParams`
            Parameters affecting the overall dimensions of the cubes (see :class:`~py21cmmc._21cmfast.wrapper.UserParams`
            for details).
        flag_options : dict or :class:`~py21cmmc._21cmfast.wrapper.FlagOptions`
            Options affecting choices for how the reionization is calculated (see
            :class:`~py21cmmc._21cmfast.wrapper.FlagOptions` for details).
        astro_params : dict or :class:`~py21cmmc._21cmfast.wrapper.AstroParams`
            Astrophysical parameters of reionization (see :class:`~py21cmmc._21cmfast.wrapper.AstroParams` for details).
        cosmo_params : dict or :class:`~py21cmmc._21cmfast.wrapper.CosmoParams`
            Cosmological parameters of the simulations (see :class:`~py21cmmc._21cmfast.wrapper.CosmoParams` for
            details).
        regenerate : bool, optional
            Whether to force regeneration of simulations, even if matching cached data is found.
        do_spin_temp: bool, optional
            Whether to use spin temperature in the calculation, or assume the saturated limit.
        z_step_factor: float, optional
            How large the logarithmic steps between redshift are (if required).
        z_heat_max: float, optional
            Controls the global `Z_HEAT_MAX` parameter, which specifies the maximum redshift up to which heating sources
            are required to specify the ionization field. Beyond this, the ionization field is specified directly from
            the perturbed density field.
        ctx_variables : list of str, optional
            A list of strings, any number of the following: "brightness_temp", "init", "perturb", "xHI". These each
            correspond to an OutputStruct which will be stored in the context on every iteration. Omitting as many as
            possible is useful in that it reduces the memory that needs to be transmitted to each process. Furthermore,
            in-built pickling has a restriction that arrays cannot be larger than 4GiB, which can be easily over-run
            when passing the hires array in the "init" structure.
        keep_data_in_memory : bool, optional
            This flag controls whether the underlying datasets ``initial_conditions`` and ``perturb_field`` are stored
            in memory as part of this object, or whether they are merely read-in from disk cache on each iteration.
            The former is only possible if these underlying datasets are unchanging with the selected MCMC parameters
            (i.e. if there are no cosmological parameters being constrained), and it should be faster. However, if the
            simulations are large, an error may arise if attempting to keep the data in memory, as this memory must
            be pickled and sent to different processes, which fails for datasets larger than 4GiB. This Core will
            attmept to automatically switch this parameter to False if the dataset is this large.

        Other Parameters
        ----------------
        store :  dict, optional
            The (derived) quantities/blobs to store in the MCMC chain, default empty. See Notes below for details.
        cache_dir : str, optional
            The directory in which to search for the boxes and write them. By default, this is the directory given by
            ``boxdir`` in the configuration file, ``~/.21CMMC/config.yml``. Note that for *reading* data, while the
            specified `direc` is searched first, the default directory will *also* be searched if no appropriate data is
            found in `direc`.
        cache_init : bool, optional
            Whether to cache init and perturb data sets, if cosmology is static. This is done before the parameter
            retention step of an MCMC; i.e. before deciding whether to retain the current set of parameters given
            the previous set, which can be useful in diagnosis. Default True.
        cache_ionize : bool, optional
            Whether to cache ionization data sets (done before parameter retention step). Default False.


        Notes
        -----
        The ``store`` keyword is a dictionary, where each key specifies the name of the resulting data entry in the
        samples object, and the value is a callable which receives the ``context``, and returns a value from it.

        This means that the context can be inspected and arbitrarily summarised before storage. In particular, this
        allows for taking slices of arrays and saving them. One thing to note is that the context is dictionary-like,
        but is not a dictionary. The elements of the context are only available by using the ``get`` method, rather than
        directly subscripting the object like a normal dictionary.

        .. note:: only scalars and arrays are supported for storage in the chain itself.
        """

        super().__init__(io_options.get("store", None))

        if ctx_variables is None:
            ctx_variables = ["brightness_temp", "xHI"]

        self.redshift = redshift
        if not hasattr(self.redshift, "__len__"):
            self.redshift = [self.redshift]

        self.user_params = p21.UserParams(user_params)
        self.flag_options = p21.FlagOptions(flag_options)
        self.astro_params = p21.AstroParams(astro_params)
        self.cosmo_params = p21.CosmoParams(cosmo_params)
        self.change_seed_every_iter = change_seed_every_iter
        self.initial_conditions_seed = initial_conditions_seed

        self.regenerate = regenerate
        self.ctx_variables = ctx_variables

        self.z_step_factor = z_step_factor
        self.z_heat_max = z_heat_max
        self.do_spin_temp = do_spin_temp

        self.io = dict(
            store={},  # (derived) quantities to store in the MCMC chain.
            cache_dir=None,  # where full data sets will be written/read from.
            cache_init=True,  # whether to cache init and perturb data sets (done before parameter retention step).
            cache_ionize=False,  # whether to cache ionization data sets (done before parameter retention step)
        )

        self.io.update(io_options)

        self.initial_conditions = None
        self.perturb_field = None

        # Attempt to auto-set the keep_memory parameter
        if keep_data_in_memory and self.user_params.DIM >= 1000:  # This gives about 4GiB for 32-bit floats.
            self.keep_data_in_memory = False
        else:
            self.keep_data_in_memory = keep_data_in_memory

        if self.initial_conditions_seed and self.change_seed_every_iter:
            logger.warning(
                "Attempting to set initial conditions seed while desiring to change seeds every iteration. Unsetting initial conditions seed.")
            self.initial_conditions_seed = None

    def setup(self):
        """
        Perform setup of the core.

        Notes
        -----
        This method is called automatically by its parent :class:`~LikelihoodComputationChain`, and should not be
        invoked directly.
        """
        # If the chain has different parameter truths, we want to use those for our defaults.
        self._update_params(self.chain.createChainContext().getParams())

        if self.z_heat_max is not None:
            p21.global_params.Z_HEAT_MAX = self.z_heat_max

        # Here we initialize the init and perturb boxes.
        # If modifying cosmo, we don't want to do this, because we'll create them
        # on the fly on every iteration.
        if not any(
                [p in self.cosmo_params.self.keys() for p in self.parameter_names]) and not self.change_seed_every_iter:
            logger.info("Initializing init and perturb boxes for the entire chain.")
            initial_conditions = p21.initial_conditions(
                user_params=self.user_params,
                cosmo_params=self.cosmo_params,
                write=self.io['cache_init'],
                direc=self.io['cache_dir'],
                regenerate=self.regenerate,
                random_seed=self.initial_conditions_seed
            )

            # update the seed
            self.initial_conditions_seed = initial_conditions.random_seed

            perturb_field = []
            for z in self.redshift:
                perturb_field += [p21.perturb_field(
                    redshift=z,
                    init_boxes=initial_conditions,
                    write=self.io['cache_init'],
                    direc=self.io['cache_dir'],
                    regenerate=self.regenerate,
                )]
            logger.info("Initialization done.")

            if self.keep_data_in_memory:
                self.initial_conditions = initial_conditions
                self.perturb_field = perturb_field

    def simulate_data(self, ctx):
        # Update parameters
        self._update_params(ctx.getParams())

        # Call C-code
        init, perturb, xHI, brightness_temp = self.run(self.astro_params, self.cosmo_params)

        for key in self.ctx_variables:
            try:
                ctx.add(key, locals()[key])
            except KeyError:
                raise KeyError(
                    "ctx_variables must be drawn from the list ['init', 'perturb', 'xHI', 'brightness_temp']")

    def _update_params(self, params):
        """
        Update all the parameter structures which get passed to the driver, for this iteration.

        Parameters
        ----------
        params : Parameter object from cosmoHammer

        """
        # Note that RANDOM_SEED is never updated. It should only change when we are modifying cosmo.
        self.astro_params.update(
            **{k: getattr(params, k) for k, v in params.items() if k in self.astro_params.defining_dict})
        self.cosmo_params.update(
            **{k: getattr(params, k) for k, v in params.items() if k in self.cosmo_params.defining_dict})

    def run(self, astro_params, cosmo_params):
        """
        Actually run the 21cmFAST code.
        """

        return p21.run_coeval(
            redshift=self.redshift,
            astro_params=astro_params, flag_options=self.flag_options,
            cosmo_params=cosmo_params, user_params=self.user_params,
            perturb=self.perturb_field,
            init_box=self.initial_conditions,
            do_spin_temp=self.do_spin_temp,
            z_step_factor=self.z_step_factor,
            regenerate=self.regenerate or self.change_seed_every_iter,
            random_seed=self.initial_conditions_seed,
            write=self.io['cache_ionize'],
            direc=self.io['cache_dir'],
        )


class CoreLightConeModule(CoreCoevalModule):
    """
    Core module for evaluating lightcone simulations.

    See :class:`~CoreCoevalModule` for info on all parameters, which are identical to this class, with the exception
    of `redshift`, which in this case must be a scalar.

    This module will add the following quantities to the context:

    1. ``lightcone``: a :class:`~py21cmmc._21cmfast.wrapper.LightCone` instance.
    """

    def __init__(self, *, max_redshift, **kwargs):
        if "ctx_variables" in kwargs:
            warnings.warn(
                "ctx_variables does not apply to the lightcone module (at least not yet). It will be ignored.")

        super().__init__(**kwargs)
        self.max_redshift = max_redshift

    def setup(self):
        super().setup()

        # Un-list redshift and perturb
        self.redshift = self.redshift[0]
        if self.perturb_field is not None:
            self.perturb_field = self.perturb_field[0]

    @property
    def lightcone_slice_redshifts(self):
        """
        The redshift at each slice of the lightcone.
        """
        # noinspection PyProtectedMember
        return p21.wrapper._get_lightcone_redshifts(
            self.cosmo_params, self.max_redshift, self.redshift,
            self.user_params, self.z_step_factor
        )

    def simulate_data(self, ctx):
        # Update parameters
        self._update_params(ctx.getParams())

        # Call C-code
        lightcone = self.run(self.astro_params, self.cosmo_params)

        ctx.add('lightcone', lightcone)

    def run(self, astro_params, cosmo_params):
        """
        Actually run the 21cmFAST code.
        """
        return p21.run_lightcone(
            redshift=self.redshift,
            max_redshift=self.max_redshift,
            astro_params=astro_params, flag_options=self.flag_options,
            cosmo_params=cosmo_params, user_params=self.user_params,
            perturb=self.perturb_field,
            init_box=self.initial_conditions,
            do_spin_temp=self.do_spin_temp,
            z_step_factor=self.z_step_factor,
            regenerate=self.regenerate,
            write=self.io['cache_ionize'],
            direc=self.io['cache_dir'],
        )
