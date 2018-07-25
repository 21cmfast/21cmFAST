"""
A module providing Core Modules for cosmoHammer. This is the basis of the plugin system for py21cmmc.
"""
import warnings
import py21cmmc as p21


class CoreCoevalModule:
    def __init__(self, redshifts,
                 user_params=None, flag_options=None, astro_params=None,
                 cosmo_params=None, regenerate=True, do_spin_temp=False, z_step_factor=1.02,
                 z_heat_max=None, **io_options):

        self.redshifts = redshifts

        self.user_params = p21.UserParams(user_params)
        self.flag_options = p21.FlagOptions(flag_options)
        self.astro_params = p21.AstroParams(astro_params)
        self.cosmo_params = p21.CosmoParams(cosmo_params)

        self.regenerate = regenerate

        self.z_step_factor = z_step_factor
        self.z_heat_max = z_heat_max
        self.do_spin_temp = do_spin_temp

        self.io = dict(
            store=[],            # (derived) quantities to store in the MCMC chain.
            cache_dir=None,      # where full data sets will be written/read from.
            cache_init=True,     # whether to cache init and perturb data sets (done before parameter retention step).
            cache_ionize=False,  # whether to cache ionization data sets (done before parameter retention step)
        )

        self.io.update(io_options)

        self.initial_conditions = None
        self.perturb_field = None
        # self._modifying_cosmo = False

    def setup(self):

        self.parameter_names = getattr(self.LikelihoodComputationChain.params,"keys", None)
        if self.parameter_names is not None:
            self.parameter_names = self.parameter_names()
        else:
            self.parameter_names = []

        if self.z_heat_max is not None:
            p21.global_params.Z_HEAT_MAX = self.z_heat_max

        # Here we initialize the init and perturb boxes.
        # If modifying cosmo, we don't want to do this, because we'll create them
        # on the fly on every iteration.
        if not any([p in p21.CosmoParams.pydict.keys() for p in self.parameter_names]):
            print("Initializing init and perturb boxes for the entire chain...", end='', flush=True)
            self.initial_conditions = p21.initial_conditions(
                user_params=self.user_params,
                cosmo_params=self.cosmo_params,
                write=self.io['cache_init'],
                direc=self.io['cache_dir'],
                regenerate=self.regenerate
            )

            self.perturb_field = []
            for z in self.redshifts:
                self.perturb_field += [p21.perturb_field(
                    redshift=z,
                    init_boxes=self.initial_conditions,
                    write=self.io['cache_init'],
                    direc=self.io['cache_dir'],
                    regenerate=self.regenerate
                )]
            print(" done.")

    def __call__(self, ctx):
        # Update parameters
        astro_params, cosmo_params = self._update_params(ctx.getParams())

        # Call C-code
        init, perturb, xHI, brightness_temp = self.run(astro_params, cosmo_params)

        ctx.add('brightness_temp', brightness_temp)
        ctx.add("init", init)
        ctx.add("perturb", perturb)
        ctx.add("xHI", xHI)

    def prepare_storage(self, ctx, storage):
        "Add variables to special dict which cosmoHammer will automatically store with the chain."
        for k in self.io['store']:
            bits = k.split(".")
            val = None

            # The first bit must come from either the context or the overall object.
            if ctx.contains(bits[0]):
                val = ctx.get(bits[0])
            else:
                try:
                    val = getattr(self, bits[0])
                except AttributeError:
                    warnings.warn("Cannot find variable %s to store" % k)

            # Now go through each bit
            for bit in bits[1:]:
                try:
                    val = getattr(val, bit)
                except AttributeError:
                    warnings.warn("Cannot find variable %s to store" % k)

            if val is not None:
                storage[k] = val

    def _update_params(self, params):
        """
        Update all the parameter structures which get passed to the driver, for this iteration.

        Parameters
        ----------
        params : Parameter object from cosmoHammer
        """
        apkeys = self.astro_params.defining_dict # Remember this does not have any SEED
        cpkeys = self.cosmo_params.defining_dict

        # TODO: should explore whether this would work just with .update()... not sure if it will screw
        # up multiple processes...

        # Update the Astrophysical/Cosmological Parameters for this iteration
        for k in params.keys():
            if k in self.astro_params.pydict:
                apkeys[k] = getattr(params, k)
            elif k in self.cosmo_params.pydict:
                cpkeys[k] = getattr(params, k)
            else:
                raise ValueError("Key %s is not in AstroParams or CosmoParams " % k)

        return p21.AstroParams(**apkeys), p21.CosmoParams(**cpkeys)

    def run(self, astro_params, cosmo_params):
        """
        Actually run the 21cmFAST code.
        """
        init, perturb, xHI, brightness_temp = p21.run_coeval(
            redshift=self.redshifts,
            astro_params=astro_params, flag_options=self.flag_options,
            cosmo_params=cosmo_params, user_params=self.user_params,
            perturb=self.perturb_field,
            init_box=self.initial_conditions,
            do_spin_temp=self.do_spin_temp,
            z_step_factor=self.z_step_factor,
            regenerate=self.regenerate,
            write=self.io['cache_ionize'],
            direc=self.io['cache_dir'],
            match_seed=True
        )

        return init, perturb, xHI, brightness_temp
