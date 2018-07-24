"""
A module providing Core Modules for cosmoHammer. This is the basis of the plugin system for py21cmmc.
"""
import warnings
import py21cmmc as p21


class CoreCoEvalModule:
    # The following disables the progress bar for perturbing and ionizing fields.
    disable_progress_bar = True

    def __init__(self, parameter_names, redshifts, store = [],
                 user_params=p21.UserParams(), flag_options=p21.FlagOptions(), astro_params=p21.AstroParams(),
                 cosmo_params=p21.CosmoParams(), direc=".", regenerate=False, do_spin_temp=False, z_step_factor=1.02,
                 z_heat_max=None):

        # SETUP variables
        self.parameter_names = parameter_names
        self.store = store
        self.redshifts = redshifts

        # Save the following only as dictionaries (not full Structure objects), so that we can avoid
        # weird pickling errors.
        self.user_params = user_params
        self.flag_options = flag_options
        self.astro_params = astro_params
        self.cosmo_params = cosmo_params

        self.direc = direc
        self.regenerate = regenerate

        self.z_step_factor = z_step_factor
        self.z_heat_max = z_heat_max
        self.do_spin_temp = do_spin_temp

    def setup(self):

        self._regen_init = False
        self._write_init = True
        self._modifying_cosmo = False

        # The following ensures that if we are changing cosmology in the MCMC, then we re-do the init
        # and perturb_field parts on each iteration.
        for p in self.parameter_names:
            if p in lib.CosmoParams._defaults_.keys():
                self._write_init = False
                self._regen_init = True
                self._modifying_cosmo = True

        if self.z_heat_max is not None:
            p21.global_params.Z_HEAT_MAX = self.z_heat_max

        # Here we create the init boxes and perturb boxes, written to file.
        # If modifying cosmo, we don't want to do this, because we'll create them
        # on the fly on every iteration. We don't need to save any values in memory because
        # they will be read from file.
        if not self._modifying_cosmo:
            print("Initializing init and perturb boxes for the run...")
            self.initial_conditions = p21.initial_conditions(
                user_params=self.user_params,
                cosmo_params=self.cosmo_params,
                write=True, # TODO: maybe shouldn't write...?
                direc=self.direc,
                regenerate=self.regenerate
            )

            self.perturb_field = []
            for z in self.redshifts:
                self.perturb_field += [p21.perturb_field(
                    redshift = z,
                    init_boxes=self.initial_conditions,
                    write=True,
                    direc=self.direc,
                    regenerate=self.regenerate
                )]
        else:
            self.initial_conditions = None
            self.perturb_field = None

    def __call__(self, ctx):
        # Update parameters
        astro_params, cosmo_params = self._update_params(ctx)

        # Call C-code
        brightness_temp = self.run(astro_params, cosmo_params)

        ctx.add('brightness_temp', brightness_temp)
        ctx.add('user_params', self.user_params)
        ctx.add('cosmo_params', cosmo_params)
        ctx.add("astro_params", astro_params)
        ctx.add("flag_options", self.flag_options)

        # Now add anything we want to actually store.
        data = ctx.getData()
        for k in self.store:
            # First check if it is one of the high-level context objects
            if ctx.contains(k):
                data[k] = ctx.get(k)
            else:
                # Next check if it is one of the parameters
                for p in ['user_params', 'cosmo_params', 'astro_params', 'flag_options']:
                    if hasattr(ctx.get(p), k):
                        data[k] = getattr(ctx.get(p), k)
                        continue

                if k not in data:
                    warnings.warn("Cannot find variable %s to store."%k)

    def _update_params(self, ctx):
        """
        Update all the parameter structures which get passed to the driver, for this iteration.

        Parameters
        ----------
        ctx : context
        """
        params = ctx.getParams()

        apkeys = {}
        cpkeys = {}

        # Update the Astrophysical/Cosmological Parameters for this iteration
        for k in self.astro_params._defaults_:
            apkeys[k] = getattr(self.astro_params, k)
        for k in self.cosmo_params._defaults_:
            cpkeys[k] = getattr(self.cosmo_params, k)

        # Update the Astrophysical/Cosmological Parameters for this iteration
        for k in params.keys:
            if k in self.astro_params._defaults_:
                apkeys[k] = getattr(params, k)
            elif k in self.cosmo_params._defaults_:
                cpkeys[k] = getattr(params, k)
            else:
                raise ValueError("Key %s is not in AstroParams or CosmoParams " % k)

        if self._regen_init:
            del cpkeys['RANDOM_SEED']

        astro_params = p21.AstroParams(**apkeys)
        cosmo_params = p21.CosmoParams(**cpkeys)

        return astro_params, cosmo_params

    def run(self, astro_params, cosmo_params):
        """
        Actually run the 21cmFAST code.
        """
        # TODO: almost certainly not the best way to iterate through redshift...
        init, perturb, xHI, brightness_temp = p21.run_coeval(
            redshift=self.redshifts,
            astro_params=astro_params, flag_options=self.flag_options,
            cosmo_params=cosmo_params, user_params=self.user_params,
            perturb=self.perturb_field,
            init_box=self.initial_conditions,
            do_spin_temp=self.do_spin_temp,
            z_step_factor=self.z_step_factor,
            regenerate=self.regenerate, # TODO: perhaps should always be true.
            write=True, # TODO: unsure if this is a good idea...
            direc=self.direc,
            match_seed=True # TODO: shouldn't matter if regenerate is always true.
        )

        return brightness_temp