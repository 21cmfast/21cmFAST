"""
A module providing Core Modules for cosmoHammer. This is the basis of the plugin system for py21cmmc.
"""
import warnings
from .._21cmfast import wrapper as lib


class CoreCoEvalModule:
    # The following disables the progress bar for perturbing and ionizing fields.
    disable_progress_bar = True

    def __init__(self, parameter_names, redshifts, store = [],
                 user_params=lib.UserParams(), flag_options=lib.FlagOptions(), astro_params=lib.AstroParams(),
                 cosmo_params=lib.CosmoParams(), direc=".", regenerate=False, do_spin_temp=False):

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

        if do_spin_temp:
            raise NotImplementedError("Can't yet do spin temperature...")

        if self.flag_options.INHOMO_RECO:
            raise NotImplementedError("Can't yet do INHOMO_RECO")

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

        # Here we create the init boxes and perturb boxes, written to file.
        # If modifying cosmo, we don't want to do this, because we'll create them
        # on the fly on every iteration. We don't need to save any values in memory because
        # they will be read from file.
        if not self._modifying_cosmo:
            self.initial_conditions = lib.initial_conditions(
                user_params=self.user_params,
                cosmo_params=self.cosmo_params,
                write=True,
                direc=self.direc,
                regenerate=self.regenerate
            )

            print("Initializing init and perturb boxes for the run...")
            for z in self.redshifts:
                self.perturb_field = lib.perturb_field(
                    redshift = z,
                    init_boxes=self.initial_conditions,
                    write=True,
                    direc=self.direc,
                    regenerate=self.regenerate
                )

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

        astro_params = lib.AstroParams(**apkeys)
        cosmo_params = lib.CosmoParams(**cpkeys)

        return astro_params, cosmo_params

    def run(self, astro_params, cosmo_params):
        """
        Actually run the 21cmFAST code.
        """
        # TODO: almost certainly not the best way to iterate through redshift...
        ionized_box = []
        brightness_temp = []
        for z in self.redshifts:
            if not self._modifying_cosmo:
                perturbed_field = self.perturb_field
            else:
                perturbed_field = None

            ionized_box += [lib.ionize_box(
                astro_params=astro_params, flag_options=self.flag_options,
                redshift=z, perturbed_field=perturbed_field,
                cosmo_params=cosmo_params, user_params=self.user_params,
                regenerate=self.regenerate,
                write=True, direc=self.direc,
                match_seed=True
            )]

            brightness_temp += [lib.brightness_temperature(
                ionized_box=ionized_box[-1],
                perturbed_field=perturbed_field
            )]

        return brightness_temp