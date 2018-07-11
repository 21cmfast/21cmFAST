"""
A module providing Core Modules for cosmoHammer. This is the basis of the plugin system for py21cmmc.
"""
import numpy as np
import warnings
from copy import deepcopy
from .._21cmfast import UserParams, CosmoParams, AstroParams, FlagOptions, perturb_field, ionize_box

class CoreCoEvalModule:
    # The following disables the progress bar for perturbing and ionizing fields.
    disable_progress_bar = True

    def __init__(self, parameter_names, redshift, store = [],
                 user_params=UserParams(), flag_options=FlagOptions(), astro_params=AstroParams(),
                 cosmo_params=CosmoParams(), direc=".", regenerate=False):

        # SETUP variables
        self.parameter_names = parameter_names
        self.store = store
        self.redshift = redshift

        # Save the following only as dictionaries (not full Structure objects), so that we can avoid
        # weird pickling errors.
        self.user_params = user_params
        self.flag_options = flag_options
        self.astro_params = astro_params
        self.cosmo_params = cosmo_params

        self.direc = direc
        self.regenerate = regenerate

    def setup(self):

        self._regen_init = False
        self._write_init = True
        self._modifying_cosmo = False

        # The following ensures that if we are changing cosmology in the MCMC, then we re-do the init
        # and perturb_field parts on each iteration.
        for p in self.parameter_names:
            if p in CosmoParams._defaults_.keys():
                self._write_init = False
                self._regen_init = True
                self._modifying_cosmo = True

        # Here we create the init boxes and perturb boxes, written to file.
        # If modifying cosmo, we don't want to do this, because we'll create them
        # on the fly on every iteration. We don't need to save any values in memory because
        # they will be read from file.
        if not self._modifying_cosmo:
            print("Initializing init and perturb boxes for the run...")
            perturb_field(
                redshift = self.redshift,
                user_params=self.user_params,
                cosmo_params=self.cosmo_params,
                write=True,
                direc=self.direc,
                regenerate=self.regenerate
            )

    def __call__(self, ctx):
        # Update parameters
        astro_params, cosmo_params = self._update_params(ctx)

        # Call C-code
        ionized_box = self.run(AstroParams, CosmoParams)

        ctx.add('ionized_box', ionized_box)
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

        astro_params = AstroParams(**apkeys)
        cosmo_params = CosmoParams(**cpkeys)

        return astro_params, cosmo_params

    def _run_21cmfast(self, AstroParams, CosmoParams):
        """
        Actually run the 21cmFAST code.

        Parameters
        ----------

        Returns
        -------
        lightcone : Lightcone object.
        """
        return p21c.run_21cmfast(
            self._flag_options['redshifts'],
            self._box_dim,
            self._flag_options,
            AstroParams, CosmoParams,
            self._write_init, self._regen_init,
            progress_bar=not self.disable_progress_bar
        )[0]
