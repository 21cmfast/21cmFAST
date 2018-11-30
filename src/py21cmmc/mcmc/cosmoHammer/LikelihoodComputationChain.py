import warnings

from cosmoHammer.ChainContext import ChainContext
from cosmoHammer.LikelihoodComputationChain import LikelihoodComputationChain as LCC

from .util import Params


class LikelihoodComputationChain(LCC):

    def __init__(self, params, *args, **kwargs):
        self.params = params
        self._setup = False  # flag to say if this chain has been setup yet.

        super().__init__(min=params[:, 1] if params is not None else None,
                         max=params[:, 2] if params is not None else None)

    def simulate(self):
        # TODO: this might not work, and if it does, it's not obvious.
        ctx = self.core_simulated_context()

        for lk in self.getLikelihoodModules():
            lk.simulate(ctx)

    def core_simulated_context(self, ctx=None):
        """
        Generate a context filled by all cores in the chain, using their "simulate_data" method.

        This context should be a "realisation" of the data, not a model of it.

        Parameters
        ----------
        ctx : An empty context object, potentially with parameters set in it.

        Returns
        -------
        ctx : A filled context object.
        """
        if ctx is None:
            ctx = self.createChainContext({})

        for m in self.getCoreModules():
            m.simulate_data(ctx)

        return ctx

    def core_context(self, ctx=None):
        """
        Generate a context filled by all cores in the chain, by formally invoking them.

        This context should be a "model" of the data, not a realisation of it (i.e. it is exactly what
        is produced in the context on every iteration of an MCMC).

        Parameters
        ----------
        ctx : An empty context object, potentially with parameters set in it.

        Returns
        -------
        ctx : A filled context object.
        """

        if ctx is None:
            ctx = self.createChainContext({})
        self.invokeCoreModules(ctx)
        return ctx

    def addLikelihoodModule(self, module):
        """
        adds a module to the likelihood module list

        :param module: callable
            the callable module to add for the likelihood computation
        """
        self.getLikelihoodModules().append(module)
        module._LikelihoodComputationChain = self

    def addCoreModule(self, module):
        """
        adds a module to the likelihood module list

        :param module: callable
            the callable module to add for the computation of the data
        """
        self.getCoreModules().append(module)
        module._LikelihoodComputationChain = self

    def invokeCoreModule(self, coremodule, ctx):
        # Ensure that the chain is setup before invoking anything.
        if not self._setup:
            self.setup()

        coremodule(ctx)
        coremodule.prepare_storage(ctx, ctx.getData())  # This adds the ability to store stuff.

    def invokeLikelihoodModule(self, module, ctx):
        # Ensure that the chain is setup before invoking anything.
        if not self._setup:
            self.setup()

        model = module.simulate(ctx)
        if hasattr(module, "store"):
            module.store(model, ctx.getData())

        return module.computeLikelihood(model)

    def createChainContext(self, p=None):
        """
        Returns a new instance of a chain context
        """
        if p is None:
            p = {}

        try:
            p = Params(*zip(self.params.keys, p))
        except Exception:
            # no params or params has no keys
            pass
        return ChainContext(self, p)

    def setup(self):
        if not self._setup:
            for cModule in self.getCoreModules():
                if hasattr(cModule, "setup"):
                    cModule.setup()

            for cModule in self.getLikelihoodModules():
                if hasattr(cModule, "setup"):
                    cModule.setup()

            self._setup = True
        else:
            warnings.warn("Attempting to setup LikelihoodComputationChain when it is already setup! Ignoring...")
