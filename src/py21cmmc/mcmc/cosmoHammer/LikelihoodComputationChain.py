from cosmoHammer.LikelihoodComputationChain import LikelihoodComputationChain as LCC
from .util import Params
from cosmoHammer.ChainContext import ChainContext

class LikelihoodComputationChain(LCC):

    def __init__(self, params, *args, **kwargs):
        self.params = params
        super().__init__(min=params[:, 1] if params is not None else None,
                         max=params[:, 2] if params is not None else None)

    def simulate(self):
        # TODO: this might not work, and if it does, it's not obvious.
        ctx = self.core_simulated_context()

        for lk in self.getLikelihoodModules():
            lk.simulate(ctx)

    def core_simulated_context(self, ctx=None):
        if ctx is None:
            ctx = self.createChainContext({})

        for m in self.getCoreModules():
            m.simulate_data(ctx)

        return ctx

    def core_context(self, ctx=None):
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
        module.LikelihoodComputationChain = self

    def addCoreModule(self, module):
        """
        adds a module to the likelihood module list

        :param module: callable
            the callable module to add for the computation of the data
        """
        self.getCoreModules().append(module)
        module.LikelihoodComputationChain = self

    def invokeCoreModule(self, coremodule, ctx):
        coremodule(ctx)
        coremodule.prepare_storage(ctx, ctx.getData())  # This adds the ability to store stuff.

    def invokeLikelihoodModule(self, module, ctx):
        model = module.simulate(ctx)
        if hasattr(module, "store"):
            module.store(model, ctx.getData())
        return module.computeLikelihood(model)

    def createChainContext(self, p):
        """
        Returns a new instance of a chain context
        """
        try:
            p = Params(*zip(self.params.keys, p))
        except Exception:
            # no params or params has no keys
            pass
        return ChainContext(self, p)

    def setup(self):
        for cModule in self.getCoreModules():
            if hasattr(cModule, "setup"):
                cModule.setup()

        for cModule in self.getLikelihoodModules():
            if hasattr(cModule, "setup"):
                cModule.setup()