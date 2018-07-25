from cosmoHammer.LikelihoodComputationChain import LikelihoodComputationChain as LCC


class LikelihoodComputationChain(LCC):

    def __init__(self, params, *args, **kwargs):
        self.params = params
        super().__init__(min=params[:,1] if params is not None else None,
                         max=params[:,2] if params is not None else None)

    def simulate(self):
        # TODO: this might not work, and if it does, it's not obvious.
        ctx = self.core_context()

        for lk in self.getLikelihoodModules():
            lk.simulate(ctx)

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
        coremodule.prepare_storage(ctx, ctx.getData()) # This adds the ability to store stuff.

    def invokeLikelihoodModule(self, module, ctx):
        module.computeLikelihood(ctx, ctx.getData())