from cosmoHammer.LikelihoodComputationChain import LikelihoodComputationChain as LCC


class LikelihoodComputationChain(LCC):

    def simulate(self):
        # TODO: this might not work, and if it does, it's not obvious.
        ctx = self.createChainContext({})
        self.invokeCoreModules(ctx)

        for lk in self.getLikelihoodModules():
            lk.simulate(ctx)

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