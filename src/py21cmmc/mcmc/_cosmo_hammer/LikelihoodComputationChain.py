from cosmoHammer.LikelihoodComputationChain import LikelihoodComputationChain as LCC
import emcee

if emcee.__version__ >= '3.0.0':
    class LikelihoodComputationChain(LCC):

        def __call__(self, p, ret_dict=False):
            likelihood, data = super().__call__(p)

            if ret_dict:
                return data

            if type(data)==dict:
                return (likelihood,) + tuple(data.values())
            else:
                return likelihood, data

else:
    LikelihoodComputationChain = LCC
