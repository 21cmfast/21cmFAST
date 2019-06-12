"""
A set of extensions to the basic CosmoHammer package, to make it use emcee v3.0+ and add a few extra features.
"""

from .CosmoHammerSampler import CosmoHammerSampler
from .LikelihoodComputationChain import LikelihoodComputationChain
from .storage import HDFStorageUtil
from .util import Params
