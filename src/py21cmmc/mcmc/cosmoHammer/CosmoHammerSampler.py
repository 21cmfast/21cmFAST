"""
A replacement module for the standard CosmoHammer.CosmoHammerSampler module.

The samplers in this module provide the ability to continue sampling if the sampling is discontinued for any reason.
Two samplers are provided -- one which works for emcee versions 3+, and one which works for the default v2. Note that
the output file structure looks quite different for these versions.
"""
import logging
import time

import emcee
import numpy as np
from cosmoHammer import CosmoHammerSampler as CHS, getLogger

from py21cmmc.mcmc.ensemble import EnsembleSampler


class CosmoHammerSampler(CHS):
    def __init__(self, likelihoodComputationChain, continue_sampling=False, log_level_stream=logging.ERROR,
                 max_init_attempts=100,
                 *args, **kwargs):
        self.continue_sampling = continue_sampling
        self._log_level_stream = log_level_stream

        super().__init__(params=likelihoodComputationChain.params,
                         likelihoodComputationChain=likelihoodComputationChain,
                         *args, **kwargs)

        self.max_init_attempts = max_init_attempts
        if not self.reuseBurnin:
            self.storageUtil.reset(self.nwalkers, self.params)

        if not continue_sampling:
            self.storageUtil.reset(self.nwalkers, self.params, burnin=False)

        if not self.storageUtil.burnin_initialized:
            self.storageUtil.reset(self.nwalkers, self.params, burnin=True, samples=False)

        if not self.storageUtil.samples_initialized:
            self.storageUtil.reset(self.nwalkers, self.params, burnin=False, samples=True)
            ''
        if self.storageUtil.burnin_storage.iteration >= self.burninIterations:
            self.log("all burnin iterations already completed")

        if self.storageUtil.sample_storage.iteration > 0 and self.storageUtil.burnin_storage.iteration < self.burninIterations:
            self.log("resetting sample iterations because more burnin iterations requested.")
            self.storageUtil.reset(self.nwalkers, self.params, burnin=False)

        if self.storageUtil.sample_storage.iteration >= self.sampleIterations:
            raise ValueError("All Samples have already been completed. Try with continue_sampling=False.")

    def _configureLogging(self, filename, logLevel):
        super()._configureLogging(filename, logLevel)

        logger = getLogger()
        logger.setLevel(logLevel)
        ch = logging.StreamHandler()
        ch.setLevel(self._log_level_stream)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    def startSampling(self):
        """
        Launches the sampling
        """
        try:
            if self.isMaster(): self.log(self.__str__())

            prob = None
            rstate = None
            datas = None
            pos = None
            if self.storageUtil.burnin_storage.iteration < self.burninIterations:
                if self.burninIterations:
                    if self.storageUtil.burnin_storage.iteration:
                        pos, prob, rstate, datas = self.loadBurnin()

                    if self.storageUtil.burnin_storage.iteration < self.burninIterations:
                        pos, prob, rstate, datas = self.startSampleBurnin(pos, prob, rstate, datas)
                else:
                    pos = self.createInitPos()
            else:
                if self.storageUtil.sample_storage.iteration:
                    pos, prob, rstate, datas = self.loadSamples()

                else:
                    pos = self.createInitPos()

            # Starting from the final position in the burn-in chain, start sampling.
            self.log("start sampling after burn in")
            start = time.time()
            self.sample(pos, prob, rstate, datas)
            end = time.time()
            self.log("sampling done! Took: " + str(round(end - start, 4)) + "s")

            # Print out the mean acceptance fraction. In general, acceptance_fraction
            # has an entry for each walker
            self.log("Mean acceptance fraction:" + str(round(np.mean(self._sampler.acceptance_fraction), 4)))
        finally:
            if self._sampler.pool is not None:
                try:
                    self._sampler.pool.close()
                except AttributeError:
                    pass
                try:
                    self.storageUtil.close()
                except AttributeError:
                    pass

    def createEmceeSampler(self, callable, **kwargs):
        """
        Factory method to create the emcee sampler
        """
        if self.isMaster(): self.log("Using emcee " + str(emcee.__version__))
        return EnsembleSampler(
            pmin=self.likelihoodComputationChain.min, pmax=self.likelihoodComputationChain.max,
            nwalkers=self.nwalkers, dim=self.paramCount, lnpostfn=callable, threads=self.threadCount, **kwargs
        )

    def _load(self, burnin=False):
        stg = self.storageUtil.burnin_storage if burnin else self.storageUtil.sample_storage

        self.log("reusing previous %s: %s iterations" % ("burnin" if burnin else "samples", stg.iteration))
        pos, prob, rstate, data = stg.get_last_sample()
        if data is not None:
            data = [{k: d[k] for k in d.dtype.names} for d in data]
        return pos, prob, rstate, data

    def loadBurnin(self):
        """
        loads the burn in from the file system
        """
        return self._load(burnin=True)

    def loadSamples(self):
        """
        loads the samples from the file system
        """
        return self._load(burnin=False)

    def startSampleBurnin(self, pos=None, prob=None, rstate=None, data=None):
        """
        Runs the sampler for the burn in
        """
        if self.storageUtil.burnin_storage.iteration:
            self.log("continue burn in")
        else:
            self.log("start burn in")
        start = time.time()

        if pos is None: pos = self.createInitPos()
        pos, prob, rstate, data = self.sampleBurnin(pos, prob, rstate, data)
        end = time.time()
        self.log("burn in sampling done! Took: " + str(round(end - start, 4)) + "s")
        self.log("Mean acceptance fraction for burn in:" + str(round(np.mean(self._sampler.acceptance_fraction), 4)))

        self.resetSampler()

        return pos, prob, rstate, data

    def _sample(self, p0, prob=None, rstate=None, datas=None, burnin=False):
        """
        Run the emcee sampler for the burnin to create walker which are independent form their starting position
        """
        stg = self.storageUtil.burnin_storage if burnin else self.storageUtil.sample_storage
        niter = self.burninIterations if burnin else self.sampleIterations

        _lastprob = prob if prob is None else [0] * len(p0)

        # Set to None in case iterations is zero.
        pos = None

        for pos, prob, rstate, realpos, realprob, datas in self._sampler.sample(
                p0,
                iterations=niter - stg.iteration,
                lnprob0=prob, rstate0=rstate, blobs0=datas
        ):
            if self.isMaster():
                # Need to grow the storage first
                if not stg.iteration:
                    stg.grow(niter - stg.iteration, datas[0])

                # If we are continuing sampling, we need to grow it more.
                if stg.size < niter:
                    stg.grow(niter - stg.size, datas[0])

                self.storageUtil.persistValues(pos, prob, datas,
                                               truepos=realpos,
                                               trueprob=realprob,
                                               accepted=prob != _lastprob,
                                               random_state=rstate,
                                               burnin=burnin)
                if stg.iteration % 10 == 0:
                    self.log("Iteration finished:" + str(stg.iteration))

                _lastprob = 1 * prob

                if self.stopCriteriaStrategy.hasFinished():
                    break

        return pos, prob, rstate, datas

    def sampleBurnin(self, p0, prob=None, rstate=None, datas=None):
        return self._sample(p0, prob, rstate, datas, burnin=True)

    def sample(self, burninPos, burninProb=None, burninRstate=None, datas=None):
        return self._sample(burninPos, burninProb, burninRstate, datas)

    @property
    def samples(self):
        if not self.storageUtil.sample_storage.initialized:
            raise ValueError("Cannot access samples before sampling.")
        else:
            return self.storageUtil.sample_storage

    def createInitPos(self):
        """
        Factory method to create initial positions
        """
        i = 0
        pos = []

        while len(pos) < self.nwalkers and i < self.max_init_attempts:
            tmp_pos = self.initPositionGenerator.generate()

            for tmp_p in tmp_pos:
                if self.likelihoodComputationChain.isValid(tmp_p):
                    pos.append(tmp_p)

            i += 1

        if i == self.max_init_attempts:
            raise ValueError(
                "No suitable initial positions for the walkers could be obtained in {max_attempts} attemps".format(
                    max_attempts=self.max_init_attempts))

        return pos[:self.nwalkers]
