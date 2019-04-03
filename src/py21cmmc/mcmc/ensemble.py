import logging

import emcee
import numpy as np
from concurrent.futures.process import BrokenProcessPool

logger = logging.getLogger("21CMMC")


class EnsembleSampler(emcee.EnsembleSampler):
    """
    An over-write of the standard emcee EnsembleSampler which ensures that sampled parameters are never outside their
    range.
    """
    max_attempts = 100

    def __init__(self, pmin=None, pmax=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pmin = -np.inf * np.ones(self.dim) if pmin is None else pmin
        self.pmax = np.inf * np.ones(self.dim) if pmax is None else pmax

    def _propose_stretch(self, p0, p1, lnprob0):
        """
        Propose a new position for one sub-ensemble given the positions of
        another.
        :param p0:
            The positions from which to jump.
        :param p1:
            The positions of the other ensemble.
        :param lnprob0:
            The log-probabilities at ``p0``.
        This method returns:
        * ``q`` - The new proposed positions for the walkers in ``ensemble``.
        * ``newlnprob`` - The vector of log-probabilities at the positions
          given by ``q``.
        * ``accept`` - A vector of type ``bool`` indicating whether or not
          the proposed position for each walker should be accepted.
        * ``blob`` - The new meta data blobs or ``None`` if nothing was
          returned by ``lnprobfn``.
        """
        logger.debug("Proposing new walker positions")

        s = np.atleast_2d(p0)
        Ns = len(s)
        c = np.atleast_2d(p1)
        Nc = len(c)

        qgood = np.zeros((Ns, self.dim), dtype=bool)
        q = np.zeros((Ns, self.dim))
        i = 0
        while not np.all(qgood) and i < self.max_attempts:
            # Generate the vectors of random numbers that will produce the proposal.
            zz = ((self.a - 1.) * self._random.rand(Ns) + 1) ** 2. / self.a
            rint = self._random.randint(Nc, size=(Ns,))

            # Calculate the proposed positions
            tmpq = c[rint] - zz[:, np.newaxis] * (c[rint] - s)
            q[np.logical_not(qgood)] = tmpq[np.logical_not(qgood)]
            qgood = np.logical_and(q >= self.pmin, q <= self.pmax)
            i += 1

        if i == self.max_attempts and not np.all(qgood):
            msg = "Faulty Parameters:\n"
            for j, qq in enumerate(qgood):
                for k, qqq in enumerate(qq):
                    if not qqq:
                        msg += "Walker {j}, parameter {k}, previous value = {prev}, range = ({min}, {max})\n".format(
                            j=j, k=k, prev=s[j, k], min=self.pmin[k], max=self.pmax[k]
                        )
            raise RuntimeError(
                """
                EnsembleSampler could not find a suitable selection of parameters in {max_attempts} attempts.
                {msg}
                """.format(max_attempts=self.max_attempts)
            )

        newlnprob, blob = self._get_lnprob(q)

        # Decide whether or not the proposals should be accepted.
        lnpdiff = (self.dim - 1.) * np.log(zz) + newlnprob - lnprob0
        accept = (lnpdiff > np.log(self._random.rand(len(lnpdiff))))

        return q, newlnprob, accept, blob

    def sample(self, p0, lnprob0=None, rstate0=None, blobs0=None,
               iterations=1, thin=1, storechain=True, mh_proposal=None):
        """
        Advance the chain ``iterations`` steps as a generator.

        :param p0:
            A list of the initial positions of the walkers in the
            parameter space. It should have the shape ``(nwalkers, dim)``.

        :param lnprob0: (optional)
            The list of log posterior probabilities for the walkers at
            positions given by ``p0``. If ``lnprob is None``, the initial
            values are calculated. It should have the shape ``(k, dim)``.

        :param rstate0: (optional)
            The state of the random number generator.
            See the :attr:`Sampler.random_state` property for details.

        :param iterations: (optional)
            The number of steps to run.

        :param thin: (optional)
            If you only want to store and yield every ``thin`` samples in the
            chain, set thin to an integer greater than 1.

        :param storechain: (optional)
            By default, the sampler stores (in memory) the positions and
            log-probabilities of the samples in the chain. If you are
            using another method to store the samples to a file or if you
            don't need to analyse the samples after the fact (for burn-in
            for example) set ``storechain`` to ``False``.

        :param mh_proposal: (optional)
            A function that returns a list of positions for ``nwalkers``
            walkers given a current list of positions of the same size. See
            :class:`utils.MH_proposal_axisaligned` for an example.

        At each iteration, this generator yields:

        * ``pos`` - A list of the current positions of the walkers in the
          parameter space. The shape of this object will be
          ``(nwalkers, dim)``.

        * ``lnprob`` - The list of log posterior probabilities for the
          walkers at positions given by ``pos`` . The shape of this object
          is ``(nwalkers, dim)``.

        * ``rstate`` - The current state of the random number generator.

        * ``blobs`` - (optional) The metadata "blobs" associated with the
          current position. The value is only returned if ``lnpostfn``
          returns blobs too.

        """
        # Try to set the initial value of the random number generator. This
        # fails silently if it doesn't work but that's what we want because
        # we'll just interpret any garbage as letting the generator stay in
        # it's current state.
        self.random_state = rstate0

        p = np.array(p0)

        # NEW: save all (even rejected) samples
        q = np.zeros_like(p0)

        halfk = int(self.k / 2)

        # If the initial log-probabilities were not provided, calculate them
        # now.
        lnprob = lnprob0
        blobs = blobs0
        if lnprob is None:
            lnprob, blobs = self._get_lnprob(p)


        newlnp = np.zeros_like(lnprob)

        # Check to make sure that the probability function didn't return
        # ``np.nan``.
        if np.any(np.isnan(lnprob)):
            raise ValueError("The initial lnprob was NaN.")

        # Store the initial size of the stored chain.
        i0 = self._chain.shape[1]

        # Here, we resize chain in advance for performance. This actually
        # makes a pretty big difference.
        if storechain:
            N = int(iterations / thin)
            self._chain = np.concatenate((self._chain,
                                          np.zeros((self.k, N, self.dim))),
                                         axis=1)
            self._lnprob = np.concatenate((self._lnprob,
                                           np.zeros((self.k, N))), axis=1)

        for i in range(int(iterations)):
            self.iterations += 1

            # If we were passed a Metropolis-Hastings proposal
            # function, use it.
            if mh_proposal is not None:
                # Draw proposed positions & evaluate lnprob there
                q = mh_proposal(p)
                newlnp, blob = self._get_lnprob(q)

                # Accept if newlnp is better; and ...
                acc = (newlnp > lnprob)

                # ... sometimes accept for steps that got worse
                worse = np.flatnonzero(~acc)
                acc[worse] = ((newlnp[worse] - lnprob[worse]) >
                              np.log(self._random.rand(len(worse))))
                del worse

                # Update the accepted walkers.
                lnprob[acc] = newlnp[acc]
                p[acc] = q[acc]
                self.naccepted[acc] += 1

                if blob is not None:
                    assert blobs is not None, (
                        "If you start sampling with a given lnprob, you also "
                        "need to provide the current list of blobs at that "
                        "position.")
                    ind = np.arange(self.k)[acc]
                    for j in ind:
                        blobs[j] = blob[j]

            else:
                # Loop over the two ensembles, calculating the proposed
                # positions.

                # Slices for the first and second halves
                first, second = slice(halfk), slice(halfk, self.k)
                for S0, S1 in [(first, second), (second, first)]:
                    q[S0], newlnp[S0], acc, blob = self._propose_stretch(p[S0], p[S1],
                                                                 lnprob[S0])
                    if np.any(acc):
                        # Update the positions, log probabilities and
                        # acceptance counts.
                        lnprob[S0][acc] = newlnp[S0][acc]
                        p[S0][acc] = q[S0][acc]
                        self.naccepted[S0][acc] += 1

                        if blob is not None:
                            assert blobs is not None, (
                                "If you start sampling with a given lnprob, "
                                "you also need to provide the current list of "
                                "blobs at that position.")
                            ind = np.arange(len(acc))[acc]
                            indfull = np.arange(self.k)[S0][acc]
                            for j in range(len(ind)):
                                blobs[indfull[j]] = blob[ind[j]]

            if storechain and i % thin == 0:
                ind = i0 + int(i / thin)
                self._chain[:, ind, :] = p
                self._lnprob[:, ind] = lnprob
                if blobs is not None:
                    self._blobs.append(list(blobs))

            # Yield the result as an iterator so that the user can do all
            # sorts of fun stuff with the results so far.
            if blobs is not None:
                # This is a bit of a hack to keep things backwards compatible.
                yield p, lnprob, self.random_state, q, newlnp, blobs
            else:
                yield p, lnprob, self.random_state, q, newlnp

    def _get_lnprob(self, pos=None):
        # a wrapper of the original which can also catch broken process pool exceptions

        try:
            return super()._get_lnprob(pos)
        except BrokenProcessPool:
            import traceback
            print(
                """
BrokenProcessPool exception (most likely an unrecoverable crash in C-code).  

  Due to the nature of this exception, it is impossible to know which of the following parameter 
  vectors were responsible for the crash. Running your likelihood function with each set
  of parameters in serial may help identify the problem.
"""
            )
            print("  params:", str(pos if pos is not None else self.pos).replace("\n", "\n          "))
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:\n")
            traceback.print_exc()
            raise