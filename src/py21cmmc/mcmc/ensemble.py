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

        logger.debug("Walkers accepted?: {}".format(accept))

        return q, newlnprob, accept, blob

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