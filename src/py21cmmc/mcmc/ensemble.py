import emcee
import numpy as np


class EnsembleSampler(emcee.EnsembleSampler):
    """
    An over-write of the standard emcee EnsembleSampler which ensures that sampled parameters are never outside their
    range.
    """

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
        s = np.atleast_2d(p0)
        Ns = len(s)
        c = np.atleast_2d(p1)
        Nc = len(c)

        # Generate the vectors of random numbers that will produce the
        # proposal.
        parameters_are_good = False
        qgood = np.zeros((Ns, self.dim), dtype=bool)
        q = np.zeros((Ns, self.dim))
        while not np.all(qgood):
            zz = ((self.a - 1.) * self._random.rand(Ns) + 1) ** 2. / self.a
            rint = self._random.randint(Nc, size=(Ns,))

            # Calculate the proposed positions and the log-probability there.
            tmpq = c[rint] - zz[:, np.newaxis] * (c[rint] - s)

            q[np.logical_not(qgood)] = tmpq[np.logical_not(qgood)]

            qgood = np.logical_and(q >= self.pmin, q <= self.pmax)

        newlnprob, blob = self._get_lnprob(q)

        # Decide whether or not the proposals should be accepted.
        lnpdiff = (self.dim - 1.) * np.log(zz) + newlnprob - lnprob0
        accept = (lnpdiff > np.log(self._random.rand(len(lnpdiff))))

        return q, newlnprob, accept, blob
