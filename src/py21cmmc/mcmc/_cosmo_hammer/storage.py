"""
A module containing classes which ease the storage of data during chain computation. The classes inherit from the
basic sampleFileUtil, and add the ability to store extra data (which seems to be intentionally possible, but not
implemented within cosmoHammer).
"""

from cosmoHammer.util import SampleFileUtil
import numpy as np
import pickle
import cosmoHammer.Constants as c
import os
import h5py

class DataSampleFileUtil(SampleFileUtil):
    """
    This writes out stuff that is stored specially in the dictionary returned by "getData" on the context (not just
    stored in the context directly).
    """

    def __init__(self, filePrefix, nwalkers, master=True, reuseBurnin=False, continue_sampling=False):
        self.filePrefix = filePrefix
        self.continue_sampling = continue_sampling
        self.samplesFileBurnin = self.probFileBurnin = self.samplesFile = self.probFile = None

        if master:
            if reuseBurnin:
                mode = "r"
            else:
                mode = "w"
            self.samplesFileBurnin = open(self.filePrefix + c.BURNIN_SUFFIX, mode)
            self.probFileBurnin = open(self.filePrefix + c.BURNIN_PROB_SUFFIX, mode)

            self.samplesFile = open(self.filePrefix + c.FILE_SUFFIX, "a" if continue_sampling else "w")
            self.probFile = open(self.filePrefix + c.PROB_SUFFIX, "a" if continue_sampling else "w")

            self._written_scalar_header = False
            self._iter = 0
            if continue_sampling:
                try:
                    data = np.loadtxt(self.filePrefix+"data_scalars")
                    if len(data):
                        self._written_scalar_header = True

                    self._iter = int(len(data)/nwalkers)
                except OSError:
                    pass

            self.dataScalarFile = open(self.filePrefix + "data_scalars", 'a' if continue_sampling else 'w')
            self.dataRank1Files = []
            self._file_handles = [self.samplesFileBurnin, self.probFileBurnin, self.samplesFile, self.probFile, self.dataScalarFile] + self.dataRank1Files

    def finalize(self):
        for fh in self._file_handles:
            if fh is not None:
                fh.close()

    def close(self):
        self.finalize()

    def persistValues(self, posFile, probFile, pos, prob, data):
        # First, call the super class to write the samples out.
        super().persistValues(posFile, probFile, pos, prob, data)

        # Now write out stuff in the data.
        self.persist_data(data)

    def persist_data(self, data):
        # data is a *list* of dicts, with length of the number of walkers.

        # We want to save scalars, lists/1D arrays and 2D arrays differently.

        scalars = [k for k,v in data[0].items() if np.isscalar(v)]
        rank1 = [k for k,v in data[0].items() if (not np.isscalar(v) and len(np.shape(v))==1)]

        pickle_vars = []
        for k in data[0]:
            if k not in scalars+rank1:
                pickle_vars += [k]

        if scalars:
            if not self._written_scalar_header:
                self.dataScalarFile.write('# ' + '\t'.join(scalars))
                self._written_scalar_header = True

            self.dataScalarFile.write("\n".join(["\t".join([str(d[k]) for k in scalars]) for d in data]))
            self.dataScalarFile.write("\n")
            self.dataScalarFile.flush()

        if rank1:
            if not self.dataRank1Files:
                self.dataRank1Files = [open(self.filePrefix + "data_%s"%k, 'a' if self.continue_sampling else 'w') for k in rank1]

            for i, k in enumerate(rank1):
                self.dataRank1Files[i].write('\n'.join(["\t".join([str(dd) for dd in d[k]]) for d in data]))
                self.dataRank1Files[i].write('\n')
                self.dataRank1Files[i].flush()

        for k in pickle_vars:
            with open(self.filePrefix + "data_%s_%s"%(k, self._iter), 'wb') as f:
                pickle.dump([d[k] for d in data], f)

        self._iter += 1

    def get_last_burnin(self):
        """
        Get last burnin step.

        Returns
        -------
        pos, prob, rstate, num_done
        """
        pos = self.storageUtil.importFromFile(self.filePrefix + c.BURNIN_SUFFIX)
        prob = self.storageUtil.importFromFile(self.filePrefix + c.BURNIN_PROB_SUFFIX)[-self.nwalkers:]
        rstate = self.storageUtil.importRandomState(self.filePrefix + c.BURNIN_STATE_SUFFIX)
        ndone = int(len(pos)/ self.nwalkers)
        return pos[-self.nwalkers:], prob ,rstate, ndone

    def get_last_sample(self):
        """
        Get last sample step.

        Returns
        -------
        pos, prob, rstate, datas, num_done
        """
        pos = self.storageUtil.importFromFile(self.filePrefix + c.FILE_SUFFIX)
        prob = self.storageUtil.importFromFile(self.filePrefix + c.PROB_SUFFIX)[-self.nwalkers:]
        rstate = self.storageUtil.importRandomState(self.filePrefix + 'state.out')

        return pos[-self.nwalkers:], prob, rstate, self.get_last_data(), len(pos)/self.nwalkers

    def get_last_data(self):
        datas = [{}]*self.nwalkers

        # First get scalars:
        with open(self.filePrefix + "data_scalars", 'r') as f:
            scalar_names = f.readline()[1:].strip().split("\t")
        scalars = np.loadtxt(self.filePrefix + "data_scalars")[-self.nwalkers:]
        for i, s in enumerate(scalars):
            for j, name in enumerate(scalar_names):
                datas[i][name] = s[j]

        # Now get 1D arrays.

class HDFStorage:
    """
    A HDF Storage utility, based pretty much exclusively on the HDFBackend from emcee v3.0.0.
    """

    def __init__(self, filename, name):
        if h5py is None:
            raise ImportError("you must install 'h5py' to use the HDFBackend")
        self.filename = filename
        self.name = name

    @property
    def initialized(self):
        if not os.path.exists(self.filename):
            return False
        try:
            with self.open() as f:
                return self.name in f
        except (OSError, IOError):
            return False

    def open(self, mode="r"):
        return h5py.File(self.filename, mode)

    def reset(self, nwalkers, ndim):
        """Clear the state of the chain and empty the backend
        Args:
            nwakers (int): The size of the ensemble
            ndim (int): The number of dimensions
        """
        if os.path.exists(self.filename):
            mode = 'a'
        else:
            mode = 'w'

        with self.open(mode) as f:
            if self.name in f:
                del f[self.name]

            g = f.create_group(self.name)
            #g.attrs["version"] = __version__
            g.attrs["nwalkers"] = nwalkers
            g.attrs["ndim"] = ndim
            g.attrs["has_blobs"] = False
            g.attrs["iteration"] = 0
            g.create_dataset("accepted", data=np.zeros(nwalkers, dtype=int))
            g.create_dataset("chain",
                             (0, nwalkers, ndim),
                             maxshape=(None, nwalkers, ndim),
                             dtype=np.float64)
            g.create_dataset("log_prob",
                             (0, nwalkers),
                             maxshape=(None, nwalkers),
                             dtype=np.float64)

    def has_blobs(self):
        with self.open() as f:
            return f[self.name].attrs["has_blobs"]

    def get_value(self, name, flat=False, thin=1, discard=0):
        if not self.initialized:
            raise AttributeError("Cannot get values from uninitialized storage.")

        with self.open() as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError("No iterations performed for this run.")

            if name == "blobs" and not g.attrs["has_blobs"]:
                return None

            v = g[name][discard + thin - 1:self.iteration:thin]
            if flat:
                s = list(v.shape[1:])
                s[0] = np.prod(v.shape[:2])
                return v.reshape(s)

            return v

    @property
    def size(self):
        with self.open() as f:
            g = f[self.name]
            return g['chain'].shape[0]

    @property
    def shape(self):
        with self.open() as f:
            g = f[self.name]
            return g.attrs["nwalkers"], g.attrs["ndim"]

    @property
    def iteration(self):
        with self.open() as f:
            return f[self.name].attrs["iteration"]

    @property
    def accepted(self):
        with self.open() as f:
            return f[self.name]["accepted"][...]

    @property
    def random_state(self):
        with self.open() as f:
            elements = [
                v
                for k, v in sorted(f[self.name].attrs.items())
                if k.startswith("random_state_")
            ]

        return elements if len(elements) else None

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples
        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs: A dictionary of extra data, or None
        """
        self._check_blobs(blobs)

        with self.open("a") as f:
            g = f[self.name]
            ntot = g.attrs["iteration"] + ngrow
            g["chain"].resize(ntot, axis=0)
            g["log_prob"].resize(ntot, axis=0)
            if blobs is not None:
                has_blobs = g.attrs["has_blobs"]
                if not has_blobs:
                    nwalkers = g.attrs["nwalkers"]
                    blobs_dtype = []
                    for k, v in blobs.items():
                        shape = np.atleast_1d(v).shape
                        if len(shape) == 1: shape = shape[0]
                        blobs_dtype += [(k, (np.atleast_1d(v).dtype, shape))]

                    #dt = np.dtype((blobs[0].dtype, blobs[0].shape))
                    g.create_dataset("blobs", (ntot, nwalkers),
                                     maxshape=(None, nwalkers),
                                     dtype=blobs_dtype)
                else:
                    g["blobs"].resize(ntot, axis=0)

                g.attrs["has_blobs"] = True

    def save_step(self, coords, log_prob, blobs, accepted, random_state):
        """Save a step to the file
        Args:
            coords (ndarray): The coordinates of the walkers in the ensemble.
            log_prob (ndarray): The log probability for each walker.
            blobs (ndarray or None): The blobs for each walker or ``None`` if
                there are no blobs.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.
            random_state: The current state of the random number generator.
        """
        self._check(coords, log_prob, blobs, accepted)

        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]

            g["chain"][iteration, :, :] = coords
            g["log_prob"][iteration, :] = log_prob
            if blobs is not None:
                blobs = np.array([tuple([b[name] for name in g['blobs'].dtype.names]) for b in blobs],
                                  dtype=g['blobs'].dtype)
                # Blobs must be a dict
                g['blobs'][iteration, ...] = blobs

            g["accepted"][:] += accepted

            for i, v in enumerate(random_state):
                g.attrs["random_state_{0}".format(i)] = v

            g.attrs["iteration"] = iteration + 1

    def _check_blobs(self, blobs):
        has_blobs = self.has_blobs()
        if has_blobs and blobs is None:
            raise ValueError("inconsistent use of blobs")
        if self.iteration > 0 and blobs is not None and not has_blobs:
            raise ValueError("inconsistent use of blobs")

    def get_chain(self, **kwargs):
        """Get the stored chain of MCMC samples
        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
        Returns:
            array[..., nwalkers, ndim]: The MCMC samples.
        """
        return self.get_value("chain", **kwargs)

    def get_blobs(self, **kwargs):
        """Get the chain of blobs for each sample in the chain
        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
        Returns:
            array[..., nwalkers]: The chain of blobs.
        """
        return self.get_value("blobs", **kwargs)

    def get_log_prob(self, **kwargs):
        """Get the chain of log probabilities evaluated at the MCMC samples
        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
        Returns:
            array[..., nwalkers]: The chain of log probabilities.
        """
        return self.get_value("log_prob", **kwargs)

    def get_last_sample(self):
        """Access the most recent sample in the chain
        This method returns a tuple with
        * ``coords`` - A list of the current positions of the walkers in the
          parameter space. The shape of this object will be
          ``(nwalkers, dim)``.
        * ``log_prob`` - The list of log posterior probabilities for the
          walkers at positions given by ``coords`` . The shape of this object
          is ``(nwalkers,)``.
        * ``rstate`` - The current state of the random number generator.
        * ``blobs`` - (optional) The metadata "blobs" associated with the
          current position. The value is only returned if blobs have been
          saved during sampling.
        """
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError("you must run the sampler with "
                                 "'store == True' before accessing the "
                                 "results")
        it = self.iteration
        last = [
            self.get_chain(discard=it - 1)[0],
            self.get_log_prob(discard=it - 1)[0],
            self.random_state,
        ]
        blob = self.get_blobs(discard=it - 1)
        if blob is not None:
            last.append(blob[0])

        return tuple(last)

    def _check(self, coords, log_prob, blobs, accepted):
        self._check_blobs(blobs)
        nwalkers, ndim = self.shape
        has_blobs = self.has_blobs()
        if coords.shape != (nwalkers, ndim):
            raise ValueError("invalid coordinate dimensions; expected {0}"
                             .format((nwalkers, ndim)))
        if log_prob.shape != (nwalkers,):
            raise ValueError("invalid log probability size; expected {0}"
                             .format(nwalkers))
        if blobs is not None and not has_blobs:
            raise ValueError("unexpected blobs")
        if blobs is None and has_blobs:
            raise ValueError("expected blobs, but none were given")
        if blobs is not None and len(blobs) != nwalkers:
            raise ValueError("invalid blobs size; expected {0}"
                             .format(nwalkers))
        if accepted.shape != (nwalkers,):
            raise ValueError("invalid acceptance size; expected {0}"
                             .format(nwalkers))


class HDFStorageUtil:
    def __init__(self, file_prefix, chain_number=0):
        self.file_prefix = file_prefix
        self.burnin_storage = HDFStorage(file_prefix + '.h5', name='burnin')
        self.sample_storage = HDFStorage(file_prefix + '.h5', name='sample_%s'%chain_number)

    def reset(self, nwalkers, ndim, burnin=True, samples=True):
        if burnin:
            self.burnin_storage.reset(nwalkers, ndim)
        if samples:
            self.sample_storage.reset(nwalkers, ndim)

    @property
    def burnin_initialized(self):
        return self.burnin_storage.initialized

    @property
    def samples_initialized(self):
        return self.sample_storage.initialized

    def persistValues(self, pos, prob, data, accepted, random_state, burnin=False):
        st = self.burnin_storage if burnin else self.sample_storage
        st.save_step(pos, prob, data, accepted, random_state)

    def close(self):
        pass

    # def get_last_burnin(self):
    #     """
    #     Get last burnin step.
    #
    #     Returns
    #     -------
    #     pos, prob, rstate, num_done
    #     """
    #     pos = self.storageUtil.importFromFile(self.filePrefix + c.BURNIN_SUFFIX)
    #     prob = self.storageUtil.importFromFile(self.filePrefix + c.BURNIN_PROB_SUFFIX)[-self.nwalkers:]
    #     rstate = self.storageUtil.importRandomState(self.filePrefix + c.BURNIN_STATE_SUFFIX)
    #     ndone = int(len(pos)/ self.nwalkers)
    #     return pos[-self.nwalkers:], prob ,rstate, ndone
    #
    # def get_last_sample(self):
    #     """
    #     Get last sample step.
    #
    #     Returns
    #     -------
    #     pos, prob, rstate, datas, num_done
    #     """
    #     pos = self.storageUtil.importFromFile(self.filePrefix + c.FILE_SUFFIX)
    #     prob = self.storageUtil.importFromFile(self.filePrefix + c.PROB_SUFFIX)[-self.nwalkers:]
    #     rstate = self.storageUtil.importRandomState(self.filePrefix + 'state.out')
    #
    #     return pos[-self.nwalkers:], prob, rstate, self.get_last_data(), len(pos)/self.nwalkers
    #
    # def get_last_data(self):
    #     datas = [{}]*self.nwalkers
    #
    #     # First get scalars:
    #     with open(self.filePrefix + "data_scalars", 'r') as f:
    #         scalar_names = f.readline()[1:].strip().split("\t")
    #     scalars = np.loadtxt(self.filePrefix + "data_scalars")[-self.nwalkers:]
    #     for i, s in enumerate(scalars):
    #         for j, name in enumerate(scalar_names):
    #             datas[i][name] = s[j]

        # Now get 1D arrays.
#
#
#
# class StorageOriginal(Storage):
#     def __init__(self, datadir, keep_global_data = True, keep_all_data = False):
#         self.datadir = datadir
#         self.keep_global_data = keep_global_data
#         self.keep_all_data = keep_all_data
#
#     def store(self, ctx):
#         """
#         Write desired data to file for permanent storage.
#
#         Currently unused, but ported from Brad's verison.
#
#         Parameters
#         ----------
#         lightcone : the lightcone object
#
#         """
#         # Get the required variables out of context.
#         random_ids = ctx.get("random_ids")  # TODO: this is not actually in the context at the moment...
#         lightcone = ctx.get("output")
#         use_lightcone = ctx.get("flag_options").USE_LIGHTCONE
#         print_files = ctx.get("flag_options").PRINT_FILES
#         store_data = ctx.get("flag_options").STORE_DATA
#
#         StringArgument_other = "%s_%s" % random_ids
#         StoredStatisticalData = []
#         StoredFileLayout = []
#
#         separator_column = "\t"
#
#         if not self.keep_global_data:
#             if not use_lightcone:
#
#                 for i in range(len(lightcone.redshift)):
#
#                     if self.keep_all_data:
#
#                         if i == 0:
#                             StoredStatisticalData.append(lightcone.k)
#                             StoredFileLayout.append("{%i}" % (i))
#
#                         StoredStatisticalData.append(lightcone.power_spectrum[i])
#                         StoredFileLayout.append("{%i}" % (i + 1))
#
#             if self.keep_all_data:
#
#                 StoredFileLayout = string.join(StoredFileLayout, separator_column)
#
#                 with open('%s/StatisticalData/TotalPSData_%s.txt' % (
#                     self.datadir, StringArgument_other), 'w') as f:
#                     for x in zip(*StoredStatisticalData):
#                         f.write("%s\n" % (StoredFileLayout).format(*x))
#
#         # Move all the information into subdirectories.
#         if use_lightcone:
#
#             if self.keep_global_data or print_files:
#                 LightconePSFilename = 'delTps_lightcone_filenames_%s.txt' % (StringArgument_other)
#                 filename = open('%s' % (LightconePSFilename), 'r')
#                 LightconePS = [line.rstrip('\n') for line in filename]
#
#             if self.keep_all_data:
#                 os.rename(LightconePSFilename,
#                           "%s/StatisticalData/%s" % (self.datadir, LightconePSFilename))
#
#             if print_files:
#                 os.remove(LightconePSFilename)
#
#             # Removal of the individual light cone files is done here as in principle these can exceed the
#             # number of mock observations provided
#             if self.keep_all_data:
#                 for i in range(len(LightconePS)):
#                     os.rename(LightconePS[i],
#                               "%s/StatisticalData/%s" % (self.datadir, LightconePS[i]))
#
#             if print_files:
#                 for i in range(len(LightconePS)):
#                     os.remove(LightconePS[i])
#         else:
#
#             if print_files:
#                 os.remove("delTps_estimate_%s_*" % StringArgument_other)
#                 os.remove("NeutralFraction_%s_*" % StringArgument_other)
#
#         if store_data:
#             avefile = "AveData_%s.txt" % StringArgument_other
#             if self.keep_all_data:
#                 os.rename(avefile, "%s/AveData/%s" % (self.datadir, avefile))
#
#             if print_files:
#                 os.remove(avefile)
#
#         walkerfile = "Walker_%s.txt" % StringArgument_other
#         walkercosmofile = "WalkerCosmology_%s.txt" % StringArgument_other
#         if self.keep_all_data:
#             os.rename(walkerfile, "%s/WalkerData/%s" % (self.datadir, walkerfile))
#             os.rename(walkercosmofile, "%s/WalkerData/%s" % (self.datadir, walkercosmofile))
#
#         if print_files:
#             os.remove(walkerfile)
#             os.remove(walkercosmofile)
