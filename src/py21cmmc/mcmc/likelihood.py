"""
A module containing (base) classes for computing 21cmFAST likelihoods under the context of CosmoHammer.
"""
import numpy as np
from decimal import *
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
from .._21cmfast import wrapper as lib
from . import core

import pickle
from os import path


np.seterr(invalid='ignore', divide='ignore')

from powerbox.tools import get_power

TWOPLACES = Decimal(10) ** -2  # same as Decimal('0.01')
FOURPLACES = Decimal(10) ** -4  # same as Decimal('0.0001')
SIXPLACES = Decimal(10) ** -6  # same as Decimal('0.000001')

def ensure_iter(a):
    try:
        iter(a)
        return a
    except TypeError:
        return [a]


class LikelihoodBase:
    required_cores = None

    def __init__(self, datafile):
        self.datafile = datafile

    def computeLikelihood(self, ctx):
        raise NotImplementedError("The Base likelihood should never be used directly!")

    def setup(self):
        # Try to get the params out of the co-eval module
        for m in self.LikelihoodComputationChain.getCoreModules():
            for k in ['user_params', 'flag_options', 'cosmo_params', 'astro_params']:
                if hasattr(m, k):
                    setattr(self, k, getattr(m,k))

        if self.required_cores:
            for rc in self.required_cores:
                if not any([isinstance(m, rc) for m in self.LikelihoodComputationChain.getCoreModules()]):
                    raise ValueError("%s needs the %s to be loaded."%(self.__class__.__name__, rc.__class__.__name__))

        self.data = self._define_data()

        if self.datafile and not path.exists(self.datafile):
            np.savez(self.datafile, **self.data)

    @property
    def default_ctx(self):
        try:
            return self.LikelihoodComputationChain.core_context()
        except AttributeError:
            raise AttributeError("default_ctx is not available unless the likelihood is embedded in a LikelihoodComputationChain")

    def _define_data(self):
        if self.datafile and path.exists(self.datafile):
            return dict(**np.load(self.datafile))
        else:
            return self.simulate(self.default_ctx)


class Likelihood1DPowerCoeval(LikelihoodBase):
    """
    A likelihood which assumes that the spherically-averaged power spectrum is iid Gaussian in each bin.

    Requires the CoreCoevalModule to be loaded to work, and inherently deals with the multiple-redshift cubes
    which that module produces.

    If a `datafile` is provided and the datafile exists, then the data will be read from that file. Otherwise,
    theoretical data will be automatically simulated to match current parameters. This will be written to
    `datafile` if provided.
    """
    required_cores = [core.CoreCoevalModule]

    def __init__(self, datafile=None, n_psbins=None, min_k=0.1, max_k = 1.0, logk=True, model_uncertainty=0.15,
                 error_on_model=True,):
        """
        Initialize the likelihood.

        Parameters
        ----------
        datafile : str, optional
            The file from which to read the data. Alternatively, the file to which to write the data (see class
            docstring for how this works).
        n_psbins : int, optional
            The number of bins for the spherically averaged power spectrum. By default automatically
            calculated from the number of cells.
        min_k : float, optional
            The minimum k value at which to compare model and data.
        max_k : float, optional
            The maximum k value at which to compare model and data.
        logk : bool, optional
            Whether the power spectrum bins should be regular in logspace or linear space.
        model_uncertainty : float, optional
            The amount of uncertainty in the modelling, per power spectral bin (as fraction of the amplitude).
        error_on_model : bool, optional
            Whether the `model_uncertainty` is applied to the model, or the data.
        """
        # TODO: 21cmSense noise!

        self.n_psbins = n_psbins

        self.datafile = datafile
        self.min_k = min_k
        self.max_k = max_k
        self.logk = logk
        self.error_on_model = error_on_model
        self.model_uncertainty = model_uncertainty

    def setup(self):
        super().setup()

        self.k_data, self.p_data = self.data['k'], self.data['p']

        self.mask = np.logical_and(self.k_data >= self.min_k, self.k_data <= self.max_k)
        self.k_data = self.k_data[self.mask]
        self.p_data = [p[self.mask] for p in self.p_data]

        # This needs to be first-order (linear) so that symmetry between data and model can be preserved.
        self.data_spline = [InterpolatedUnivariateSpline(self.k_data, p, k=1) for p in self.p_data]


    @staticmethod
    def compute_power(brightness_temp, L, n_psbins, log_bins=True):
        res = get_power(
            brightness_temp.brightness_temp,
            boxlength = L,
            bins=n_psbins, bin_ave=False, get_variance=False, log_bins=log_bins
        )

        res = list(res)
        k = res[1]
        if log_bins:
            k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
        else:
            k = (k[1:] + k[:-1]) / 2

        res[1] = k
        return res

    @property
    def core_module(self):
        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, self.required_cores[0]):
                return m

    @property
    def redshift(self):
        return self.core_module.redshift

    def computeLikelihood(self, ctx, storage):
        "Compute the likelihood"
        brightness_temp = ctx.get("brightness_temp")

        # add the power to the written data
        storage['power'] = []

        lnl = 0

        for bt, pd in zip(brightness_temp, self.data_spline):
            power, k = self.compute_power(bt, self.user_params.BOX_LEN, self.n_psbins, log_bins=self.logk)

            # add the power to the written data
            storage['power'] += [power[self.mask]]
            storage['k'] = k[self.mask]

            denom = self.model_uncertainty*pd(k) if not self.error_on_model else self.model_uncertainty*power[self.mask]

            lnl += -0.5 * np.sum((power[self.mask] - pd(k)) ** 2 / denom**2)
        return lnl

    def simulate(self, ctx):
        brightness_temp = ctx.get("brightness_temp")
        p = []
        for bt in brightness_temp:

            power, k = self.compute_power(bt, self.user_params.BOX_LEN, self.n_psbins, log_bins=self.logk)
            p += [power]

        return dict(k=k, p=np.array(p))


class Likelihood1DPowerLightcone(Likelihood1DPowerCoeval):
    """
    A likelihood very similar to :class:`Likelihood1DPowerCoeval`, except for a lightcone.

    Since most of the functionality is the same, please see the other documentation for details.
    """
    required_cores = [core.CoreLightConeModule]

    def setup(self):
        super().setup()
        self.p_data = self.p_data[0] # un-list it
        self.data_spline = self.data_spline[0]

    @staticmethod
    def compute_power(lightcone, n_psbins, log_bins=True):
        res = get_power(
            lightcone.brightness_temp,
            boxlength = lightcone.lightcone_dimensions,
            bins=n_psbins, bin_ave=False, get_variance=False, log_bins=log_bins
        )

        res = list(res)
        k = res[1]
        if log_bins:
            k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
        else:
            k = (k[1:] + k[:-1]) / 2

        res[1] = k
        return res

    def computeLikelihood(self, ctx, storage):
        "Compute the likelihood"
        lightcone = ctx.get("lightcone")

        # add the power to the written data
        storage['power'] = []

        power, k = self.compute_power(lightcone, self.n_psbins, log_bins=self.logk)

        # add the power to the written data
        storage['power'] = power[self.mask]
        storage['k'] = k[self.mask]

        denom = self.model_uncertainty * self.data_spline(k) if not self.error_on_model else self.model_uncertainty * power[self.mask]
        lnl = -0.5 * np.sum((power[self.mask] - self.data_spline(k)) ** 2 / denom**2)
        return lnl

    def simulate(self, ctx):
        lightcone = ctx.get("lightcone")

        power, k = self.compute_power(lightcone, self.n_psbins, log_bins=self.logk)

        return dict(k=k, p=[power])


class LikelihoodPlanck(LikelihoodBase):
    # Mean and one sigma errors for the Planck constraints
    # The Planck prior is modelled as a Gaussian: tau = 0.058 \pm 0.012 (https://arxiv.org/abs/1605.03507)
    PlanckTau_Mean = 0.058
    PlanckTau_OneSigma = 0.012

    # Simple linear extrapolation of the redshift range provided by the user, to be able to estimate the optical depth
    nZinterp = 15

    # The minimum of the extrapolation is chosen to 5.9, to correspond to the McGreer et al. prior on the IGM neutral fraction.
    # The maximum is chosed to be z = 18., which is arbitrary.
    ZExtrap_min = 5.9
    ZExtrap_max = 20.0

    def computeLikelihood(self, ctx):
        """
        Contribution to the likelihood arising from Planck (2016) (https://arxiv.org/abs/1605.03507)
        """
        # Extract relevant info from the context.
        output = ctx.get("output")

        if len(output.redshift) < 3:
            print(output.redshift)
            raise ValueError("You cannot use the Planck prior likelihood with less than 3 redshift")

        # The linear interpolation/extrapolation function, taking as input the redshift supplied by the user and
        # the corresponding neutral fractions recovered for the specific EoR parameter set
        LinearInterpolationFunction = InterpolatedUnivariateSpline(output.redshift, output.average_nf, k=1)

        ZExtrapVals = np.zeros(self.nZinterp)
        XHI_ExtrapVals = np.zeros(self.nZinterp)

        for i in range(self.nZinterp):
            ZExtrapVals[i] = self.ZExtrap_min + (self.ZExtrap_max - self.ZExtrap_min) * float(i) / (self.nZinterp - 1)

            XHI_ExtrapVals[i] = LinearInterpolationFunction(ZExtrapVals[i])

            # Ensure that the neutral fraction does not exceed unity, or go negative
            if XHI_ExtrapVals[i] > 1.0:
                XHI_ExtrapVals[i] = 1.0
            if XHI_ExtrapVals[i] < 0.0:
                XHI_ExtrapVals[i] = 0.0

        # Set up the arguments for calculating the estimate of the optical depth. Once again, performed using command
        # line code.
        tau_value = lib.compute_tau(ZExtrapVals, XHI_ExtrapVals, ctx.get('cosmo_params'))

        # As the likelihood is computed in log space, the addition of the prior is added linearly to the existing chi^2
        # likelihood
        lnprob = np.square((self.PlanckTau_Mean - tau_value) / (self.PlanckTau_OneSigma))

        return lnprob

        # TODO: not sure what to do about this:
        # it is len(self.AllRedshifts) as the indexing begins at zero


#        nf_vals[len(self.AllRedshifts) + 2] = tau_value

class LikelihoodNeutralFraction(LikelihoodBase):
    def __init__(self, redshift, xHI, xHI_sigma):
        self.redshift = ensure_iter(redshift)
        self.xHI = ensure_iter(xHI)
        self.xHI_sigma = ensure_iter(xHI_sigma)

        self.require_spline = False

    def setup(self):
        self.lightcone_modules = [m for m in self.LikelihoodComputationChain.getCoreModules() if isinstance(m, core.CoreLightConeModule)]
        self.coeval_modules = [m for m in self.LikelihoodComputationChain.getCoreModules() if isinstance(m, core.CoreCoevalModule)]

        if not self.lightcone_modules + self.coeval_modules:
            raise ValueError("LikelihoodNeutralFraction needs the CoreLightConeModule *or* CoreCoevalModule to be loaded.")

        if self.coeval_modules:
            # Get all unique redshifts from all coeval boxes in cores.
            self.coeval_redshifts = list(set(sum([x.redshift for x in self.coeval_modules], [])))

            for z in self.redshift:
                if z not in self.coeval_redshifts and len(self.coeval_redshifts) < 3:
                    raise ValueError("To use LikelihoodNeutralFraction, the core must be a lightcone, or coeval with >=3 redshifts, or containing the desired redshift")
                elif z not in self.coeval_redshifts:
                    self.require_spline = True

            self.use_coeval = True

        else:
            self.use_coeval = False

    def computeLikelihood(self, ctx):
        lnprob = 0
        if self.use_coeval:
            xHI = np.array([np.mean(x) for x in ctx.get('xHI')])

            if self.require_spline:
                ind = np.argsort(self.coeval_redshifts)
                model_spline = InterpolatedUnivariateSpline(self.coeval_redshifts[ind], xHI[ind], k=1)

            for z, data, sigma in zip(self.redshift, self.xHI, self.xHI_sigma):
                if z in self.coeval_redshifts:
                    lnprob += self.lnprob(xHI[self.coeval_redshifts.index(z)], data, sigma)
                else:
                    lnprob += self.lnprob(model_spline(z), data, sigma)

        else:
            pass

    @staticmethod
    def lnprob(model, data, sigma):
        model = np.clip(model, 0, 1)

        if model > 0.06:
            return ((data - model) / sigma)**2
        else:
            return 0

class LikelihoodMcGreer(LikelihoodBase):
    # Mean and one sigma errors for the McGreer et al. constraints
    # Modelled as a flat, unity prior at x_HI <= 0.06, and a one sided Gaussian at x_HI > 0.06
    # ( Gaussian of mean 0.06 and one sigma of 0.05 )
    McGreer_Mean = 0.06
    McGreer_OneSigma = 0.05
    McGreer_Redshift = 5.9

    def computeLikelihood(self, ctx):
        """
        Limit on the IGM neutral fraction at z = 5.9, from dark pixels by I. McGreer et al.
        (2015) (http://adsabs.harvard.edu/abs/2015MNRAS.447..499M)
        """
        lightcone = ctx.get("output")

        if self.McGreer_Redshift in lightcone.redshift:
            for i in range(len(lightcone.redshift)):
                if lightcone.redshift[i] == self.McGreer_Redshift:
                    McGreer_NF = lightcone.average_nf[i]
        elif len(lightcone.redshift) > 2:
            # The linear interpolation/extrapolation function, taking as input the redshift supplied by the user and
            # the corresponding neutral fractions recovered for the specific EoR parameter set
            LinearInterpolationFunction = InterpolatedUnivariateSpline(lightcone.redshift, lightcone.average_nf, k=1)
            McGreer_NF = LinearInterpolationFunction(self.McGreer_Redshift)
        else:
            raise ValueError(
                "You cannot use the McGreer prior likelihood with either less than 3 redshift or the redshift being directly evaluated.")

        McGreer_NF = np.clip(McGreer_NF, 0, 1)

        lnprob = 0
        if McGreer_NF > 0.06:
            lnprob = np.square((self.McGreer_Mean - McGreer_NF) / (self.McGreer_OneSigma))

        return lnprob


class LikelihoodGreig(LikelihoodBase):
    QSO_Redshift = 7.0842  # The redshift of the QSO

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self):
        with open(path.expanduser(path.join("~", '.py21cmmc', 'PriorData', "NeutralFractionsForPDF.out")),
                  'rb') as handle:
            self.NFValsQSO = pickle.loads(handle.read())

        with open(path.expanduser(path.join("~", '.py21cmmc', 'PriorData', "NeutralFractionPDF_SmallHII.out")),
                  'rb') as handle:
            self.PDFValsQSO = pickle.loads(handle.read())

        # Normalising the PDF to have a peak probability of unity (consistent with how other priors are treated)
        # Ultimately, this step does not matter
        normalisation = np.amax(self.PDFValsQSO)
        self.PDFValsQSO /= normalisation

    def computeLikelihood(self, ctx):
        """
        Constraints on the IGM neutral fraction at z = 7.1 from the IGM damping wing of ULASJ1120+0641
        Greig et al (2016) (http://arxiv.org/abs/1606.00441)
        """

        lightcone = ctx.get("output")

        Redshifts = lightcone.redshift
        AveNF = lightcone.average_nf

        # Interpolate the QSO damping wing PDF
        spline_QSODampingPDF = interpolate.splrep(self.NFValsQSO, self.PDFValsQSO, s=0)

        if self.QSO_Redshift in Redshifts:

            for i in range(len(Redshifts)):
                if Redshifts[i] == self.QSO_Redshift:
                    NF_QSO = AveNF[i]

        elif len(lightcone.redshift) > 2:

            # Check the redshift range input by the user to determine whether to interpolate or extrapolate the IGM
            # neutral fraction to the QSO redshift
            if self.QSO_Redshift < np.amin(Redshifts):
                # The QSO redshift is outside the range set by the user. Need to extrapolate the reionisation history
                # to obtain the neutral fraction at the QSO redshift

                # The linear interpolation/extrapolation function, taking as input the redshift supplied by the user
                # and the corresponding neutral fractions recovered for the specific EoR parameter set
                LinearInterpolationFunction = InterpolatedUnivariateSpline(Redshifts, AveNF, k=1)

                NF_QSO = LinearInterpolationFunction(self.QSO_Redshift)

            else:
                # The QSO redshift is within the range set by the user. Can interpolate the reionisation history to
                # obtain the neutral fraction at the QSO redshift
                if lightcone.params.n_redshifts == 3:
                    spline_reionisationhistory = interpolate.splrep(Redshifts, AveNF, k=2, s=0)
                else:
                    spline_reionisationhistory = interpolate.splrep(Redshifts, AveNF, s=0)

                NF_QSO = interpolate.splev(self.QSO_Redshift, spline_reionisationhistory, der=0)

        else:
            raise ValueError(
                """
                You cannot use the Greig prior likelihood with either less than 3 redshift or the redshift being 
                directly evaluated.""")

        # Ensure that the neutral fraction does not exceed unity, or go negative
        NF_QSO = np.clip(NF_QSO, 0, 1)

        QSO_Prob = interpolate.splev(NF_QSO, spline_QSODampingPDF, der=0)

        # Interpolating the PDF from the QSO damping wing might cause small negative values at the edges (i.e. x_HI ~ 0
        # or ~1) In case it is zero, or negative, set it to a very small non zero number (we take the log of this value,
        # it cannot be zero)
        if QSO_Prob <= 0.0:
            QSO_Prob = 0.000006

        # We work with the log-likelihood, therefore convert the IGM Damping wing PDF to log space
        QSO_Prob = -2. * np.log(QSO_Prob)

        lnprob = QSO_Prob
        return lnprob


class LikelihoodGlobal(LikelihoodBase):

    def __init__(self, FIXED_ERROR=False,
                 model_name="FaintGalaxies", mock_dir=None,
                 fixed_global_error=10.0, fixed_global_bandwidth=4.0, FrequencyMin=40.,
                 FrequencyMax=200, *args, **kwargs):

        # Run the LikelihoodBase init.
        super().__init__(*args, **kwargs)

        self.FIXED_ERROR = FIXED_ERROR

        self.model_name = model_name
        self.mock_dir = mock_dir or path.expanduser(path.join("~", '.py21cmmc'))

        self.fixed_global_error = fixed_global_error
        self.fixed_global_bandwidth = fixed_global_bandwidth
        self.FrequencyMin = FrequencyMin
        self.FrequencyMax = FrequencyMax

        self.obs_filename = path.join(self.mock_dir, "MockData", self.model_name, "GlobalSignal",
                                      self.model_name + "_GlobalSignal.txt")
        self.obs_error_filename = path.join(self.mock_dir, 'NoiseData', self.model_name, "GlobalSignal",
                                            'TotalError_%s_GlobalSignal_ConstantError_1000hr.txt' % self.model_name)

    def setup(self):
        """
        Contains any setup specific to this likelihood, that should only be performed once. Must save variables
        to the class.
        """

        # Read in the mock 21cm PS observation. Read in both k and the dimensionless PS.
        self.k_values = []
        self.PS_values = []

        mock = np.loadtxt(self.obs_filename, usecols=(0, 2))
        self.k_values.append(mock[:, 0])
        self.PS_values.append(mock[:, 1])

        self.Error_k_values = []
        self.PS_Error = []

        if not self.FIXED_ERROR:
            errs = np.loadtxt(self.obs_error_filename, usecols=(0, 1))

            self.Error_k_values.append(errs[:, 0])
            self.PS_Error.append(errs[:, 1])

        self.Error_k_values = np.array(self.Error_k_values)
        self.PS_Error = np.array(self.PS_Error)

    def computeLikelihood(self, ctx):
        """
        Compute the likelihood, given the lightcone output from 21cmFAST.
        """
        lightcone = ctx.get("output")

        # Get some useful variables out of the Lightcone box
        NumRedshifts = len(lightcone.redshift)
        Redshifts = lightcone.redshift
        AveTb = lightcone.average_Tb

        total_sum = 0

        # Converting the redshift to frequencies for the interpolation (must be in increasing order, it is by default
        # redshift which is decreasing)
        FrequencyValues_mock = np.zeros(len(self.k_values[0]))
        FrequencyValues_model = np.zeros(NumRedshifts)

        # Shouldn't need two, as they should be the same sampling. However, just done it for now
        for j in range(len(self.k_values[0])):
            FrequencyValues_mock[j] = ((2.99792e8) / (.2112 * (1. + self.k_values[0][j]))) / (1e6)

        for j in range(NumRedshifts):
            FrequencyValues_model[j] = ((2.99792e8) / (.2112 * (1. + Redshifts[j]))) / (1e6)

        splined_mock = interpolate.splrep(FrequencyValues_mock, self.PS_values[0], s=0)
        splined_model = interpolate.splrep(FrequencyValues_model, AveTb, s=0)

        FrequencyMin = self.FrequencyMin
        FrequencyMax = self.FrequencyMax

        if self.FIXED_ERROR:
            ErrorOnGlobal = self.fixed_global_error
            Bandwidth = self.fixed_global_bandwidth

            FrequencyBins = int(np.floor((FrequencyMax - FrequencyMin) / Bandwidth)) + 1

            for j in range(FrequencyBins):
                FrequencyVal = FrequencyMin + Bandwidth * j

                MockPS_val = interpolate.splev(FrequencyVal, splined_mock, der=0)

                ModelPS_val = interpolate.splev(FrequencyVal, splined_model, der=0)

                total_sum += np.square((MockPS_val - ModelPS_val) / ErrorOnGlobal)

        else:

            for j in range(len(self.Error_k_values[0])):

                FrequencyVal = ((2.99792e8) / (.2112 * (1. + self.Error_k_values[0][j]))) / (1e6)

                if FrequencyVal >= FrequencyMin and FrequencyVal <= FrequencyMax:
                    MockPS_val = interpolate.splev(FrequencyVal, splined_mock, der=0)

                    ModelPS_val = interpolate.splev(FrequencyVal, splined_model, der=0)

                    total_sum += np.square((MockPS_val - ModelPS_val) / self.PS_Error[0][j])

        return -0.5 * total_sum  # , nf_vals
