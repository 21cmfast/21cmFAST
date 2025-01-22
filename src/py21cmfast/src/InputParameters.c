#include "InputParameters.h"

void Broadcast_struct_global_all(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options){
    user_params_global = user_params;
    cosmo_params_global = cosmo_params;
    astro_params_global = astro_params;
    flag_options_global = flag_options;
}

void Broadcast_struct_global_noastro(UserParams *user_params, CosmoParams *cosmo_params){
    user_params_global = user_params;
    cosmo_params_global = cosmo_params;
}

/*GLOBAL INPUT STRUCT DEFINITION*/
UserParams *user_params_global;
CosmoParams *cosmo_params_global;
AstroParams *astro_params_global;
FlagOptions *flag_options_global;

// These need to be removed, I am commenting below roughly where/when
//  they are used, and my tentative decision on what to do with them
GlobalParams global_params = {
    .ALPHA_UVB = 5.0, // IonisationBox, for Gamma (move to AstroParams)
    .EVOLVE_DENSITY_LINEARLY = 0, //PerturnbedField, uses linear growth instead of 1/2LPT (move to UserParams)
    .SMOOTH_EVOLVED_DENSITY_FIELD = 0, //PerturbedField, smooth field after perturb (delete)
    .R_smooth_density = 0.2, //With above, radius for smoothing (delete)
    .HII_ROUND_ERR = 1e-5, // IonisationBox, Nion threshold to run the ES (move to define)
    .FIND_BUBBLE_ALGORITHM = 2, // IonisationBox central pixel vs sphere (move to UserParams)
    .MAX_DVDR = 0.2, //BrightnessTemp, maximum velocity gradient for RSDS (move to AstroParams?)
    .DELTA_R_HII_FACTOR = 1.1, //IonisationBox, radius factor for reion ES (move to AstroParams)
    .DELTA_R_FACTOR = 1.1, //HaloField, radius factor for DexM halo ES (move to UserParams)

    .HII_FILTER = 0, //IonisationBox, filter for reion ES (move to AstroParams)
    .HALO_FILTER = 0, //HaloField, filter for halo ES (move to UserParams)
    .HEAT_FILTER = 0, //SpinTemperature, filter for Ts model (move to AstroParams)
    .FILTER = 0, //Filter for calculating sigma(M) and R <---> M conversions Unlike others this is 0: Tophat, 1: Gaussian

    .INITIAL_REDSHIFT = 300., //PerturbField, initial redshift for moving the particles (move to UserParams)

    .OPTIMIZE = 0, //HaloField, excluding zones around existing halos for speed (delete?)
    .R_OVERLAP_FACTOR = 1., //HaloField, exclude cells within X*radius for optimisation (delete?)
    .OPTIMIZE_MIN_MASS = 1e11, //don't optimize below this mass (delete?)

    .CRIT_DENS_TRANSITION = 1.2, //Gauss-Legendre does poorly at high delta, switch to GSL-QAG here (define?)
    .MIN_DENSITY_LOW_LIMIT = 9e-8, //lower limit for extrapolated density in ionised temperatures (define)

    //all the z-photoncons parameters
    //TODO: walk through the model again, it's confusing
    .RecombPhotonCons = 0,
    .PhotonConsStart = 0.995,
    .PhotonConsEnd = 0.3,
    .PhotonConsAsymptoteTo = 0.01,
    .PhotonConsEndCalibz = 3.5,
    .PhotonConsSmoothing = 1,

    .CLUMPING_FACTOR = 2., //SpinTemperature, for recombinations in neutral IGM
    .R_XLy_MAX = 500., //SpinTemperature, max radius for ES (Move to AstroParams)
    .NUM_FILTER_STEPS_FOR_Ts = 40, //SpinTemperature, self-explanatory (move to AstroParams)

    .Pop = 2, //Only used in lyman alpha spectral calculations, when minihalos are off (delete)
    .Pop2_ion = 5000, //Move to AstroParams
    .Pop3_ion = 44021, // Move to AstroParams

    .NU_X_BAND_MAX = 2000.0, //Used in SFR -> Lx conversion factor (define)
    .NU_X_MAX = 10000.0, //Limit for nu integrals (define)

    .P_CUTOFF = 0, //Used in EH power spectrum and with WDM in setting minimum mass, seems inconsistent (delete?)
    .M_WDM = 2, //Used in setting minimum mass if P_CUTOFF is True. (delete?)
    .g_x = 1.5, //Used in setting minimum mass if P_CUTOFF is True. (delete?)

     //Move These to CosmoParams
    .OMn = 0.0,
    .OMk = 0.0,
    .OMr = 8.6e-5,
    .OMtot = 1.0,
    .Y_He = 0.245,
    .wl = -1.0,

    //Adjustments to SMT Barrier so that DexM fits the mass function (Move to UserParams?)
    .SHETH_b = 0.15, //In the literature this is 0.485 (RM08) or 0.5 (SMT01) or 0.34 (Barkana+01)
    .SHETH_c = 0.05, //In the literature this is 0.615 (RM08) or 0.6 (SMT01) or 0.81 (Barkana+01)

    .Zreion_HeII = 3.0, //used in tau computation, move to tau arguments
    .R_BUBBLE_MIN = 0.620350491, //minimum bubble radius in cMpc, used as a ceiling for the last radius (define?)
    .M_MIN_INTEGRAL = 1e5, //minimum mass for the integrals of the mass function (define?)
    .M_MAX_INTEGRAL = 1e16, //maximum mass for the integrals of the mass function (define?)

    .T_RE = 2e4, //reionisation temperature, move to AstroParams

    .VAVG=25.86, //Move to AstroParams, when FlagOptions.FIX_VCB_AVG is true, this is the value

    .USE_ADIABATIC_FLUCTUATIONS = 1, //For first Ts Box, (fix to True)
};
