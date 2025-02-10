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
    .SMOOTH_EVOLVED_DENSITY_FIELD = 0, //PerturbedField, smooth field after perturb (delete)
    .R_smooth_density = 0.2, //With above, radius for smoothing (delete)

    .OPTIMIZE = 0, //HaloField, excluding zones around existing halos for speed (delete?)
    .R_OVERLAP_FACTOR = 1., //HaloField, exclude cells within X*radius for optimisation (delete?)
    .OPTIMIZE_MIN_MASS = 1e11, //don't optimize below this mass (delete?)

    .CRIT_DENS_TRANSITION = 1.2, //Gauss-Legendre does poorly at high delta, switch to GSL-QAG here (define?)

    .NU_X_BAND_MAX = 2000.0, //Used in SFR -> Lx conversion factor (define)
    .NU_X_MAX = 10000.0, //Limit for nu integrals (define)

    .R_BUBBLE_MIN = 0.620350491, //minimum bubble radius in cMpc, used as a ceiling for the last radius (define?)
};
