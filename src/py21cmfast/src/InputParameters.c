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
    .OPTIMIZE = 0, //HaloField, excluding zones around existing halos for speed (delete?)
    .R_OVERLAP_FACTOR = 1., //HaloField, exclude cells within X*radius for optimisation (delete?)
    .OPTIMIZE_MIN_MASS = 1e11, //don't optimize below this mass (delete?)
};
