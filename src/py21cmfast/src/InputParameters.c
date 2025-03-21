#include <stdlib.h>
#include <string.h>
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

GlobalParams global_params = {
    .ALPHA_UVB = 5.0,
    .EVOLVE_DENSITY_LINEARLY = 0,
    .SMOOTH_EVOLVED_DENSITY_FIELD = 0,
    .R_smooth_density = 0.2,
    .HII_ROUND_ERR = 1e-5,
    .FIND_BUBBLE_ALGORITHM = 2,
    .N_POISSON = 5,
    .T_USE_VELOCITIES = 1,
    .MAX_DVDR = 0.2,
    .DELTA_R_HII_FACTOR = 1.1,
    .DELTA_R_FACTOR = 1.1,
    .HII_FILTER = 0,
    .INITIAL_REDSHIFT = 300.,
    .R_OVERLAP_FACTOR = 1.,
    .DELTA_CRIT_MODE = 1,
    .HALO_FILTER = 0,
    .OPTIMIZE = 0,
    .OPTIMIZE_MIN_MASS = 1e11,


    .CRIT_DENS_TRANSITION = 1.2,
    .MIN_DENSITY_LOW_LIMIT = 9e-8,

    .RecombPhotonCons = 0,
    .PhotonConsStart = 0.995,
    .PhotonConsEnd = 0.3,
    .PhotonConsAsymptoteTo = 0.01,
    .PhotonConsEndCalibz = 3.5,
    .PhotonConsSmoothing = 1,

    .HEAT_FILTER = 0,
    .CLUMPING_FACTOR = 2.,
    .Z_HEAT_MAX = 35.0,
    .R_XLy_MAX = 500.,
    .NUM_FILTER_STEPS_FOR_Ts = 40,
    .ZPRIME_STEP_FACTOR = 1.02,
    .TK_at_Z_HEAT_MAX = -1,
    .XION_at_Z_HEAT_MAX = -1,
    .Pop = 2,
    .Pop2_ion = 5000,
    .Pop3_ion = 44021,

    .NU_X_BAND_MAX = 2000.0,
    .NU_X_MAX = 10000.0,

    .NBINS_LF = 100,

    .P_CUTOFF = 0,
    .M_WDM = 2,
    .g_x = 1.5,
    .OMn = 0.0,
    .OMk = 0.0,
    .OMr = 8.6e-5,
    .OMtot = 1.0,
    .Y_He = 0.245,
    .wl = -1.0,
    .SHETH_b = 0.15, //In the literature this is 0.485 (RM08) or 0.5 (SMT01) or 0.34 (Barkana+01) Master 21cmFAST currently has 0.15
    .SHETH_c = 0.05, //In the literature this is 0.615 (RM08) or 0.6 (SMT01) or 0.81 (Barkana+01) Master 21cmFAST currently has 0.05
    .Zreion_HeII = 3.0,
    .FILTER = 0,
    .R_BUBBLE_MIN = 0.620350491,
    .M_MIN_INTEGRAL = 1e5,
    .M_MAX_INTEGRAL = 1e16,

    .T_RE = 2e4,

    .VAVG=25.86,

    .USE_ADIABATIC_FLUCTUATIONS = 1,
};

void set_external_table_path(GlobalParams *params, const char *value) {
  if (params->external_table_path != 0) {
      free(params->external_table_path);
  }
  params->external_table_path = (char *)malloc(strlen(value) + 1);
  strcpy(params->external_table_path, value);
}
char* get_external_table_path(GlobalParams *params) {
  return params->external_table_path ? params->external_table_path : "";
}
void set_wisdoms_path(GlobalParams *params, const char *value) {
  if (params->wisdoms_path != 0) {
      free(params->wisdoms_path);
  }
  params->wisdoms_path = (char *)malloc(strlen(value) + 1);
  strcpy(params->wisdoms_path, value);
}
char* get_wisdoms_path(GlobalParams *params) {
  return params->wisdoms_path ? params->wisdoms_path : "";
}
