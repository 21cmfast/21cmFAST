#include "InputParameters.h"

void Broadcast_struct_global_all(UserParams *user_params,
                                 CosmoParams *cosmo_params,
                                 AstroParams *astro_params,
                                 FlagOptions *flag_options) {
  user_params_global = user_params;
  cosmo_params_global = cosmo_params;
  astro_params_global = astro_params;
  flag_options_global = flag_options;
}

void Broadcast_struct_global_noastro(UserParams *user_params,
                                     CosmoParams *cosmo_params) {
  user_params_global = user_params;
  cosmo_params_global = cosmo_params;
}

/*GLOBAL INPUT STRUCT DEFINITION*/
UserParams *user_params_global;
CosmoParams *cosmo_params_global;
AstroParams *astro_params_global;
FlagOptions *flag_options_global;

// data paths, wisdoms, etc
ConfigSettings config_settings;
