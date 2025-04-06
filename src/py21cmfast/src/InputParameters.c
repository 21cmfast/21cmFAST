#include "InputParameters.h"

void Broadcast_struct_global_all(MatterParams *matter_params, MatterFlags *matter_flags,
                                 CosmoParams *cosmo_params, AstroParams *astro_params,
                                 AstroFlags *astro_flags) {
    matter_params_global = matter_params;
    matter_flags_global = matter_flags;
    cosmo_params_global = cosmo_params;
    astro_params_global = astro_params;
    astro_flags_global = astro_flags;
}

void Broadcast_struct_global_noastro(MatterParams *matter_params, MatterFlags *matter_flags,
                                     CosmoParams *cosmo_params) {
    matter_params_global = matter_params;
    matter_flags_global = matter_flags;
    cosmo_params_global = cosmo_params;
}

/*GLOBAL INPUT STRUCT DEFINITION*/
MatterParams *matter_params_global;
MatterFlags *matter_flags_global;
CosmoParams *cosmo_params_global;
AstroParams *astro_params_global;
AstroFlags *astro_flags_global;

// data paths, wisdoms, etc
ConfigSettings config_settings;
