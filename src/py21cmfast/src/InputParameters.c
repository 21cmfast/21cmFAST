#include "InputParameters.h"

#include <stdlib.h>
#include <string.h>

void Broadcast_struct_global_all(SimulationOptions *simulation_options,
                                 MatterOptions *matter_options, CosmoParams *cosmo_params,
                                 AstroParams *astro_params, AstroOptions *astro_options) {
    simulation_options_global = simulation_options;
    matter_options_global = matter_options;
    cosmo_params_global = cosmo_params;
    astro_params_global = astro_params;
    astro_options_global = astro_options;
}

void Broadcast_struct_global_noastro(SimulationOptions *simulation_options,
                                     MatterOptions *matter_options, CosmoParams *cosmo_params) {
    simulation_options_global = simulation_options;
    matter_options_global = matter_options;
    cosmo_params_global = cosmo_params;
}

/*GLOBAL INPUT STRUCT DEFINITION*/
SimulationOptions *simulation_options_global;
MatterOptions *matter_options_global;
CosmoParams *cosmo_params_global;
AstroParams *astro_params_global;
AstroOptions *astro_options_global;

// data paths, wisdoms, etc
ConfigSettings config_settings;
