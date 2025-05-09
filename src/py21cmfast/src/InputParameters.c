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

void set_external_table_path(ConfigSettings *params, const char *value) {
    if (params->external_table_path != 0) {
        free(params->external_table_path);
    }
    params->external_table_path = (char *)malloc(strlen(value) + 1);
    strcpy(params->external_table_path, value);
}
char *get_external_table_path(ConfigSettings *params) {
    return params->external_table_path ? params->external_table_path : "";
}
void set_wisdoms_path(ConfigSettings *params, const char *value) {
    if (params->wisdoms_path != 0) {
        free(params->wisdoms_path);
    }
    params->wisdoms_path = (char *)malloc(strlen(value) + 1);
    strcpy(params->wisdoms_path, value);
}
char *get_wisdoms_path(ConfigSettings *params) {
    return params->wisdoms_path ? params->wisdoms_path : "";
}

// data paths, wisdoms, etc
ConfigSettings config_settings;
