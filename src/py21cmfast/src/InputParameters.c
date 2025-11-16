#include "InputParameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void Broadcast_struct_global_all(SimulationOptions *simulation_options,
                                 MatterOptions *matter_options, CosmoParams *cosmo_params,
                                 AstroParams *astro_params, AstroOptions *astro_options,
                                 CosmoTables *cosmo_tables) {
    simulation_options_global = simulation_options;
    matter_options_global = matter_options;
    cosmo_params_global = cosmo_params;
    astro_params_global = astro_params;
    astro_options_global = astro_options;
    if (matter_options_global->POWER_SPECTRUM == 5 && cosmo_tables_global == NULL) {
        cosmo_tables_global = malloc(sizeof(CosmoTables));
        int n;

        n = cosmo_tables->transfer_density->size;
        cosmo_tables_global->transfer_density = malloc(sizeof(Table1D));
        cosmo_tables_global->transfer_density->size = n;
        cosmo_tables_global->transfer_density->x_values = malloc(n * sizeof(double));
        cosmo_tables_global->transfer_density->y_values = malloc(n * sizeof(double));
        memcpy(cosmo_tables_global->transfer_density->x_values,
               cosmo_tables->transfer_density->x_values, n * sizeof(double));
        memcpy(cosmo_tables_global->transfer_density->y_values,
               cosmo_tables->transfer_density->y_values, n * sizeof(double));

        n = cosmo_tables->transfer_vcb->size;
        cosmo_tables_global->transfer_vcb = malloc(sizeof(Table1D));
        cosmo_tables_global->transfer_vcb->size = n;
        cosmo_tables_global->transfer_vcb->x_values = malloc(n * sizeof(double));
        cosmo_tables_global->transfer_vcb->y_values = malloc(n * sizeof(double));
        memcpy(cosmo_tables_global->transfer_vcb->x_values, cosmo_tables->transfer_vcb->x_values,
               n * sizeof(double));
        memcpy(cosmo_tables_global->transfer_vcb->y_values, cosmo_tables->transfer_vcb->y_values,
               n * sizeof(double));
    }
}

void Broadcast_struct_global_noastro(SimulationOptions *simulation_options,
                                     MatterOptions *matter_options, CosmoParams *cosmo_params) {
    simulation_options_global = simulation_options;
    matter_options_global = matter_options;
    cosmo_params_global = cosmo_params;
}

void Free_cosmo_tables_global() {
    if (matter_options_global->POWER_SPECTRUM == 5 && cosmo_tables_global != NULL) {
        free(cosmo_tables_global->transfer_density->x_values);
        free(cosmo_tables_global->transfer_density->y_values);
        free(cosmo_tables_global->transfer_density);
        free(cosmo_tables_global->transfer_vcb->x_values);
        free(cosmo_tables_global->transfer_vcb->y_values);
        free(cosmo_tables_global->transfer_vcb);
        free(cosmo_tables_global);
        cosmo_tables_global = NULL;
    }
}

/*GLOBAL INPUT STRUCT DEFINITION*/
SimulationOptions *simulation_options_global;
MatterOptions *matter_options_global;
CosmoParams *cosmo_params_global;
AstroParams *astro_params_global;
AstroOptions *astro_options_global;
CosmoTables *cosmo_tables_global = NULL;

// data paths, wisdoms, etc
ConfigSettings config_settings;
