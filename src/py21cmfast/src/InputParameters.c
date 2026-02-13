#include "InputParameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logger.h"

bool allocated_cosmo_tables = false;

void Broadcast_struct_global_all(SimulationOptions *simulation_options,
                                 MatterOptions *matter_options, CosmoParams *cosmo_params,
                                 AstroParams *astro_params, AstroOptions *astro_options,
                                 CosmoTables *cosmo_tables,
                                 OptionalQuantities *optional_quantities) {
    simulation_options_global = simulation_options;
    matter_options_global = matter_options;
    cosmo_params_global = cosmo_params;
    astro_params_global = astro_params;
    astro_options_global = astro_options;
    optional_quantities_global = optional_quantities;
    int n;
    if (!allocated_cosmo_tables) {
        cosmo_tables_global = malloc(sizeof(CosmoTables));
        cosmo_tables_global->ps_norm = cosmo_tables->ps_norm;
        cosmo_tables_global->USE_SIGMA_8 = cosmo_tables->USE_SIGMA_8;

        if (matter_options_global->POWER_SPECTRUM == 5) {
            n = cosmo_tables->transfer_density->size;
            cosmo_tables_global->transfer_density = malloc(sizeof(Table1D));
            cosmo_tables_global->transfer_density->size = n;
            cosmo_tables_global->transfer_density->x_values = malloc(n * sizeof(double));
            cosmo_tables_global->transfer_density->y_values = malloc(n * sizeof(double));

            n = cosmo_tables->transfer_vcb->size;
            cosmo_tables_global->transfer_vcb = malloc(sizeof(Table1D));
            cosmo_tables_global->transfer_vcb->size = n;
            cosmo_tables_global->transfer_vcb->x_values = malloc(n * sizeof(double));
            cosmo_tables_global->transfer_vcb->y_values = malloc(n * sizeof(double));
        }

        allocated_cosmo_tables = true;
        LOG_DEBUG("Allocated memory for cosmo_tables_global");
    }
    // NOTE: While this is somewhat wasteful (re-copying EVERY time, though not re-allocating every
    // time), it's essentially impossible to know the user's mind and when they might want to
    // refresh the transfer density to a new cosmology, so it's better to just assume every time
    // that they might have a different transfer function.
    if (matter_options_global->POWER_SPECTRUM == 5) {
        n = cosmo_tables->transfer_density->size;
        memcpy(cosmo_tables_global->transfer_density->x_values,
               cosmo_tables->transfer_density->x_values, n * sizeof(double));
        memcpy(cosmo_tables_global->transfer_density->y_values,
               cosmo_tables->transfer_density->y_values, n * sizeof(double));

        n = cosmo_tables->transfer_vcb->size;
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
    if (allocated_cosmo_tables) {
        if (matter_options_global->POWER_SPECTRUM == 5) {
            free(cosmo_tables_global->transfer_density->x_values);
            free(cosmo_tables_global->transfer_density->y_values);
            free(cosmo_tables_global->transfer_density);
            free(cosmo_tables_global->transfer_vcb->x_values);
            free(cosmo_tables_global->transfer_vcb->y_values);
            free(cosmo_tables_global->transfer_vcb);
        }
        free(cosmo_tables_global);
        allocated_cosmo_tables = false;
        LOG_DEBUG("Freed cosmo_tables_global");
    }
}

/*GLOBAL INPUT STRUCT DEFINITION*/
SimulationOptions *simulation_options_global;
MatterOptions *matter_options_global;
CosmoParams *cosmo_params_global;
AstroParams *astro_params_global;
AstroOptions *astro_options_global;
CosmoTables *cosmo_tables_global;
OptionalQuantities *optional_quantities_global;

// data paths, wisdoms, etc
ConfigSettings config_settings;
