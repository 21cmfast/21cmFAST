#ifndef _PARAMSTRUCTURES_H
#define _PARAMSTRUCTURES_H

#include <stdbool.h>
// since ffi.cdef() cannot include directives, we store the types and globals in another file
//   Since it is unguarded, make sure to ONLY include this file from here
#include "_inputparams_wrapper.h"

void Broadcast_struct_global_all(SimulationOptions *simulation_options,
                                 MatterOptions *matter_options, CosmoParams *cosmo_params,
                                 AstroParams *astro_params, AstroOptions *astro_options,
                                 CosmoTables *cosmo_tables,
                                 OptionalQuantities *optional_quantities);
void Broadcast_struct_global_noastro(SimulationOptions *simulation_options,
                                     MatterOptions *matter_options, CosmoParams *cosmo_params);

#endif
