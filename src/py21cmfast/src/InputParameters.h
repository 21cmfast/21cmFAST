#ifndef _PARAMSTRUCTURES_H
#define _PARAMSTRUCTURES_H

#include <stdbool.h>
// Since it is unguarded, make sure to ONLY include this file from here
#include "_inputparams_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

void Broadcast_struct_global_all(SimulationOptions *simulation_options,
                                 MatterOptions *matter_options, CosmoParams *cosmo_params,
                                 AstroParams *astro_params, AstroOptions *astro_options);
void Broadcast_struct_global_noastro(SimulationOptions *simulation_options,
                                     MatterOptions *matter_options, CosmoParams *cosmo_params);

#ifdef __cplusplus
}
#endif
#endif
