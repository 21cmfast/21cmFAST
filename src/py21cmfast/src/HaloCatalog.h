/* Function Prototypes and definitions for the Halo Catalogs */
#ifndef _HALOCATALOG_H
#define _HALOCATALOG_H

#include "InputParameters.h"
#include "OutputStructs.h"

#ifdef __cplusplus
extern "C" {
#endif

int ComputeHaloCatalog(float redshift_desc, float redshift, InitialConditions *boxes,
                       unsigned long long int random_seed, HaloCatalog *halos_desc,
                       HaloCatalog *halos);

// CUDA utility function (defined in HaloField.cu)
void updateGlobalParams(SimulationOptions *h_simulation_options, MatterOptions *h_matter_options,
                        CosmoParams *h_cosmo_params, AstroParams *h_astro_params);

#ifdef __cplusplus
}
#endif
#endif
