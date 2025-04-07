/* Function Prototypes and definitions for the Halo Catalogs */
#ifndef _HALOFIELD_H
#define _HALOFIELD_H

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputeHaloField(float redshift_desc, float redshift, UserParams *user_params,
                     CosmoParams *cosmo_params, AstroParams *astro_params,
                     FlagOptions *flag_options, InitialConditions *boxes,
                     unsigned long long int random_seed, HaloField *halos_desc, HaloField *halos);

#endif
