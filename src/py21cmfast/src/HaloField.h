/* Function Prototypes and definitions for the Halo Catalogs */
#ifndef _HALOFIELD_H
#define _HALOFIELD_H

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputeHaloField(float redshift_desc, float redshift, MatterParams *matter_params,
                     MatterFlags *matter_flags, CosmoParams *cosmo_params, InitialConditions *boxes,
                     unsigned long long int random_seed, HaloField *halos_desc, HaloField *halos);

#endif
