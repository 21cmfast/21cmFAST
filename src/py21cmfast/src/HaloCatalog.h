/* Function Prototypes and definitions for the Halo Catalogs */
#ifndef _HALOFIELD_H
#define _HALOFIELD_H

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputeHaloCatalog(float redshift_desc, float redshift, InitialConditions *boxes,
                       unsigned long long int random_seed, HaloCatalog *halos_desc,
                       HaloCatalog *halos);

#endif
