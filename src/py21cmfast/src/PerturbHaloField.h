#ifndef _PTHALOS_H
#define _PTHALOS_H

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputePerturbHaloField(float redshift, UserParams *user_params, CosmoParams *cosmo_params,
                            AstroParams *astro_params, FlagOptions *flag_options,
                            InitialConditions *boxes, HaloField *halos,
                            PerturbHaloField *halos_perturbed);

#endif
