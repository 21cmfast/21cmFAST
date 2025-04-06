
#ifndef _INITCONDITIONS_H
#define _INITCONDITIONS_H

#include <gsl/gsl_rng.h>

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputeInitialConditions(unsigned long long random_seed, MatterParams *matter_params,
                             MatterFlags *matter_flags, CosmoParams *cosmo_params,
                             InitialConditions *boxes);

#endif
