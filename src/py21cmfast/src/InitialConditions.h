
#ifndef _INITCONDITIONS_H
#define _INITCONDITIONS_H

#include <gsl/gsl_rng.h>

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputeInitialConditions(unsigned long long random_seed,
                             UserParams *user_params, CosmoParams *cosmo_params,
                             InitialConditions *boxes);

#endif
