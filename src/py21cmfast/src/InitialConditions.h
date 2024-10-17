
#ifndef _INITCONDITIONS_H
#define _INITCONDITIONS_H

#include "InputParameters.h"
#include "OutputStructs.h"
#include <gsl/gsl_rng.h>

#ifdef __cplusplus
extern "C" {
#endif
int ComputeInitialConditions(
    unsigned long long random_seed, UserParams *user_params,
    CosmoParams *cosmo_params, InitialConditions *boxes
);

void seed_rng_threads(gsl_rng * rng_arr[], unsigned long long int seed);
void free_rng_threads(gsl_rng * rng_arr[]);

#ifdef __cplusplus
}
#endif
#endif
