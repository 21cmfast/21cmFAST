
#ifndef _INITCONDITIONS_H
#define _INITCONDITIONS_H

#include <gsl/gsl_rng.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "OutputStructs.h"

int ComputeInitialConditions(unsigned long long random_seed, InitialConditions *boxes);

void seed_rng_threads(gsl_rng *rng_arr[], unsigned long long int seed);
void free_rng_threads(gsl_rng *rng_arr[]);

#ifdef __cplusplus
}
#endif
#endif
