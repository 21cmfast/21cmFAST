
#ifndef _INITCONDITIONS_H
#define _INITCONDITIONS_H

#include "InputParameters.h"
#include "OutputStructs.h"
#include <gsl/gsl_rng.h>

int ComputeInitialConditions(
    unsigned long long random_seed, UserParams *user_params,
    CosmoParams *cosmo_params, InitialConditions *boxes
);

//TODO: these seeding functions should probably be somewhere else
//  Possibly make an rng.c/h
void seed_rng_threads(gsl_rng * rng_arr[], unsigned long long int seed);
void seed_rng_threads_fast(gsl_rng * rng_arr[], unsigned long long int seed);
void free_rng_threads(gsl_rng * rng_arr[]);

#endif
