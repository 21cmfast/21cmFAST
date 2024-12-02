#ifndef _21CMFRNG_H
#define _21CMFRNG_H

#include <gsl/gsl_rng.h>

void seed_rng_threads(gsl_rng * rng_arr[], unsigned long long int seed);
void seed_rng_threads_fast(gsl_rng * rng_arr[], unsigned long long int seed);
void free_rng_threads(gsl_rng * rng_arr[]);

#endif
