#ifndef _21CMFRNG_H
#define _21CMFRNG_H

#include <gsl/gsl_rng.h>

#include "indexing.h"

void seed_rng_threads(gsl_rng *rng_arr[], random_huge seed);
void seed_rng_threads_fast(gsl_rng *rng_arr[], random_huge seed);
void free_rng_threads(gsl_rng *rng_arr[]);

#endif
