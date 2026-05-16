#ifndef CORRELATED_SFH_H
#define CORRELATED_SFH_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <stdbool.h>

void initialise_sfh_structs(double z0, double z1, double z2, bool conditioned);
int test_sfh_corr(double z0, double z1, double z2);
void sample_correlated_sfh(gsl_rng *rng, double prev_values[3], double out_values[3]);
void get_current_vars(double out[3]);
void cleanup_sfh_structs();

#endif
