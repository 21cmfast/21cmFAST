
#include <gsl/gsl_matrix.h>

#ifndef CORRELATED_SFH_H
#define CORRELATED_SFH_H

void initialise_sfh_correlation(double z, double z_prev, double z_prev_2);
void eval_sfh_moments(double tau, gsl_matrix *out_chol_cov, gsl_matrix *out_mean_correction);
void free_sfh_correlation();

#endif
