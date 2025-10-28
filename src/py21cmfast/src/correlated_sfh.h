
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>

#ifndef CORRELATED_SFH_H
#define CORRELATED_SFH_H

void initialise_psd_corrfunc_tables(double tau, double tau_prev);
void fill_covar_from_tables(double tau, gsl_matrix *curr_cov, gsl_matrix *prev_cov,
                            gsl_matrix *cross_cov);
void fill_covar_analytic(double tau, double tau_prev, gsl_matrix *curr_cov, gsl_matrix *prev_cov,
                         gsl_matrix *cross_cov);
void eval_sfh_moments(gsl_matrix *prev_cov, gsl_matrix *curr_cov, gsl_matrix *cross_cov,
                      gsl_matrix *out_chol_cov, gsl_matrix *out_mean_correction);
void sample_correlated_sfh(gsl_rng *rng, double prev_values[3], gsl_matrix *L_cov,
                           gsl_matrix *mean_corr, double out_values[3]);
void free_sfh_correlation();

#endif
